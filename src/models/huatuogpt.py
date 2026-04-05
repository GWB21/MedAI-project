"""
HuatuoGPT-Vision-7B wrapper.

Follows the official evaluation code from:
    - repos/HuatuoGPT-Vision/cli.py  (HuatuoChatbot.inference, get_image_tensors)
    - repos/HuatuoGPT-Vision/eval.py (llava_prompt format)
    - repos/HuatuoGPT-Vision/scorer.py (match_choice3 answer parsing)

Setup:
    1. Clone the HuatuoGPT-Vision repo:
       git clone https://github.com/FreedomIntelligence/HuatuoGPT-Vision.git /tmp/HuatuoGPT-Vision
    2. Download weights (auto via HuggingFace cache):
       python -c "from huggingface_hub import snapshot_download; snapshot_download('FreedomIntelligence/HuatuoGPT-Vision-7B')"

Ref:
    https://github.com/FreedomIntelligence/HuatuoGPT-Vision
    https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B
"""

import re
import sys
import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, Optional, List
from difflib import SequenceMatcher

from .base_model import BaseMedVQAModel, ModelOutput

_DEFAULT_REPO = os.path.join(os.path.dirname(__file__), "..", "..", "repos", "HuatuoGPT-Vision")
HUATUOGPT_REPO_PATH = os.environ.get("HUATUOGPT_REPO_PATH", os.path.abspath(_DEFAULT_REPO))


# ---------------------------------------------------------------------------
# Answer parsing – faithful port of scorer.py match_choice3
# ---------------------------------------------------------------------------
_MATCH_CHOICE3_RE = re.compile(
    r"(is |是|\*|\W|\ |\(|为|^|'|\"|#)(?![aA] )([abcdefghijklmnABCDEFGHIJKLMN])(\W|$)"
)


def _match_choice3(
    text: str,
    options: List[str],
    choice_letters: List[str] = None,
) -> Optional[str]:
    """
    Port of HuatuoGPT-Vision scorer.py match_choice3.

    1. Regex match for a choice letter.
    2. Fallback: find most similar option text.
    3. Fallback: first character.
    """
    if choice_letters is None:
        choice_letters = ["A", "B", "C", "D"]

    text = text.strip()
    if not text:
        return choice_letters[0] if choice_letters else None

    # Step 1: regex match
    match = _MATCH_CHOICE3_RE.search(text)
    if match:
        letter = match.group(2).upper()
        if letter in choice_letters:
            return letter

    # Step 2: most similar option text
    if options:
        best_ratio = 0.0
        best_idx = 0
        for idx, opt in enumerate(options):
            ratio = SequenceMatcher(None, text.lower(), opt.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = idx
        if best_ratio > 0.0 and best_idx < len(choice_letters):
            return choice_letters[best_idx]

    # Step 3: first character
    first = text[0].upper()
    if first in choice_letters:
        return first

    return choice_letters[0] if choice_letters else None


# ---------------------------------------------------------------------------
# Official generation kwargs from cli.py
# ---------------------------------------------------------------------------
_OFFICIAL_GEN_KWARGS = {
    "do_sample": True,
    "max_new_tokens": 256,
    "temperature": 0.2,
    "repetition_penalty": 1.2,
}


class HuatuoGPTVisionModel(BaseMedVQAModel):

    MODEL_ID = "FreedomIntelligence/HuatuoGPT-Vision-7B"

    def __init__(self, repo_path: str = HUATUOGPT_REPO_PATH):
        self.repo_path = repo_path
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.device = None
        self._precision = "bf16"

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(self, device: str = "cuda") -> None:
        if not os.path.exists(self.repo_path):
            raise FileNotFoundError(
                f"HuatuoGPT-Vision repo not found at {self.repo_path}. "
                f"Clone it: git clone https://github.com/FreedomIntelligence/HuatuoGPT-Vision.git {self.repo_path}"
            )

        # Add HuatuoGPT-Vision repo to path for its custom llava module
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)

        self.device = device

        # Resolve model dir from HF cache
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo_id=self.MODEL_ID)

        # Load using HuatuoGPT's own LlavaQwen2ForCausalLM
        from llava.model.language_model.llava_qwen2 import LlavaQwen2ForCausalLM
        from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor

        # init_vision_encoder_from_ckpt=False to avoid meta device conflict with transformers>=5.x
        self.model = LlavaQwen2ForCausalLM.from_pretrained(
            model_dir,
            init_vision_encoder_from_ckpt=False,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Manually load vision tower (bypasses transformers meta device issue)
        vision_tower = self.model.get_vision_tower()
        vit_path = os.path.join(model_dir, "vit", "clip_vit_large_patch14_336")
        if os.path.exists(vit_path):
            vision_tower.vision_tower = CLIPVisionModel.from_pretrained(vit_path)
            vision_tower.image_processor = CLIPImageProcessor.from_pretrained(vit_path)
        else:
            vision_tower.vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
            vision_tower.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        vision_tower.is_loaded = True
        vision_tower.vision_tower.requires_grad_(False)
        vision_tower.to(dtype=torch.bfloat16, device=self.device)
        self.image_processor = vision_tower.image_processor

        self.model.eval()
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Image preprocessing  (cli.py get_image_tensors)
    # ------------------------------------------------------------------
    def _process_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Pad to square with image_mean background colour, then CLIP process."""
        processor = self.image_processor

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        image = expand2square(pil_image, tuple(int(x * 255) for x in processor.image_mean))
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image.to(self.device, dtype=torch.bfloat16)

    # ------------------------------------------------------------------
    # Tokenize with image tokens  (cli.py inference)
    # ------------------------------------------------------------------
    def _tokenize_with_image(self, text: str) -> torch.Tensor:
        from llava.constants import IMAGE_TOKEN_INDEX

        prompt_chunks = [
            self.tokenizer(chunk, add_special_tokens=False).input_ids
            for chunk in text.split("<image>")
        ]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if (
            len(prompt_chunks) > 0
            and len(prompt_chunks[0]) > 0
            and prompt_chunks[0][0] == self.tokenizer.bos_token_id
        ):
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [IMAGE_TOKEN_INDEX] * (offset + 1)):
            input_ids.extend(x[offset:])

        return torch.tensor(input_ids, dtype=torch.long)

    # ------------------------------------------------------------------
    # Build prompt  (cli.py conversation format)
    # ------------------------------------------------------------------
    def _build_prompt(self, text: str) -> str:
        """HuatuoGPT conversation format with image placeholder."""
        return f"<image>\n<|user|>\n{text}\n<|assistant|>\n"

    # ------------------------------------------------------------------
    # Inference  (follows cli.py HuatuoChatbot.inference exactly)
    # ------------------------------------------------------------------
    def inference(self, image: np.ndarray, prompt: str, max_new_tokens: int = 512) -> ModelOutput:
        pil_image = Image.fromarray(image)
        image_tensor = self._process_image(pil_image).unsqueeze(0)

        full_prompt = self._build_prompt(prompt)
        input_ids = self._tokenize_with_image(full_prompt).unsqueeze(0).to(self.device)

        # --- Step 1: generate() with official gen_kwargs ---
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,  # positional (keyword causes NoneType in LLaVA-based models)
                images=image_tensor,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                **_OFFICIAL_GEN_KWARGS,
            )

        # Decode: full output = generated text only (image token expansion
        # makes output shorter than input_ids, so just decode everything)
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # --- Step 2: parse answer using match_choice3 logic ---
        # Try to extract option texts from the prompt for similarity fallback
        options = self._extract_options_from_prompt(prompt)
        parsed = _match_choice3(generated_text, options)
        parse_success = parsed is not None
        if not parse_success:
            parsed = "A"  # final fallback

        # --- Step 3: logit extraction via separate forward pass ---
        logits = self.get_choice_logits(image, prompt)

        return ModelOutput(
            raw_text=generated_text,
            parsed_answer=parsed,
            logits=logits,
            parse_success=parse_success,
        )

    # ------------------------------------------------------------------
    # Helper: extract option texts from prompt for similarity fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_options_from_prompt(prompt: str) -> List[str]:
        """Pull out option texts like 'A. foo' from the prompt."""
        options = []
        for letter in ("A", "B", "C", "D"):
            pattern = re.compile(rf"^{letter}\.\s*(.+)$", re.MULTILINE)
            m = pattern.search(prompt)
            if m:
                options.append(m.group(1).strip())
        return options

    # ------------------------------------------------------------------
    # Logit extraction  (separate forward pass for CSV logging)
    # ------------------------------------------------------------------
    def get_choice_logits(self, image: np.ndarray, prompt: str) -> Dict[str, float]:
        pil_image = Image.fromarray(image)
        image_tensor = self._process_image(pil_image).unsqueeze(0)

        full_prompt = self._build_prompt(prompt)
        input_ids = self._tokenize_with_image(full_prompt).unsqueeze(0).to(self.device)

        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    images=image_tensor,
                    return_dict=True,
                )
                next_logits = outputs.logits[:, -1, :]

            choice_logits = {}
            for choice in ("A", "B", "C", "D"):
                token_id = self.tokenizer.encode(choice, add_special_tokens=False)[0]
                choice_logits[choice] = next_logits[0, token_id].item()
            return choice_logits
        except Exception:
            return {"A": float("nan"), "B": float("nan"), "C": float("nan"), "D": float("nan")}

    @property
    def name(self) -> str:
        return "huatuogpt"

    @property
    def precision(self) -> str:
        return self._precision
