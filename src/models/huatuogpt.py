"""
HuatuoGPT-Vision-7B wrapper.

Setup:
    1. Clone the HuatuoGPT-Vision repo:
       git clone https://github.com/FreedomIntelligence/HuatuoGPT-Vision.git /tmp/HuatuoGPT-Vision
    2. Download weights (auto via HuggingFace cache):
       python -c "from huggingface_hub import snapshot_download; snapshot_download('FreedomIntelligence/HuatuoGPT-Vision-7B')"

Ref:
    https://github.com/FreedomIntelligence/HuatuoGPT-Vision
    https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
from typing import Dict

from .base_model import BaseMedVQAModel, ModelOutput
from ..parse_answer import parse_answer

_DEFAULT_REPO = os.path.join(os.path.dirname(__file__), "..", "..", "repos", "HuatuoGPT-Vision")
HUATUOGPT_REPO_PATH = os.environ.get("HUATUOGPT_REPO_PATH", os.path.abspath(_DEFAULT_REPO))


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
    # Image preprocessing
    # ------------------------------------------------------------------
    def _process_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Pad to square and process with CLIP processor."""
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
    # Tokenize with image tokens
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
    # Build prompt
    # ------------------------------------------------------------------
    def _build_prompt(self, text: str) -> str:
        """HuatuoGPT conversation format with image placeholder."""
        return f"<image>\n<|user|>\n{text}\n<|assistant|>\n"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def inference(self, image: np.ndarray, prompt: str, max_new_tokens: int = 32) -> ModelOutput:
        pil_image = Image.fromarray(image)
        image_tensor = self._process_image(pil_image).unsqueeze(0)

        full_prompt = self._build_prompt(prompt)
        input_ids = self._tokenize_with_image(full_prompt).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        raw_text = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        parsed = parse_answer(raw_text)
        logits = self.get_choice_logits(image, prompt)

        return ModelOutput(
            raw_text=raw_text,
            parsed_answer=parsed if parsed else "PARSE_FAIL",
            logits=logits,
            parse_success=parsed is not None,
        )

    # ------------------------------------------------------------------
    # Logit extraction
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
