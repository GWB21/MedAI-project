"""
LLaVA-v1.5-7B wrapper.

Follows the standard LLaVA evaluation approach:
    - conv_templates['v1'] for prompt formatting
    - model.generate() with greedy decoding
    - Full output decode, strip input prompt
    - Answer parsing from generated text

Setup:
    pip install git+https://github.com/haotian-liu/LLaVA.git --no-deps
    Model: liuhaotian/llava-v1.5-7b (auto-downloaded via HuggingFace)

Ref:
    https://github.com/haotian-liu/LLaVA
    https://huggingface.co/liuhaotian/llava-v1.5-7b
"""

import re
import torch
import numpy as np
from PIL import Image
from typing import Dict, Optional

from .base_model import BaseMedVQAModel, ModelOutput


def _parse_mc_answer(text: str) -> Optional[str]:
    """Parse A/B/C/D from generated text."""
    raw = text.strip()
    if not raw:
        return None

    if raw.upper() in ("A", "B", "C", "D"):
        return raw.upper()

    match = re.search(r"(?:answer|option)\s*(?:is|:)\s*([A-Da-d])", raw, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-Da-d])\s+is\s+(?:correct|right|the answer)", raw, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.match(r"^([A-Da-d])[.)\s]", raw)
    if match:
        return match.group(1).upper()

    if raw[0].upper() in ("A", "B", "C", "D"):
        return raw[0].upper()

    for char in raw.upper():
        if char in ("A", "B", "C", "D"):
            return char

    return None


class LLaVAv15Model(BaseMedVQAModel):

    MODEL_ID = "liuhaotian/llava-v1.5-7b"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.device = None
        self._precision = "fp16"

    def load(self, device: str = "cuda") -> None:
        from llava.model.builder import load_pretrained_model

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=self.MODEL_ID,
                model_base=None,
                model_name="llava-v1.5-7b",
                device_map="auto",
            )
        )
        self.device = self.model.device
        self.model.eval()

    def _build_prompt(self, text: str) -> str:
        from llava.conversation import conv_templates
        from llava.constants import DEFAULT_IMAGE_TOKEN

        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _prepare_inputs(self, image: np.ndarray, full_prompt: str):
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX

        pil_image = Image.fromarray(image)
        image_tensor = process_images(
            [pil_image], self.image_processor, self.model.config
        )
        image_tensor = image_tensor.to(self.device, dtype=torch.float16)

        input_ids = tokenizer_image_token(
            full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).to(self.device)

        return input_ids, image_tensor

    def inference(self, image: np.ndarray, prompt: str, max_new_tokens: int = 256) -> ModelOutput:
        full_prompt = self._build_prompt(prompt)
        input_ids, image_tensor = self._prepare_inputs(image, full_prompt)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,  # positional (keyword causes NoneType in LLaVA)
                images=image_tensor,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )

        # Decode: full output then strip prompt
        full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        from llava.constants import DEFAULT_IMAGE_TOKEN
        input_text = full_prompt.replace(DEFAULT_IMAGE_TOKEN, "").strip()
        if input_text in full_output:
            generated_text = full_output[full_output.index(input_text) + len(input_text):].strip()
        else:
            generated_text = full_output.strip()

        parsed = _parse_mc_answer(generated_text)
        parse_success = parsed is not None
        if not parse_success:
            parsed = "A"

        logits = {"A": float("nan"), "B": float("nan"), "C": float("nan"), "D": float("nan")}

        return ModelOutput(
            raw_text=generated_text,
            parsed_answer=parsed,
            logits=logits,
            parse_success=parse_success,
        )

    def get_choice_logits(self, image: np.ndarray, prompt: str) -> Dict[str, float]:
        return {"A": float("nan"), "B": float("nan"), "C": float("nan"), "D": float("nan")}

    @property
    def name(self) -> str:
        return "llava_v15"

    @property
    def precision(self) -> str:
        return self._precision
