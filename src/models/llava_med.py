"""
LLaVA-Med-1.5 (Mistral-7B) wrapper.

Setup:
    pip install git+https://github.com/haotian-liu/LLaVA.git
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict

from .base_model import BaseMedVQAModel, ModelOutput
from ..parse_answer import parse_answer


class LLavaMedModel(BaseMedVQAModel):

    MODEL_ID = "microsoft/llava-med-v1.5-mistral-7b"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.device = None
        self._precision = "fp16"

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(self, device: str = "cuda") -> None:
        from llava.model.builder import load_pretrained_model

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                model_path=self.MODEL_ID,
                model_base=None,
                model_name="llava-med-v1.5-mistral-7b",
                device_map="auto",
            )
        )
        self.device = self.model.device
        self.model.eval()

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------
    def _build_prompt(self, text: str) -> str:
        from llava.conversation import conv_templates

        conv = conv_templates["mistral_instruct"].copy()
        from llava.constants import DEFAULT_IMAGE_TOKEN

        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    # ------------------------------------------------------------------
    # Prepare inputs
    # ------------------------------------------------------------------
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
        attention_mask = torch.ones_like(input_ids)

        return input_ids, image_tensor, attention_mask

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def inference(self, image: np.ndarray, prompt: str, max_new_tokens: int = 32) -> ModelOutput:
        full_prompt = self._build_prompt(prompt)
        input_ids, image_tensor, attention_mask = self._prepare_inputs(image, full_prompt)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=attention_mask,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                use_cache=True,
            )

        # Image token expands from 1 to ~576 embeddings internally,
        # so we decode the full output and strip the input prompt portion
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        from llava.constants import DEFAULT_IMAGE_TOKEN
        input_text = full_prompt.replace(DEFAULT_IMAGE_TOKEN, "").strip()
        if input_text in full_text:
            raw_text = full_text[full_text.index(input_text) + len(input_text) :].strip()
        else:
            raw_text = full_text.strip()
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
        full_prompt = self._build_prompt(prompt)
        input_ids, image_tensor, attention_mask = self._prepare_inputs(image, full_prompt)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                images=image_tensor,
                attention_mask=attention_mask,
                return_dict=True,
            )
            next_logits = outputs.logits[:, -1, :]

        choice_logits = {}
        for choice in ("A", "B", "C", "D"):
            token_id = self.tokenizer.encode(choice, add_special_tokens=False)[0]
            choice_logits[choice] = next_logits[0, token_id].item()
        return choice_logits

    @property
    def name(self) -> str:
        return "llava_med"

    @property
    def precision(self) -> str:
        return self._precision
