"""
HuatuoGPT-Vision-7B wrapper.

Setup:
    pip install transformers>=4.37.0
    # Model uses trust_remote_code=True to load custom architecture (Qwen2-VL based)

Ref:
    https://github.com/FreedomIntelligence/HuatuoGPT-Vision
    https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict

from .base_model import BaseMedVQAModel, ModelOutput
from ..parse_answer import parse_answer


class HuatuoGPTVisionModel(BaseMedVQAModel):

    MODEL_ID = "FreedomIntelligence/HuatuoGPT-Vision-7B"

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self._precision = "bf16"

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(self, device: str = "cuda") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.device = device
        self.model.eval()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def inference(self, image: np.ndarray, prompt: str, max_new_tokens: int = 32) -> ModelOutput:
        pil_image = Image.fromarray(image)

        # HuatuoGPT-Vision provides a chat() method via trust_remote_code
        if hasattr(self.model, "chat"):
            raw_text = self.model.chat(
                tokenizer=self.tokenizer,
                image=pil_image,
                msgs=[{"role": "user", "content": prompt}],
                temperature=0,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        else:
            raw_text = self._manual_inference(pil_image, prompt, max_new_tokens)

        raw_text = raw_text.strip() if isinstance(raw_text, str) else str(raw_text).strip()
        parsed = parse_answer(raw_text)
        logits = self.get_choice_logits(image, prompt)

        return ModelOutput(
            raw_text=raw_text,
            parsed_answer=parsed if parsed else "PARSE_FAIL",
            logits=logits,
            parse_success=parsed is not None,
        )

    def _manual_inference(self, pil_image: Image.Image, prompt: str, max_new_tokens: int = 32) -> str:
        """Fallback if model doesn't expose a chat() method."""
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=pil_image, text=text, return_tensors="pt").to(
            self.model.device
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
                num_beams=1,
            )
        return processor.decode(
            output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    # ------------------------------------------------------------------
    # Logit extraction
    # ------------------------------------------------------------------
    def get_choice_logits(self, image: np.ndarray, prompt: str) -> Dict[str, float]:
        """
        Extract first-token logits for A/B/C/D.
        Uses the model's forward pass to get next-token prediction logits.
        """
        pil_image = Image.fromarray(image)

        try:
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                self.MODEL_ID, trust_remote_code=True
            )
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images=pil_image, text=text, return_tensors="pt").to(
                self.model.device
            )

            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)
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
