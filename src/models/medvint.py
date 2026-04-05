"""
MedVInT-TD wrapper.

The original PMC-VQA author model. Uses PMC-CLIP vision encoder + LLaMA with LoRA.

Setup:
    1. Clone the PMC-VQA repo for model code:
       git clone https://github.com/xiaoman-zhang/PMC-VQA.git /tmp/PMC-VQA
    2. Download checkpoint from HuggingFace:
       huggingface-cli download xmcmic/MedVInT-TD --local-dir ./checkpoints/MedVInT-TD

Ref:
    https://github.com/xiaoman-zhang/PMC-VQA
    https://huggingface.co/xmcmic/MedVInT-TD
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
from typing import Dict
from pathlib import Path

from .base_model import BaseMedVQAModel, ModelOutput
from ..parse_answer import parse_answer


class MedVInTModel(BaseMedVQAModel):

    HF_REPO = "xmcmic/MedVInT-TD"
    PMC_VQA_REPO = "https://github.com/xiaoman-zhang/PMC-VQA.git"

    def __init__(
        self,
        pmc_vqa_repo_path: str = "/tmp/PMC-VQA",
        checkpoint_dir: str = "./checkpoints/MedVInT-TD",
    ):
        self.pmc_vqa_repo_path = pmc_vqa_repo_path
        self.checkpoint_dir = checkpoint_dir
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.device = None
        self._precision = "fp16"

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(self, device: str = "cuda") -> None:
        # Ensure PMC-VQA repo is available for model code
        if not os.path.exists(self.pmc_vqa_repo_path):
            raise FileNotFoundError(
                f"PMC-VQA repo not found at {self.pmc_vqa_repo_path}. "
                f"Clone it: git clone {self.PMC_VQA_REPO} {self.pmc_vqa_repo_path}"
            )

        # Add PMC-VQA repo to path for model imports
        if self.pmc_vqa_repo_path not in sys.path:
            sys.path.insert(0, self.pmc_vqa_repo_path)

        self.device = device
        self._load_model()

    def _load_model(self) -> None:
        """Load MedVInT-TD using the PMC-VQA codebase."""
        from transformers import LlamaTokenizer

        # Find checkpoint file
        ckpt_path = self._find_checkpoint()

        # Import model class from PMC-VQA repo
        try:
            from models.QA_model import QA_model
        except ImportError:
            raise ImportError(
                "Cannot import QA_model from PMC-VQA repo. "
                f"Make sure {self.pmc_vqa_repo_path} contains models/QA_model.py"
            )

        # Model configuration matching the original training setup
        class ModelArgs:
            def __init__(self, ckpt_dir, **kwargs):
                self.llama_model = kwargs.get("llama_model", "chaoyi-wu/PMC_LLAMA_7B")
                self.vision_model = kwargs.get("vision_model", "CLIP")
                self.lora_rank = kwargs.get("lora_rank", 8)
                self.num_img_tokens = kwargs.get("num_img_tokens", 32)
                self.max_seq_len = kwargs.get("max_seq_len", 512)
                self.max_batch_size = kwargs.get("max_batch_size", 1)

        args = ModelArgs(self.checkpoint_dir)
        self.model = QA_model(args)

        # Load checkpoint weights
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(ckpt, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained("chaoyi-wu/PMC_LLAMA_7B")

    def _find_checkpoint(self) -> str:
        """Locate the model checkpoint file."""
        ckpt_dir = Path(self.checkpoint_dir)
        candidates = list(ckpt_dir.glob("**/*.bin")) + list(ckpt_dir.glob("**/*.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint found in {self.checkpoint_dir}. "
                f"Download: huggingface-cli download {self.HF_REPO} --local-dir {self.checkpoint_dir}"
            )
        # Prefer pytorch_model.bin
        for c in candidates:
            if c.name == "pytorch_model.bin":
                return str(c)
        return str(candidates[0])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def inference(self, image: np.ndarray, prompt: str) -> ModelOutput:
        pil_image = Image.fromarray(image)

        # Prepare image tensor through the model's vision pipeline
        image_tensor = self._process_image(pil_image)

        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            if hasattr(self.model, "generate_long_sentence"):
                raw_text = self.model.generate_long_sentence(
                    input_ids=input_ids,
                    images=image_tensor,
                    max_new_tokens=64,
                )
            elif hasattr(self.model, "generate"):
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    images=image_tensor,
                    max_new_tokens=64,
                )
                raw_text = self.tokenizer.decode(
                    output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
                )
            else:
                raise RuntimeError("MedVInT model has no generate method")

        raw_text = raw_text.strip() if isinstance(raw_text, str) else str(raw_text).strip()
        parsed = parse_answer(raw_text)
        logits = self.get_choice_logits(image, prompt)

        return ModelOutput(
            raw_text=raw_text,
            parsed_answer=parsed if parsed else "PARSE_FAIL",
            logits=logits,
            parse_success=parsed is not None,
        )

    def _process_image(self, pil_image: Image.Image) -> torch.Tensor:
        """Process image through the model's vision pipeline."""
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        image_tensor = transform(pil_image.convert("RGB"))
        return image_tensor.unsqueeze(0).to(self.device).half()

    # ------------------------------------------------------------------
    # Logit extraction
    # ------------------------------------------------------------------
    def get_choice_logits(self, image: np.ndarray, prompt: str) -> Dict[str, float]:
        pil_image = Image.fromarray(image)
        image_tensor = self._process_image(pil_image)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    images=image_tensor,
                    return_dict=True,
                )
                if hasattr(outputs, "logits"):
                    next_logits = outputs.logits[:, -1, :]
                else:
                    return {"A": float("nan"), "B": float("nan"), "C": float("nan"), "D": float("nan")}

            choice_logits = {}
            for choice in ("A", "B", "C", "D"):
                token_id = self.tokenizer.encode(choice, add_special_tokens=False)
                # LLaMA tokenizer may produce different token IDs
                tid = token_id[-1] if token_id else token_id[0]
                choice_logits[choice] = next_logits[0, tid].item()
            return choice_logits
        except Exception:
            return {"A": float("nan"), "B": float("nan"), "C": float("nan"), "D": float("nan")}

    @property
    def name(self) -> str:
        return "medvint"

    @property
    def precision(self) -> str:
        return self._precision
