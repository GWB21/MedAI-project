"""
MedVInT-TD wrapper.

Uses PMC-CLIP vision encoder + PMC-LLaMA + LoRA (Q-Former decoder).

Setup (handled by scripts/setup.py):
    1. Clone PMC-VQA repo → repos/PMC-VQA
    2. Download MedVInT-TD checkpoint → checkpoints/MedVInT-TD
    3. Download PMC-LLaMA base model (HF cache)
    4. Download PMC-CLIP checkpoint

Ref:
    https://github.com/xiaoman-zhang/PMC-VQA
    https://huggingface.co/xmcmic/MedVInT-TD
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
from typing import Dict, Optional
from pathlib import Path

from .base_model import BaseMedVQAModel, ModelOutput
from ..parse_answer import parse_answer

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_PMC_VQA_REPO = os.path.join(_PROJECT_ROOT, "repos", "PMC-VQA")
_DEFAULT_CKPT_DIR = os.path.join(_PROJECT_ROOT, "checkpoints", "MedVInT-TD")


class MedVInTModel(BaseMedVQAModel):

    HF_REPO = "xmcmic/MedVInT-TD"

    def __init__(
        self,
        pmc_vqa_repo_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        llama_path: str = "chaoyi-wu/PMC_LLAMA_7B",
    ):
        self.pmc_vqa_repo_path = pmc_vqa_repo_path or _DEFAULT_PMC_VQA_REPO
        self.checkpoint_dir = checkpoint_dir or _DEFAULT_CKPT_DIR
        self.llama_path = llama_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self._precision = "fp16"

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    def load(self, device: str = "cuda") -> None:
        medvint_src = os.path.join(self.pmc_vqa_repo_path, "src", "MedVInT_TD")
        if not os.path.exists(medvint_src):
            raise FileNotFoundError(
                f"MedVInT-TD source not found at {medvint_src}. "
                f"Clone: git clone https://github.com/xiaoman-zhang/PMC-VQA.git {self.pmc_vqa_repo_path}"
            )

        # Add MedVInT-TD source to path
        if medvint_src not in sys.path:
            sys.path.insert(0, medvint_src)

        self.device = device

        # Resolve PMC-LLaMA path from HF cache
        from huggingface_hub import snapshot_download
        llama_local = snapshot_download(repo_id=self.llama_path)

        # Find checkpoint
        ckpt_path = self._find_checkpoint()
        print(f"MedVInT-TD checkpoint: {ckpt_path}")
        print(f"PMC-LLaMA: {llama_local}")

        # Build model args matching test.py defaults
        from dataclasses import dataclass, field
        from typing import Optional as Opt

        @dataclass
        class ModelArgs:
            model_path: str = llama_local
            ckp: str = ""
            checkpointing: bool = False
            N: int = 12
            H: int = 8
            img_token_num: int = 32
            voc_size: int = 32000
            hidden_dim: int = 4096
            Vision_module: str = "PMC-CLIP"
            visual_model_path: str = ""
            is_lora: bool = True
            peft_mode: str = "lora"
            lora_rank: int = 8

        args = ModelArgs()

        # Check for PMC-CLIP checkpoint
        pmc_clip_path = self._find_pmc_clip()
        if pmc_clip_path:
            args.Vision_module = "PMC-CLIP"
            args.visual_model_path = pmc_clip_path
        else:
            # Fallback to standard CLIP (works but may have slightly different performance)
            print("[WARN] PMC-CLIP checkpoint not found, using standard CLIP")
            args.Vision_module = "CLIP"
            args.visual_model_path = "openai/clip-vit-large-patch14"

        # Load model
        from models.QA_model import QA_model
        self.model = QA_model(args)

        # Load checkpoint weights (weights_only=False for PMC-CLIP numpy compat)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Handle peft version compatibility
        fixed_ckpt = {}
        for name, v in ckpt.items():
            if "self_attn.q_proj.weight" in name and "vision_model" not in name:
                name = name.replace("self_attn.q_proj.weight", "self_attn.q_proj.base_layer.weight")
            if "self_attn.v_proj.weight" in name and "vision_model" not in name:
                name = name.replace("self_attn.v_proj.weight", "self_attn.v_proj.base_layer.weight")
            if "lora_A" in name and "lora_A.default" not in name:
                name = name.replace("lora_A", "lora_A.default")
            if "lora_B" in name and "lora_B.default" not in name:
                name = name.replace("lora_B", "lora_B.default")
            fixed_ckpt[name] = v

        self.model.load_state_dict(fixed_ckpt, strict=False)
        self.model.to(device)
        self.model.eval()

        # Load tokenizer (PMC-LLaMA has broken empty special tokens in config)
        import json, shutil, tempfile
        from transformers import LlamaTokenizerFast
        tmp_tok = tempfile.mkdtemp()
        for fn in os.listdir(llama_local):
            if "tokenizer" in fn or "special" in fn:
                shutil.copy2(os.path.join(llama_local, fn), tmp_tok)
        cfg_p = os.path.join(tmp_tok, "tokenizer_config.json")
        with open(cfg_p) as f:
            tcfg = json.load(f)
        tcfg["bos_token"] = "<s>"
        tcfg["eos_token"] = "</s>"
        tcfg["unk_token"] = "<unk>"
        with open(cfg_p, "w") as f:
            json.dump(tcfg, f)
        self.tokenizer = LlamaTokenizerFast.from_pretrained(tmp_tok)
        self._precision = "fp32"  # MedVInT loads in fp32 by default

    def _find_checkpoint(self) -> str:
        # Standard path from test.py
        std_path = os.path.join(
            self.checkpoint_dir,
            "VQA_lora_PMC_LLaMA_PMCCLIP", "choice", "checkpoint-4000", "pytorch_model.bin"
        )
        if os.path.exists(std_path):
            return std_path
        # Search
        for root, dirs, files in os.walk(self.checkpoint_dir):
            for f in files:
                if f == "pytorch_model.bin" and "choice" in root:
                    return os.path.join(root, f)
        raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_dir}")

    def _find_pmc_clip(self) -> Optional[str]:
        """Search for PMC-CLIP checkpoint.pt"""
        candidates = [
            os.path.join(_PROJECT_ROOT, "checkpoints", "PMC-CLIP", "checkpoint.pt"),
            os.path.join(self.pmc_vqa_repo_path, "img_checkpoint", "PMC-CLIP", "checkpoint.pt"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------
    def _process_image(self, pil_image: Image.Image) -> torch.Tensor:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        return transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def inference(self, image: np.ndarray, prompt: str, max_new_tokens: int = 32) -> ModelOutput:
        logits = self.get_choice_logits(image, prompt)
        pred = max(logits, key=logits.get)

        return ModelOutput(
            raw_text="",
            parsed_answer=pred,
            logits=logits,
            parse_success=True,
        )

    # ------------------------------------------------------------------
    # Logit extraction
    # ------------------------------------------------------------------
    def get_choice_logits(self, image: np.ndarray, prompt: str) -> Dict[str, float]:
        pil_image = Image.fromarray(image)
        image_tensor = self._process_image(pil_image)
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)

        try:
            with torch.no_grad():
                # forward returns CausalLMOutputWithPast with .loss and .logits
                outputs = self.model(input_ids, image_tensor)
                next_logits = outputs.logits[:, -1, :]

            choice_logits = {}
            for choice in ("A", "B", "C", "D"):
                token_id = self.tokenizer.encode(choice, add_special_tokens=False)[-1]
                choice_logits[choice] = next_logits[0, token_id].item()
            return choice_logits
        except Exception:
            return {"A": float("nan"), "B": float("nan"), "C": float("nan"), "D": float("nan")}

    @property
    def name(self) -> str:
        return "medvint"

    @property
    def precision(self) -> str:
        return self._precision
