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

        # PyTorch 2.6+: weights_only=True 기본값 → PMC-CLIP numpy 호환 위해 전역 패치
        import functools
        _orig_torch_load = torch.load
        if not hasattr(_orig_torch_load, '_patched'):
            torch.load = functools.partial(_orig_torch_load, weights_only=False)
            torch.load._patched = True

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
    # Image preprocessing (원본 PMC_QA_Dataset Test 모드와 동일)
    # ------------------------------------------------------------------
    def _process_image(self, pil_image: Image.Image) -> torch.Tensor:
        from torchvision import transforms
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
        return transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Build prompt (원본 PMC_QA_Dataset choice 형식)
    # ------------------------------------------------------------------
    def build_prompt(self, question: str, choice_A: str, choice_B: str, choice_C: str, choice_D: str) -> str:
        """원본 형식: 'Question: {Q}Choices: {A}{B}{C}{D}The Answer is:'"""
        combined = choice_A + choice_B + choice_C + choice_D
        return f"Question: {question}Choices:{combined}The Answer is:"

    # ------------------------------------------------------------------
    # Inference (원본 test.py 방식: generate → argmax → 마지막 글자 → 선택지 매칭)
    # ------------------------------------------------------------------
    def inference(self, image: np.ndarray, prompt: str, max_new_tokens: int = 32) -> ModelOutput:
        import difflib

        pil_image = Image.fromarray(image)
        image_tensor = self._process_image(pil_image)
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)

        with torch.no_grad():
            # model.generate() returns logits (single forward, NOT autoregressive)
            generation_logits = self.model.generate(input_ids, image_tensor)
            pred_ids = generation_logits.argmax(-1)
            raw_text = self.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()

        # 원본 방식: 마지막 글자로 답변 결정
        pred_char = raw_text[-1] if raw_text else ""

        # 선택지 텍스트 추출 (프롬프트에서)
        choices = self._extract_choices(prompt)
        if choices:
            idx = self._find_most_similar_index(choices, pred_char)
            pred = ["A", "B", "C", "D"][idx] if idx is not None else "A"
        else:
            pred = pred_char.upper() if pred_char.upper() in ("A", "B", "C", "D") else "A"

        # logit도 추출 (분석용)
        logits = self._get_choice_logits_from_generation(generation_logits)

        return ModelOutput(
            raw_text=raw_text,
            parsed_answer=pred,
            logits=logits,
            parse_success=True,
        )

    def _extract_choices(self, prompt: str):
        """프롬프트에서 A~D 선택지 텍스트 추출"""
        import re
        choices = []
        for letter in ("A", "B", "C", "D"):
            match = re.search(rf'{letter}\.\s*(.+?)(?:\n|$)', prompt)
            if match:
                choices.append(match.group(1).strip())
        return choices if len(choices) == 4 else None

    def _find_most_similar_index(self, str_list, target_str):
        """원본 test.py의 find_most_similar_index 그대로"""
        import difflib
        best_idx, best_sim = 0, 0
        for i, s in enumerate(str_list):
            sim = difflib.SequenceMatcher(None, s, target_str).ratio()
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        return best_idx

    def _get_choice_logits_from_generation(self, generation_logits) -> Dict[str, float]:
        """generate() 결과에서 마지막 위치의 A/B/C/D logit 추출"""
        try:
            last_logits = generation_logits[:, -1, :]
            choice_logits = {}
            for choice in ("A", "B", "C", "D"):
                token_id = self.tokenizer.encode(choice, add_special_tokens=False)[-1]
                choice_logits[choice] = last_logits[0, token_id].item()
            return choice_logits
        except Exception:
            return {"A": float("nan"), "B": float("nan"), "C": float("nan"), "D": float("nan")}

    # ------------------------------------------------------------------
    # Logit extraction (standalone, for compatibility)
    # ------------------------------------------------------------------
    def get_choice_logits(self, image: np.ndarray, prompt: str) -> Dict[str, float]:
        pil_image = Image.fromarray(image)
        image_tensor = self._process_image(pil_image)
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        with torch.no_grad():
            generation_logits = self.model.generate(input_ids, image_tensor)
        return self._get_choice_logits_from_generation(generation_logits)

    @property
    def name(self) -> str:
        return "medvint"

    @property
    def precision(self) -> str:
        return self._precision
