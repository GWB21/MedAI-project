"""
PMC-VQA dataset loader.

Expected CSV columns (test_clean.csv):
    Figure_path, Question, Answer, Choice A, Choice B, Choice C, Choice D, Answer_label
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image


PROMPT_TEMPLATE = """{question}
A. {choice_A}
B. {choice_B}
C. {choice_C}
D. {choice_D}
Answer with the option's letter from the given choices directly."""


class PMCVQADataset:
    """PMC-VQA test_clean dataset with on-the-fly perturbation support."""

    def __init__(self, data_dir: str, csv_name: str = "test_clean.csv"):
        self.data_dir = Path(data_dir)
        self.csv_path = self.data_dir / csv_name

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        self._validate()

    def _validate(self) -> None:
        required = {"Figure_path", "Question", "Choice A", "Choice B", "Choice C", "Choice D", "Answer_label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}. Found: {list(self.df.columns)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image_path = self._resolve_image_path(row["Figure_path"])

        return {
            "image_id": Path(row["Figure_path"]).stem,
            "image_path": str(image_path),
            "question": row["Question"],
            "choice_A": row["Choice A"],
            "choice_B": row["Choice B"],
            "choice_C": row["Choice C"],
            "choice_D": row["Choice D"],
            "gt_answer": row["Answer_label"].strip().upper(),
        }

    def _resolve_image_path(self, fig_path: str) -> Path:
        """Try multiple locations for the image file."""
        candidates = [
            self.data_dir / fig_path,
            self.data_dir / "images" / fig_path,
            self.data_dir / "images" / Path(fig_path).name,
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Image not found: {fig_path}. Searched: {[str(c) for c in candidates]}"
        )

    def get_prompt(self, idx: int) -> str:
        item = self[idx]
        return PROMPT_TEMPLATE.format(
            question=item["question"],
            choice_A=item["choice_A"],
            choice_B=item["choice_B"],
            choice_C=item["choice_C"],
            choice_D=item["choice_D"],
        )

    def load_image(self, idx: int) -> np.ndarray:
        """Load image as RGB numpy array (H, W, 3) uint8."""
        item = self[idx]
        img = Image.open(item["image_path"]).convert("RGB")
        return np.array(img)

    def get_all_items(self) -> List[Dict]:
        return [self[i] for i in range(len(self))]

    def check_images(self) -> Dict[str, int]:
        """Check how many images exist and are loadable."""
        found, missing, corrupt = 0, 0, 0
        for i in range(len(self)):
            try:
                img = self.load_image(i)
                if img is not None and img.size > 0:
                    found += 1
                else:
                    corrupt += 1
            except FileNotFoundError:
                missing += 1
            except Exception:
                corrupt += 1
        return {"found": found, "missing": missing, "corrupt": corrupt, "total": len(self)}
