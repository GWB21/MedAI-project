"""
Abstract base class for all Medical VQA model wrappers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict
import numpy as np


@dataclass
class ModelOutput:
    raw_text: str                                    # Raw model output
    parsed_answer: Optional[str]                     # Parsed A/B/C/D or None
    logits: Dict[str, float] = field(default_factory=dict)  # {"A": 1.23, "B": -0.45, ...}
    parse_success: bool = True


class BaseMedVQAModel(ABC):
    @abstractmethod
    def load(self, device: str = "cuda") -> None:
        """Load model and processor onto the specified device."""
        pass

    @abstractmethod
    def inference(self, image: np.ndarray, prompt: str, max_new_tokens: int = 32) -> ModelOutput:
        """
        Run inference on a single image + prompt.

        Args:
            image: numpy array (H, W, 3), uint8, RGB
            prompt: Full prompt string with question and choices
            max_new_tokens: Maximum tokens to generate (32 is enough for MCQ)

        Returns:
            ModelOutput with raw_text, parsed_answer, logits
        """
        pass

    @abstractmethod
    def get_choice_logits(self, image: np.ndarray, prompt: str) -> Dict[str, float]:
        """
        Return logit values for each choice token (A/B/C/D).
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def precision(self) -> str:
        pass
