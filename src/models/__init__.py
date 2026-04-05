from .base_model import BaseMedVQAModel, ModelOutput
from .llava_med import LLavaMedModel
from .huatuogpt import HuatuoGPTVisionModel
from .medvint import MedVInTModel

MODEL_REGISTRY = {
    "llava_med": LLavaMedModel,
    "huatuogpt": HuatuoGPTVisionModel,
    "medvint": MedVInTModel,
}

def get_model(model_name: str) -> BaseMedVQAModel:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name]()
