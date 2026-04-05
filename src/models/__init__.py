from .base_model import BaseMedVQAModel, ModelOutput


def get_model(model_name: str) -> BaseMedVQAModel:
    """Lazy-load model class to avoid importing torch at module level."""
    if model_name == "llava_med":
        from .llava_med import LLavaMedModel
        return LLavaMedModel()
    elif model_name == "huatuogpt":
        from .huatuogpt import HuatuoGPTVisionModel
        return HuatuoGPTVisionModel()
    elif model_name == "medvint":
        from .medvint import MedVInTModel
        return MedVInTModel()
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: llava_med, huatuogpt, medvint")
