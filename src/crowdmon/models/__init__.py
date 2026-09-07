from .csrnet import CSRNet
from .dmcount import DMCount
from .weights import AVAILABLE_WEIGHTS, WEIGHTS_NOTES, ensure_weights, is_downloaded, weight_path

_ARCHITECTURES = {"DM-Count": DMCount, "CSRNet": CSRNet}


def build_model(model_name: str, model_weights: str, device, download: bool = True):
    """Instantiate a density-regression model and load pretrained weights onto `device`."""
    import torch

    if model_name not in _ARCHITECTURES:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(_ARCHITECTURES)}")

    if download:
        path = ensure_weights(model_name, model_weights)
    else:
        path = weight_path(model_name, model_weights)
        if not path.exists():
            raise FileNotFoundError(
                f"Pretrained weights not found at {path}. "
                f"Run scripts/download_weights.py or call with download=True."
            )

    model = _ARCHITECTURES[model_name]()
    state = torch.load(path, map_location="cpu")
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


__all__ = [
    "CSRNet",
    "DMCount",
    "AVAILABLE_WEIGHTS",
    "WEIGHTS_NOTES",
    "build_model",
    "ensure_weights",
    "is_downloaded",
    "weight_path",
]
