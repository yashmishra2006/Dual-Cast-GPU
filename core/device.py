# core/device.py
import torch
import logging

class DeviceManager:
    """Handles CUDA/CPU selection with fallback."""

    def __init__(self, preferred="cuda"):
        self.logger = logging.getLogger(__name__)
        self.device = self._select_device(preferred)

    def _select_device(self, preferred):
        if preferred == "cuda" and torch.cuda.is_available():
            self.logger.info("CUDA available. Using GPU.")
            torch.backends.cudnn.benchmark = True
            return torch.device("cuda")
        else:
            self.logger.warning("Falling back to CPU.")
            return torch.device("cpu")

    def get(self):
        return self.device


# core/model_loader.py
import torch
import torchvision.models as models
from pathlib import Path
import logging


def load_resnet(model_path: str, num_classes: int, device: torch.device, half: bool = False):
    logger = logging.getLogger(__name__)
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    if not Path(model_path).exists():
        logger.error(f"ResNet model file not found: {model_path}")
        return None

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        if device.type == "cuda" and half:
            model = model.half()
            logger.info("Using half precision on CUDA.")

        logger.info(f"ResNet model loaded from {model_path}")
        return model

    except Exception as e:
        logger.error(f"Failed to load ResNet: {e}")
        return None