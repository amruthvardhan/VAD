# detection/utils.py

import torch
from detection.model import get_text_detector

def load_detector_model(checkpoint_path: str, device="cpu"):
    """
    Load your Faster-R-CNN text detector from a .pth checkpoint.
    """
    # 1) Build the model architecture
    model = get_text_detector(num_classes=2)

    # 2) Load the saved state dict
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)

    # 3) Move to device & set eval() mode
    model.to(device).eval()
    return model
