# inference/utils.py
import os
import csv
import cv2
import torch
from recognition.trocr_model import TrOCRRecognizer

def load_detector_model(checkpoint_path: str, device="cpu"):
    """
    Load your Faster-RCNN text detector from a .pth checkpoint.
    """
    from detection.model import get_text_detector
    model = get_text_detector(num_classes=2)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

def load_recognizer_model(model_dir: str, device="cpu"):
    """
    Load the fine-tuned TrOCR model from a folder (containing config.json etc).
    """
    return TrOCRRecognizer(model_name_or_path=model_dir, device=device)

class CSVLogger:
    """
    Simple CSV logger: write one dict per row.
    """
    def __init__(self, path, fieldnames):
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, row: dict):
        self.writer.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()

def visualize_detections(image, boxes, scores=None, color=(0,255,0), thickness=2):
    """
    Draw bounding boxes (and optional scores) on a BGR image.
    """
    out = image.copy()
    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(out, (x1,y1), (x2,y2), color, thickness)
        if scores is not None:
            cv2.putText(out,
                        f"{scores[i]:.2f}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color, 1)
    return out
