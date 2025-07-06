# recognition/trocr_model.py

import os
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class TrOCRRecognizer:
    def __init__(self, model_name_or_path: str, device="cpu"):
        # resolve to an absolute path
        path = model_name_or_path
        if not os.path.isabs(path):
            path = os.path.join(os.getcwd(), path)
        if not os.path.isdir(path):
            raise ValueError(f"Cannot find local model folder at '{path}'")

        # load processor & model from that folder ONLY
        self.processor = TrOCRProcessor.from_pretrained(
            path, local_files_only=True
        )
        self.model = VisionEncoderDecoderModel.from_pretrained(
            path, local_files_only=True
        )

        self.device = torch.device(device)
        self.model.to(self.device).eval()

    def predict(self, image):
        """
        image: H×W×3 BGR numpy array
        returns: decoded string
        """
        # 1) run processor (RGB conversion + tensor)
        pixel_values = self.processor(
            images=image[..., ::-1],
            return_tensors="pt"
        ).pixel_values.to(self.device)

        # 2) generate tokens
        generated_ids = self.model.generate(pixel_values)

        # 3) decode to text
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        return text
