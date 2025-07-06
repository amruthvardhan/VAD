import os
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

class LVADTextDetDataset(torch.utils.data.Dataset):
    """
    Dataset for LVAD UI text‐region detection.
    Expects:
      data/imgs/  ← screenshots (PNG/JPG)
      data/ann/   ← .txt annotations (xmin ymin xmax ymax TEXT)
    Returns:
      img_tensor, target_dict with keys:
        - boxes: Tensor[N,4] (xmin,ymin,xmax,ymax)
        - labels: Tensor[N] (all ones, since every region is “text”)
    """
    def __init__(self, data_root, transforms=None):
        self.img_dir = os.path.join(data_root, "imgs")
        self.ann_dir = os.path.join(data_root, "ann")

        # Gather all image files
        all_images = sorted(
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        # Include any image that has a corresponding .txt in ann/
        self.images = []
        for fname in all_images:
            base = os.path.splitext(fname)[0]
            ann_path = os.path.join(self.ann_dir, base + ".txt")
            if os.path.isfile(ann_path):
                self.images.append(fname)
            else:
                print(f"Skipping {fname}: no annotation file found.")

        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1) Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        # 2) Load annotation lines
        base = os.path.splitext(img_name)[0]
        ann_path = os.path.join(self.ann_dir, base + ".txt")
        boxes = []
        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=4)
                if len(parts) < 5:
                    continue
                xmin, ymin, xmax, ymax, _ = parts
                boxes.append([float(xmin), float(ymin),
                              float(xmax), float(ymax)])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # 3) All regions are “text” => label=1
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        # 4) Apply transforms (if any), else convert to tensor
        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)

        return img, target

def get_transform(train: bool):
    """
    Returns a torchvision.transforms.Compose for LVAD detection:
      - train=True: small affine + ToTensor (for augmentation)
      - train=False: just ToTensor()
    """
    transform_list = []
    if train:
        transform_list.append(
            T.RandomAffine(
                degrees=5,
                translate=(0.02, 0.02),
                scale=(0.98, 1.02),
                shear=(-2, 2),
                interpolation=InterpolationMode.BILINEAR,
                fill=0
            )
        )
    transform_list.append(T.ToTensor())
    return T.Compose(transform_list)
