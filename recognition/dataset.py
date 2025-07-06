import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Character set for CRNN (blank index 0 for CTC)
CHARS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.-")

class OCRDataset(torch.utils.data.Dataset):
    """
    A Dataset of all cropped text‐regions from data/ann/*.txt over data/imgs/*.{png,jpg}.
    Returns:
      - tensor [1,H,W] grayscale image
      - label_tensor [L] (int indices into CHARS, +1)
      - length_tensor scalar
    """
    def __init__(self, data_root, img_height=32, transforms=None):
        self.img_dir    = os.path.join(data_root, "imgs")
        self.ann_dir    = os.path.join(data_root, "ann")
        self.img_height = img_height
        self.transforms = transforms

        exts = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]
        self.entries = []

        for ann in sorted(os.listdir(self.ann_dir)):
            if not ann.lower().endswith(".txt"):
                continue
            base = os.path.splitext(ann)[0]
            # find matching image file
            img_path = None
            for e in exts:
                cand = os.path.join(self.img_dir, base + e)
                if os.path.exists(cand):
                    img_path = cand
                    break
            if img_path is None:
                continue

            # read each line: x1 y1 x2 y2 TEXT
            with open(os.path.join(self.ann_dir, ann), "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=4)
                    if len(parts) < 5:
                        continue
                    x1, y1, x2, y2 = map(int, parts[:4])
                    raw_text = parts[4].strip()
                    # remove any spaces (annotations should be single tokens)
                    text = raw_text.replace(" ", "")
                    if not text:
                        continue
                    self.entries.append((img_path, x1, y1, x2, y2, text))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, x1, y1, x2, y2, text = self.entries[idx]

        # load & crop
        im   = Image.open(img_path).convert("L")
        crop = im.crop((x1, y1, x2, y2))

        # resize to fixed height
        w, h    = crop.size
        new_h   = self.img_height
        new_w   = max(1, int(w * new_h / h))
        crop    = crop.resize((new_w, new_h), Image.BILINEAR)

        # to tensor [1,H,W], with optional augmentations
        if self.transforms:
            img_t = self.transforms(crop)
        else:
            img_t = transforms.ToTensor()(crop)

        # encode text → [L] of ints (1..len(CHARS)), 0 is blank for CTC
        label_indices = []
        for c in text:
            if c in CHARS:
                label_indices.append(CHARS.index(c) + 1)
            # else skip any unexpected char
        label_t = torch.tensor(label_indices, dtype=torch.int64)

        # length tensor
        len_t = torch.tensor(len(label_indices), dtype=torch.int64)

        return img_t, label_t, len_t

def collate_fn_ocr(batch):
    """
    Pads every image in the batch to the same width, stacks them,
    pads labels to same length, and returns (imgs, labels, lengths).
    """
    images, labels, lengths = zip(*batch)

    # 1) pad images
    max_w = max(img.shape[2] for img in images)
    padded = []
    for img in images:
        pad_w = max_w - img.shape[2]
        img_p = F.pad(img, (0, pad_w, 0, 0), value=0.0)
        padded.append(img_p)
    imgs_tensor = torch.stack(padded, dim=0)  # [B,1,H,max_w]

    # 2) pad labels
    labels_tensor  = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    lengths_tensor = torch.stack(lengths)

    return imgs_tensor, labels_tensor, lengths_tensor

def get_ocr_transform(train: bool):
    """
    Returns a torchvision.transforms.Compose for OCR crops:
      - If train=True: random affine, random invert/blur, then ToTensor()
      - If train=False: only ToTensor()
    """
    transform_list = []
    if train:
        transform_list.append(
            transforms.RandomAffine(
                degrees=5,
                translate=(0.02, 0.02),
                scale=(0.9, 1.1),
                shear=(-3, 3),
                interpolation=InterpolationMode.BILINEAR,
                fill=0
            )
        )
        transform_list.append(transforms.RandomInvert(p=0.5))
        transform_list.append(transforms.RandomApply(
            [transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3
        ))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)
