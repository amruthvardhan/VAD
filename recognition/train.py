#!/usr/bin/env python3
import os
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from recognition.dataset import OCRDataset, collate_fn_ocr
from recognition.utils   import label_to_string

def train(args):
    # 1) Load & split dataset
    ds = OCRDataset(data_root=args.data_root)
    total = len(ds)
    train_n = int(0.8 * total)
    val_n   = total - train_n
    torch.manual_seed(42)
    train_ds, val_ds = random_split(ds, [train_n, val_n])

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_ocr
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_ocr
    )

    # 2) Load TrOCR & processor
    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model     = VisionEncoderDecoderModel.from_pretrained(args.model_name)
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3) Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_val = float("inf")
    no_improve = 0

    # for converting tensors → PIL
    to_pil = transforms.ToPILImage()

    # 4) Training loop
    for epoch in range(1, args.epochs + 1):
        # ——— Train ———
        model.train()
        train_loss = 0.0
        for imgs, labels, lengths in train_loader:
            # convert to RGB PIL images
            pil_imgs = []
            for img in imgs:  # img: [1,H,W]
                pil = to_pil(img)
                if pil.mode != "RGB":
                    pil = pil.convert("RGB")
                pil_imgs.append(pil)

            # ground‐truth strings
            texts = [label_to_string(lbl[:l]) for lbl, l in zip(labels, lengths)]

            # a) encode images
            img_enc = processor(
                images=pil_imgs,
                return_tensors="pt",
                padding=True
            )
            pixel_values = img_enc.pixel_values.to(device)

            # b) tokenize text separately
            txt_enc = processor.tokenizer(
                text=texts,
                padding=True,
                return_tensors="pt"
            )
            input_ids = txt_enc.input_ids.to(device)
            input_ids[input_ids == processor.tokenizer.pad_token_id] = -100

            # c) forward pass
            outputs = model(pixel_values=pixel_values, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # ——— Validate ———
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels, lengths in val_loader:
                # same PIL conversion
                pil_imgs = []
                for img in imgs:
                    pil = to_pil(img)
                    if pil.mode != "RGB":
                        pil = pil.convert("RGB")
                    pil_imgs.append(pil)

                texts = [label_to_string(lbl[:l]) for lbl, l in zip(labels, lengths)]

                img_enc = processor(
                    images=pil_imgs,
                    return_tensors="pt",
                    padding=True
                )
                pixel_values = img_enc.pixel_values.to(device)

                txt_enc = processor.tokenizer(
                    text=texts,
                    padding=True,
                    return_tensors="pt"
                )
                input_ids = txt_enc.input_ids.to(device)
                input_ids[input_ids == processor.tokenizer.pad_token_id] = -100

                outputs = model(pixel_values=pixel_values, labels=input_ids)
                val_loss += outputs.loss.item()

        avg_val = val_loss / len(val_loader)

        print(f"[Epoch {epoch}/{args.epochs}] "
              f"Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}")

        # 5) Schedule & early‐stop
        scheduler.step(avg_val)
        if avg_val < best_val:
            best_val = avg_val
            no_improve = 0
            os.makedirs(args.output_dir, exist_ok=True)
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            print(f"→ Saved best model (val_loss={best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= 5:
                print("Early stopping: no improvement for 5 epochs.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune TrOCR on your UI crops"
    )
    parser.add_argument("--data_root",  type=str, default="data",
                        help="root folder with imgs/ and ann/")
    parser.add_argument("--model_name", type=str,
                        default="microsoft/trocr-base-printed")
    parser.add_argument("--output_dir", type=str, default="trocr-finetuned")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--lr",         type=float, default=2e-5)
    args = parser.parse_args()
    train(args)
