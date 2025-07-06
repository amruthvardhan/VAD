import os
import torch
from torch.utils.data import DataLoader, Subset
from detection.model import get_text_detector  # alias: get_model
from detection.dataset import LVADTextDetDataset, get_transform

def collate_fn(batch):
    # Collate function for detection DataLoader (returns tuples of lists)
    return tuple(zip(*batch))

def main(data_root, num_epochs, lr, batch_size, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1. Load dataset and split into train/val sets
    full_dataset = LVADTextDetDataset(data_root, transforms=get_transform(train=False))
    total = len(full_dataset)
    train_n = int(0.8 * total)
    val_n = total - train_n
    torch.manual_seed(42)
    indices = torch.randperm(total)
    train_idx = indices[:train_n]
    val_idx = indices[train_n:]
    train_dataset = LVADTextDetDataset(data_root, transforms=get_transform(train=True))
    val_dataset   = LVADTextDetDataset(data_root, transforms=get_transform(train=False))
    train_ds = Subset(train_dataset, train_idx)
    val_ds   = Subset(val_dataset, val_idx)
    print(f"Total images in '{data_root}': {total}  →  {len(train_idx)} train / {len(val_idx)} val")

    # 2. DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False,
                              num_workers=4, collate_fn=collate_fn)

    # 3. Build model and move to device
    model = get_text_detector(num_classes=2)
    model.to(device)

    # 4. Optimizer and LR scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = float("inf")
    for epoch in range(1, num_epochs + 1):
        # ---- Training ----
        model.train()
        total_train_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_train_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        lr_scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"[Epoch {epoch}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")

        # ---- Validation ----
        model.train()  # use train mode for val to compute loss
        total_val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                total_val_loss += sum(loss for loss in loss_dict.values()).item()
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"[Epoch {epoch}/{num_epochs}] Val Loss: {avg_val_loss:.4f}")

        # ---- Save best model checkpoint ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(output_dir, exist_ok=True)
            ckpt_path = os.path.join(output_dir, "checkpoint.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss
            }, ckpt_path)
            print(f"Saved best model to {ckpt_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train LVAD text detector")
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root folder containing imgs/ and ann/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    main(data_root=args.data_root, num_epochs=args.epochs,
         lr=args.lr, batch_size=args.batch_size, output_dir=args.output_dir)
