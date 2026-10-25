"""
Lightweight convolutional autoencoder baseline for MVTec AD.
Train on normal images only; score test images by reconstruction error.

Folder layout (after downloading MVTec AD):
  mvtec_ad/<category>/train/good/*.png
  mvtec_ad/<category>/test/good/*.png
  mvtec_ad/<category>/test/<defect_type>/*.png

Install:  pip install torch torchvision scikit-learn pillow
Run:      python mvtec_autoencoder.py --root mvtec_ad --category bottle
"""
import argparse
from pathlib import Path
from pyexpat import model

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MVTecDataset(Dataset):
    def __init__(self, root, category, split, size=128):
        self.tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
        base = Path(root) / category / split
        self.items = []  # (path, label) label 0 = normal, 1 = defect
        for folder in sorted(base.iterdir()):
            label = 0 if folder.name == "good" else 1
            for p in sorted(folder.glob("*.png")):
                self.items.append((p, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label


class LightAE(nn.Module):
    """Small conv autoencoder (~100k parameters)."""

    def __init__(self):
        super().__init__()
        def down(i, o): return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        def up(i, o): return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.enc = nn.Sequential(down(3, 16), down(16, 32), down(32, 64), down(64, 64))
        self.dec = nn.Sequential(up(64, 64), up(64, 32), up(32, 16), nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="mvtec_ad")
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl = DataLoader(MVTecDataset(args.root, args.category, "train"), args.bs, shuffle=True)
    test_dl = DataLoader(MVTecDataset(args.root, args.category, "test"), 1)

    model = LightAE().to(dev)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for ep in range(args.epochs):
        model.train()
        total = 0
        for x, _ in train_dl:
            x = x.to(dev)
            loss = loss_fn(model(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep + 1}/{args.epochs}  loss {total / len(train_dl.dataset):.5f}")
            
    # Save the trained model weights
    model_path = f"{args.category}_autoencoder.pth"
    torch.save(model.state_dict(), model_path)

    print(f"Model saved successfully: {model_path}")

    # Evaluate: image-level AUROC using max pixel reconstruction error
    
    model.eval()
    scores, labels = [], []
    shown = False

    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(dev)

            # Reconstruct the input image
            recon = model(x)

            # Calculate pixel-wise reconstruction error
            err = ((recon - x) ** 2).mean(dim=1)

            # Keep the existing anomaly scores and labels
            scores.append(err.max().item())
            labels.append(int(y))

            # Visualize the first test image only
            if not shown:
                original = x[0].cpu().permute(1, 2, 0).numpy()
                reconstructed = recon[0].cpu().permute(1, 2, 0).numpy()
                error_map = err[0].cpu().numpy()

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                axes[0].imshow(original)
                axes[0].set_title("Original Image")
                axes[0].axis("off")

                axes[1].imshow(reconstructed.clip(0, 1))
                axes[1].set_title("Reconstructed Image")
                axes[1].axis("off")

                axes[2].imshow(error_map, cmap="hot")
                axes[2].set_title("Reconstruction Error Map")
                axes[2].axis("off")

                plt.tight_layout()
                plt.show()

                shown = True

    print(f"Image-level AUROC ({args.category}): {roc_auc_score(labels, scores):.4f}")


if __name__ == "__main__":
    main()
