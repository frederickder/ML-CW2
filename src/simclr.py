"""
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.
(Chen et al., 2020)

Components:
  1. Encoder (ResNet-18 backbone) → 512-d features
  2. Projection head (MLP) → 128-d projection for contrastive loss
  3. NT-Xent (Normalised Temperature-scaled Cross Entropy) loss

After training, we discard the projection head and use the 512-d penultimate
layer as the feature representation for TypiClust.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .resnet import cifar_resnet18_backbone


# SimCLR Model

class SimCLR(nn.Module):
    """SimCLR model: encoder + projection head."""

    def __init__(self, feature_dim: int = 512, projection_dim: int = 128):
        super().__init__()
        self.encoder, enc_dim = cifar_resnet18_backbone()
        assert enc_dim == feature_dim

        # MLP projection head (as in Chen et al., 2020)
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input images [B, 3, H, W].
        Returns:
            features: 512-d encoder output (used for downstream tasks).
            projections: 128-d projected output (used for contrastive loss).
        """
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalised 512-d features (for TypiClust)."""
        features = self.encoder(x)
        return F.normalize(features, dim=1)


# NT-Xent Loss

class NTXentLoss(nn.Module):
    """Normalised Temperature-scaled Cross Entropy Loss (NT-Xent).
    
    For a batch of N images producing 2N augmented views, each positive pair
    is the two views of the same image, and the 2(N-1) other views are negatives.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: Projections from view 1 [B, D].
            z_j: Projections from view 2 [B, D].
        Returns:
            Scalar loss.
        """
        batch_size = z_i.size(0)

        # L2 normalise
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate: [z_i; z_j] → [2B, D]
        z = torch.cat([z_i, z_j], dim=0)

        # Similarity matrix [2B, 2B]
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        # Mask out self-similarity (diagonal)
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # Loss = -log(exp(pos) / sum(exp(all_negatives)))
        # Using log-sum-exp trick via cross entropy
        # For each row i, the "label" is the index of its positive pair
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ])

        loss = F.cross_entropy(sim_matrix, labels)
        return loss


# Training

def train_simclr(
    model: SimCLR,
    dataloader: DataLoader,
    epochs: int = 200,
    lr: float = 0.4,
    weight_decay: float = 1e-4,
    temperature: float = 0.5,
    device: torch.device = None,
    save_path: str = None,
) -> list[float]:
    """
    Train SimCLR model.
    
    Follows the paper's setup:
      - SGD with 0.9 momentum, cosine LR schedule
      - Initial LR 0.4 (scaled for batch size 512)
      - Weight decay 1e-4
    
    Args:
        model: SimCLR model.
        dataloader: DataLoader yielding (view1, view2) pairs.
        epochs: Number of training epochs.
        lr: Initial learning rate.
        weight_decay: L2 regularisation.
        temperature: NT-Xent temperature.
        device: Compute device.
        save_path: Path to save the trained encoder weights.
    
    Returns:
        List of per-epoch losses.
    """
    if device is None:
        device = next(model.parameters()).device

    model.to(device)
    model.train()

    criterion = NTXentLoss(temperature=temperature)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for (view1, view2), _ in pbar:
            view1, view2 = view1.to(device), view2.to(device)

            _, proj1 = model(view1)
            _, proj2 = model(view2)

            loss = criterion(proj1, proj2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save encoder weights
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"SimCLR model saved to {save_path}")

    return losses


# Feature extraction

@torch.no_grad()
def extract_features(
    model: SimCLR,
    dataloader: DataLoader,
    device: torch.device = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract L2-normalised features from trained SimCLR encoder.
    
    Args:
        model: Trained SimCLR model.
        dataloader: DataLoader (no augmentation — use embedding transform).
        device: Compute device.
    
    Returns:
        (features, labels): features [N, 512], labels [N].
    """
    if device is None:
        device = next(model.parameters()).device

    model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        features = model.encode(images)
        all_features.append(features.cpu())
        all_labels.append(labels)

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
