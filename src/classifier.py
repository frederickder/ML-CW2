"""
Supervised classifier training for active learning evaluation.

Follows the paper's "fully supervised" framework (Section 4.2.1):
  - Train ResNet-18 from scratch on ONLY the labeled set
  - SGD with 0.9 momentum + Nesterov, cosine LR schedule
  - Initial LR 0.025
  - Augmentations: random crops + horizontal flips
  - Re-initialise weights between AL iterations (paper Section F.2.1)

Also includes:
  - Dropout variant for DBAL/BALD (MC Dropout)
  - Linear classifier for Framework 2 (self-supervised embeddings)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .resnet import cifar_resnet18


def train_classifier(
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int = 10,
    epochs: int = 100,
    lr: float = 0.025,
    weight_decay: float = 5e-4,
    device: torch.device = None,
    verbose: bool = True,
    use_dropout: bool = False,
    eval_every: int = 0,
) -> tuple[float, list[float], nn.Module]:
    """
    Train a ResNet-18 classifier from scratch on the provided labeled data.
    
    Args:
        train_loader: DataLoader for labeled training data (with augmentation).
        test_loader: DataLoader for test data.
        num_classes: Number of classes.
        epochs: Number of training epochs.
        lr: Initial learning rate.
        weight_decay: L2 regularisation.
        device: Compute device.
        verbose: Print training progress.
        use_dropout: Add dropout before FC for DBAL/BALD.
        eval_every: Evaluate every N epochs (0 = only final epoch, for speed on GPU pods).
    
    Returns:
        (final_accuracy, epoch_accuracies, model): Accuracy info and the trained model.
    """
    if device is None:
        device = torch.device("cpu")

    # Fresh model each time (re-initialise weights)
    model = cifar_resnet18(num_classes=num_classes).to(device)

    # Add dropout before FC if needed (for DBAL/BALD MC Dropout)
    if use_dropout:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes),
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    epoch_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Evaluate only at specified intervals or final epoch
        should_eval = (
            epoch == epochs - 1
            or (eval_every > 0 and (epoch + 1) % eval_every == 0)
        )
        if should_eval:
            test_acc = evaluate_classifier(model, test_loader, device)
            epoch_accuracies.append(test_acc)

            if verbose and (epoch + 1) % 20 == 0:
                avg_loss = train_loss / max(n_batches, 1)
                print(
                    f"  Epoch {epoch+1}/{epochs} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Test Acc: {test_acc:.2f}%"
                )

    final_acc = epoch_accuracies[-1] if epoch_accuracies else 0.0
    return final_acc, epoch_accuracies, model


# ──────────────────────────────────────────────
# Linear classifier (Framework 2)
# ──────────────────────────────────────────────


class LinearClassifier(nn.Module):
    """Simple linear classifier on frozen features."""
    def __init__(self, feature_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_linear_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    labeled_indices: list[int],
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int = 10,
    feature_dim: int = 512,
    epochs: int = 200,
    lr: float = 2.5,
    weight_decay: float = 5e-4,
    device: torch.device = None,
    verbose: bool = True,
) -> float:
    """
    Train a linear classifier on frozen SimCLR features (Framework 2).
    
    Paper (Appendix F.2.2): LR scaled up by 100x to 2.5, epochs doubled.
    """
    if device is None:
        device = torch.device("cpu")

    model = LinearClassifier(feature_dim, num_classes).to(device)

    # Get labeled subset
    train_feats = features[labeled_indices].to(device)
    train_labels = labels[labeled_indices].to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9,
        weight_decay=weight_decay, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        # With few examples, just do full-batch training
        outputs = model(train_feats)
        loss = criterion(outputs, train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_feats_dev = test_features.to(device)
        test_labels_dev = test_labels.to(device)
        preds = model(test_feats_dev).argmax(dim=1)
        acc = 100.0 * preds.eq(test_labels_dev).sum().item() / len(test_labels_dev)

    if verbose:
        print(f"  Linear classifier: {acc:.2f}%")

    return acc


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate classifier on test set. Returns accuracy as percentage."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total