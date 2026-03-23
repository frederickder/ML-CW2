"""
Supervised classifier training for active learning evaluation.

Follows the paper's "fully supervised" framework (Section 4.2.1):
  - Train ResNet-18 from scratch on ONLY the labeled set
  - SGD with 0.9 momentum + Nesterov, cosine LR schedule
  - Initial LR 0.025
  - Augmentations: random crops + horizontal flips
  - Re-initialise weights between AL iterations (paper Section F.2.1)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .resnet import cifar_resnet18
from .utils import accuracy


def train_classifier(
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int = 10,
    epochs: int = 100,
    lr: float = 0.025,
    weight_decay: float = 5e-4,
    device: torch.device = None,
    verbose: bool = True,
) -> tuple[float, list[float]]:
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
    
    Returns:
        (final_accuracy, epoch_accuracies): Final test accuracy and per-epoch history.
    """
    if device is None:
        device = torch.device("cpu")

    # Fresh model each time (re-initialise weights)
    model = cifar_resnet18(num_classes=num_classes).to(device)

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

        # Only evaluate at final epoch (massive speedup)
        if epoch == epochs - 1:
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
    return final_acc, epoch_accuracies


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
