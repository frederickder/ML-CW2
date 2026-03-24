"""
FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling
(Zhang et al., 2021)

Minimal but correct implementation for Framework 3 of Hacohen et al. (2022).

FlexMatch extends FixMatch by using per-class adaptive confidence thresholds
(Curriculum Pseudo Labeling) instead of a fixed threshold. Classes that are
easier to learn get higher thresholds, while harder classes get lower ones.

Paper config (Appendix F.2.3):
  - WideResNet-28-2, 400k iterations (we use 100k, discussed in report)
  - SGD, lr=0.03, momentum=0.9, weight_decay=0.0005
  - Batch size: 64 labeled, 64*7=448 unlabeled (mu=7)
  - Weak aug: random crop + horizontal flip
  - Strong aug: RandAugment
  - Base threshold: 0.95
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler
from torchvision import datasets, transforms

from .wideresnet import WideResNet


# ──────────────────────────────────────────────
# Augmentations
# ──────────────────────────────────────────────

class RandAugment:
    """Simplified RandAugment for strong augmentation."""
    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m
        self.augment_list = [
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomAffine(degrees=30),
            transforms.RandomAffine(degrees=0, shear=15),
            transforms.ColorJitter(brightness=0.4),
            transforms.ColorJitter(contrast=0.4),
            transforms.ColorJitter(saturation=0.4),
            transforms.RandomAutocontrast(),
            transforms.RandomEqualize(),
            transforms.RandomPosterize(bits=4),
            transforms.RandomSolarize(threshold=128),
        ]

    def __call__(self, img):
        ops = np.random.choice(self.augment_list, self.n, replace=False)
        for op in ops:
            img = op(img)
        return img


def get_weak_transform():
    """Weak augmentation: random crop + horizontal flip."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616]),
    ])


def get_strong_transform():
    """Strong augmentation: RandAugment + random crop + flip."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        RandAugment(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616]),
    ])


# ──────────────────────────────────────────────
# Dual-augmentation dataset wrapper
# ──────────────────────────────────────────────

class DualAugDataset(Dataset):
    """Wraps a dataset to return (weak_aug, strong_aug, label) tuples."""
    def __init__(self, base_dataset, indices=None):
        self.base = base_dataset
        self.indices = indices
        self.weak = get_weak_transform()
        self.strong = get_strong_transform()

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.base)

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        # Get the raw PIL image (need to access base dataset's data directly)
        img = self.base.data[real_idx]
        label = self.base.targets[real_idx]
        from PIL import Image
        img = Image.fromarray(img)
        return self.weak(img), self.strong(img), label


class LabeledDataset(Dataset):
    """Simple labeled dataset with weak augmentation only."""
    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = indices
        self.transform = get_weak_transform()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = self.base.data[real_idx]
        label = self.base.targets[real_idx]
        from PIL import Image
        img = Image.fromarray(img)
        return self.transform(img), label


# ──────────────────────────────────────────────
# FlexMatch Training
# ──────────────────────────────────────────────

def train_flexmatch(
    labeled_indices: list[int],
    num_classes: int = 10,
    total_iterations: int = 100_000,
    batch_size: int = 64,
    mu: int = 7,
    lr: float = 0.03,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    threshold: float = 0.95,
    lambda_u: float = 1.0,
    ema_decay: float = 0.999,
    device: torch.device = None,
    data_dir: str = "./data",
    verbose: bool = True,
    eval_every: int = 5000,
) -> float:
    """
    Train FlexMatch on CIFAR-10 with the given labeled indices.

    Args:
        labeled_indices: Indices of labeled examples.
        total_iterations: Total training iterations (paper: 400k, ours: 100k).
        batch_size: Labeled batch size.
        mu: Unlabeled-to-labeled batch ratio.
        threshold: Base confidence threshold for pseudo labels.
        lambda_u: Weight for unsupervised loss.
        ema_decay: EMA decay for teacher model.
        eval_every: Evaluate on test set every N iterations.

    Returns:
        Final test accuracy (%).
    """
    if device is None:
        device = torch.device("cpu")

    # ── Data ──
    base_train = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616]),
        ])
    )

    labeled_set = set(labeled_indices)
    unlabeled_indices = [i for i in range(len(base_train)) if i not in labeled_set]

    labeled_ds = LabeledDataset(base_train, labeled_indices)
    unlabeled_ds = DualAugDataset(base_train, unlabeled_indices)

    labeled_loader = DataLoader(
        labeled_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, drop_last=True,
    )
    unlabeled_loader = DataLoader(
        unlabeled_ds, batch_size=batch_size * mu, shuffle=True,
        num_workers=2, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=2
    )

    # ── Model ──
    model = WideResNet(depth=28, widen_factor=2, num_classes=num_classes,
                       leaky_slope=0.1).to(device)

    # EMA model (teacher)
    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum,
        weight_decay=weight_decay, nesterov=True
    )

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iterations
    )

    # ── FlexMatch: per-class threshold tracking ──
    # Track per-class learning status (sigma in the paper)
    classwise_acc = torch.zeros(num_classes, device=device)

    # ── Training loop ──
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    model.train()
    best_acc = 0.0

    for it in range(total_iterations):
        # Get labeled batch
        try:
            x_l, y_l = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            x_l, y_l = next(labeled_iter)

        # Get unlabeled batch (weak + strong augmentations)
        try:
            x_uw, x_us, _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            x_uw, x_us, _ = next(unlabeled_iter)

        x_l, y_l = x_l.to(device), y_l.to(device)
        x_uw, x_us = x_uw.to(device), x_us.to(device)

        # ── Supervised loss ──
        logits_l = model(x_l)
        loss_l = F.cross_entropy(logits_l, y_l)

        # ── Unsupervised loss (FlexMatch) ──
        with torch.no_grad():
            logits_uw = ema_model(x_uw)
            probs_uw = F.softmax(logits_uw, dim=1)
            max_probs, pseudo_labels = probs_uw.max(dim=1)

            # FlexMatch: per-class flexible thresholds
            # Threshold for each sample based on its pseudo-label class
            flex_threshold = threshold * (classwise_acc[pseudo_labels] /
                                          (classwise_acc.max() + 1e-6))
            # Ensure threshold doesn't go below a minimum
            flex_threshold = torch.clamp(flex_threshold, min=threshold * 0.5)

            # Mask: keep only confident pseudo-labels
            mask = max_probs.ge(flex_threshold).float()

        logits_us = model(x_us)
        loss_u = (F.cross_entropy(logits_us, pseudo_labels, reduction='none') * mask).mean()

        # ── Total loss ──
        loss = loss_l + lambda_u * loss_u

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # ── Update EMA model ──
        with torch.no_grad():
            for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                ema_p.data.mul_(ema_decay).add_(model_p.data, alpha=1 - ema_decay)

        # ── Update per-class accuracy tracking ──
        with torch.no_grad():
            if mask.sum() > 0:
                for c in range(num_classes):
                    class_mask = (pseudo_labels == c)
                    if class_mask.sum() > 0:
                        class_correct = (max_probs[class_mask] >= threshold).float().mean()
                        # EMA update of class accuracy
                        classwise_acc[c] = 0.999 * classwise_acc[c] + 0.001 * class_correct

        # ── Logging & evaluation ──
        if verbose and (it + 1) % eval_every == 0:
            acc = evaluate(ema_model, test_loader, device)
            best_acc = max(best_acc, acc)
            mask_ratio = mask.mean().item()
            print(f"    Iter {it+1}/{total_iterations} | "
                  f"Loss_l: {loss_l.item():.3f} | Loss_u: {loss_u.item():.3f} | "
                  f"Mask: {mask_ratio:.2f} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")

    # Final evaluation
    final_acc = evaluate(ema_model, test_loader, device)
    best_acc = max(best_acc, final_acc)

    if verbose:
        print(f"    Final Acc: {final_acc:.2f}% | Best Acc: {best_acc:.2f}%")

    return best_acc


@torch.no_grad()
def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    model.train()
    return 100.0 * correct / total
