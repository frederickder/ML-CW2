"""
Augmentation pipelines for SimCLR training and supervised classifier training.
Follows the paper's setup: SimCLR uses the augmentations from Van Gansbeke et al. (2020).
"""

import torch
from torchvision import transforms


# SimCLR augmentations (contrastive learning)

class SimCLRTransform:
    """
    Generates two augmented views of the same image for contrastive learning.
    Augmentations: random resized crop, horizontal flip, color jitter, grayscale.
    Matches the setup from Van Gansbeke et al. (2020) / Chen et al. (2020).
    """

    def __init__(self, image_size: int = 32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),
        ])

    def __call__(self, x):
        """Return two independently augmented views."""
        return self.transform(x), self.transform(x)


# Supervised classifier augmentations

def get_classifier_train_transform(image_size: int = 32):
    """Training augmentations for the supervised classifier.
    Paper uses: random crops + horizontal flips (Munjal et al. 2020 framework).
    """
    return transforms.Compose([
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])


def get_classifier_test_transform(image_size: int = 32):
    """Test-time transform (no augmentation, just normalise)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])


# Feature extraction transform (for embeddings)

def get_embedding_transform(image_size: int = 32):
    """Transform for extracting features from the trained SimCLR encoder.
    No augmentation — just normalise.
    """
    return get_classifier_test_transform(image_size)
