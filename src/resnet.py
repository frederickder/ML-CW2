"""
ResNet-18 adapted for CIFAR-10 (32x32 images).

Standard torchvision ResNet-18 is designed for ImageNet (224x224). For CIFAR-10,
we follow the common practice of modifying the first conv layer:
  - kernel_size 3 instead of 7
  - stride 1 instead of 2
  - remove the initial max pooling layer

This matches the setup used in Van Gansbeke et al. (2020) / SCAN / SimCLR papers
for CIFAR-scale experiments.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18


def cifar_resnet18(num_classes: int = 10) -> nn.Module:
    """
    ResNet-18 modified for CIFAR-10 sized inputs (32x32).
    
    Args:
        num_classes: Number of output classes.
    Returns:
        Modified ResNet-18 model.
    """
    model = resnet18(weights=None)

    # Replace first conv: 7x7/2 → 3x3/1, no initial maxpool
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool

    # Replace the final FC layer for the target number of classes
    in_features = model.fc.in_features  # 512 for ResNet-18
    model.fc = nn.Linear(in_features, num_classes)

    return model


def cifar_resnet18_backbone() -> tuple[nn.Module, int]:
    """
    ResNet-18 backbone for SimCLR (without the final FC layer).
    
    Returns:
        (backbone, feature_dim): The backbone model and its output dimension.
    """
    model = resnet18(weights=None)

    # CIFAR modifications
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    feature_dim = model.fc.in_features  # 512
    model.fc = nn.Identity()  # Remove FC — output is the 512-d embedding

    return model, feature_dim
