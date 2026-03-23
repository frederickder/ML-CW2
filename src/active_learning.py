"""
Active Learning experiment loop.

Ties together:
  - Query strategies (TypiClust, Random)
  - Classifier training (ResNet-18)
  - Evaluation (test accuracy per round)

Runs multiple AL iterations and (optionally) multiple repetitions.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from .typicclust import typiclust_query, random_query
from .classifier import train_classifier
from .augmentations import get_classifier_train_transform, get_classifier_test_transform
from .utils import set_seed


def run_al_experiment(
    features: np.ndarray,
    train_dataset: datasets.CIFAR10,
    test_dataset: datasets.CIFAR10,
    strategy: str = "typiclust",
    budget_per_round: int = 10,
    n_rounds: int = 5,
    classifier_epochs: int = 100,
    device: torch.device = None,
    typicality_fn: str = "euclidean",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run a single active learning experiment.
    
    Args:
        features: [N, D] precomputed SimCLR features for the training set.
        train_dataset: CIFAR-10 training dataset (with classifier augmentations).
        test_dataset: CIFAR-10 test dataset.
        strategy: "typiclust" or "random".
        budget_per_round: Number of examples to query per AL round (B).
        n_rounds: Number of AL iterations.
        classifier_epochs: Epochs to train classifier each round.
        device: Compute device.
        typicality_fn: Typicality measure (for TypiClust).
        seed: Random seed.
        verbose: Print progress.
    
    Returns:
        Dict with keys:
          - "cumulative_budget": [n_rounds] list of cumulative labeled counts
          - "accuracies": [n_rounds] list of test accuracies
          - "queried_indices": [n_rounds] list of lists of queried indices
    """
    set_seed(seed)

    labeled_indices = []
    cumulative_budgets = []
    accuracies = []
    queried_per_round = []

    # Test loader (fixed)
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0
    )

    for round_idx in range(n_rounds):
        if verbose:
            print(f"\n{'='*60}")
            print(f"AL Round {round_idx + 1}/{n_rounds} | Strategy: {strategy} | "
                  f"Typicality: {typicality_fn}")
            print(f"{'='*60}")

        # ── Query ──
        if strategy == "typiclust":
            new_indices = typiclust_query(
                features=features,
                labeled_indices=labeled_indices,
                budget=budget_per_round,
                typicality_fn=typicality_fn,
                random_state=seed + round_idx,
            )
        elif strategy == "random":
            new_indices = random_query(
                n_total=len(features),
                labeled_indices=labeled_indices,
                budget=budget_per_round,
                random_state=seed + round_idx,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        labeled_indices.extend(new_indices)
        queried_per_round.append(new_indices)

        if verbose:
            print(f"  Queried {len(new_indices)} examples. "
                  f"Total labeled: {len(labeled_indices)}")

        # ── Train classifier on labeled set ──
        train_subset = Subset(train_dataset, labeled_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=min(64, len(labeled_indices)),
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        final_acc, _ = train_classifier(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=classifier_epochs,
            device=device,
            verbose=verbose,
        )

        cumulative_budgets.append(len(labeled_indices))
        accuracies.append(final_acc)

        if verbose:
            print(f"  → Test Accuracy: {final_acc:.2f}% "
                  f"(Budget: {len(labeled_indices)})")

    return {
        "cumulative_budget": cumulative_budgets,
        "accuracies": accuracies,
        "queried_indices": queried_per_round,
    }


def run_repeated_experiment(
    features: np.ndarray,
    train_dataset: datasets.CIFAR10,
    test_dataset: datasets.CIFAR10,
    strategy: str = "typiclust",
    budget_per_round: int = 10,
    n_rounds: int = 5,
    n_reps: int = 3,
    classifier_epochs: int = 100,
    device: torch.device = None,
    typicality_fn: str = "euclidean",
    base_seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run multiple repetitions of an AL experiment for statistical analysis.
    
    Returns:
        Dict with keys:
          - "cumulative_budget": [n_rounds] budget sizes
          - "mean_accuracy": [n_rounds] mean accuracy across reps
          - "std_accuracy": [n_rounds] std dev across reps
          - "se_accuracy": [n_rounds] standard error across reps
          - "all_accuracies": [n_reps, n_rounds] full results
    """
    all_accuracies = []

    for rep in range(n_reps):
        print(f"\n{'#'*60}")
        print(f"REPETITION {rep + 1}/{n_reps}")
        print(f"{'#'*60}")

        result = run_al_experiment(
            features=features,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            strategy=strategy,
            budget_per_round=budget_per_round,
            n_rounds=n_rounds,
            classifier_epochs=classifier_epochs,
            device=device,
            typicality_fn=typicality_fn,
            seed=base_seed + rep * 1000,
            verbose=verbose,
        )
        all_accuracies.append(result["accuracies"])

    all_accuracies = np.array(all_accuracies)  # [n_reps, n_rounds]

    return {
        "strategy": strategy,
        "typicality_fn": typicality_fn,
        "cumulative_budget": result["cumulative_budget"],
        "mean_accuracy": all_accuracies.mean(axis=0).tolist(),
        "std_accuracy": all_accuracies.std(axis=0).tolist(),
        "se_accuracy": (all_accuracies.std(axis=0) / np.sqrt(n_reps)).tolist(),
        "all_accuracies": all_accuracies.tolist(),
    }
