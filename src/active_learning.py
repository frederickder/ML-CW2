"""
Active Learning experiment loop.

Supports:
  - All query strategies (random, typiclust, uncertainty, margin, entropy,
    coreset, dbal, bald, badge)
  - Framework 1: Fully supervised (ResNet-18 from scratch)
  - Framework 2: Self-supervised embedding (linear classifier on SimCLR features)

Runs multiple AL iterations and (optionally) multiple repetitions.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from .strategies import query, _needs_model
from .classifier import train_classifier, train_linear_classifier
from .augmentations import get_classifier_train_transform, get_classifier_test_transform
from .utils import set_seed


# ──────────────────────────────────────────────
# Framework 1: Fully supervised
# ──────────────────────────────────────────────

def run_al_experiment(
    features: np.ndarray,
    train_dataset: datasets.CIFAR10,
    test_dataset: datasets.CIFAR10,
    strategy: str = "typiclust",
    budget_per_round: int = 10,
    n_rounds: int = 5,
    classifier_epochs: int = 200,
    device: torch.device = None,
    typicality_fn: str = "euclidean",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run a single active learning experiment (Framework 1: fully supervised).
    """
    set_seed(seed)

    labeled_indices = []
    cumulative_budgets = []
    accuracies = []
    queried_per_round = []
    current_model = None  # For model-dependent strategies

    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0
    )

    use_dropout = strategy in ("dbal", "bald")

    for round_idx in range(n_rounds):
        if verbose:
            print(f"\n  AL Round {round_idx + 1}/{n_rounds} | "
                  f"Strategy: {strategy}")

        # ── Query ──
        new_indices = query(
            strategy=strategy,
            features=features,
            labeled_indices=labeled_indices,
            budget=budget_per_round,
            model=current_model,
            dataset=train_dataset,
            device=device,
            typicality_fn=typicality_fn,
            random_state=seed + round_idx,
        )

        labeled_indices.extend(new_indices)
        queried_per_round.append(new_indices)

        if verbose:
            print(f"    Queried {len(new_indices)}, total labeled: {len(labeled_indices)}")

        # ── Train classifier on labeled set ──
        train_subset = Subset(train_dataset, labeled_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=min(64, len(labeled_indices)),
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        final_acc, _, current_model = train_classifier(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=classifier_epochs,
            device=device,
            verbose=False,
            use_dropout=use_dropout,
        )

        cumulative_budgets.append(len(labeled_indices))
        accuracies.append(final_acc)

        if verbose:
            print(f"    -> Accuracy: {final_acc:.2f}% (Budget: {len(labeled_indices)})")

    return {
        "cumulative_budget": cumulative_budgets,
        "accuracies": accuracies,
        "queried_indices": queried_per_round,
    }


# ──────────────────────────────────────────────
# Framework 2: Self-supervised embedding
# ──────────────────────────────────────────────

def run_al_experiment_linear(
    features: np.ndarray,
    labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    strategy: str = "typiclust",
    budget_per_round: int = 10,
    n_rounds: int = 5,
    linear_epochs: int = 200,
    device: torch.device = None,
    typicality_fn: str = "euclidean",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run a single AL experiment (Framework 2: linear classifier on SimCLR features).
    """
    set_seed(seed)

    features_t = torch.from_numpy(features).float()
    labels_t = torch.from_numpy(labels).long()
    test_features_t = torch.from_numpy(test_features).float()
    test_labels_t = torch.from_numpy(test_labels).long()

    labeled_indices = []
    cumulative_budgets = []
    accuracies = []

    for round_idx in range(n_rounds):
        if verbose:
            print(f"\n  AL Round {round_idx + 1}/{n_rounds} | "
                  f"Strategy: {strategy} (linear)")

        # ── Query ──
        # Feature-based strategies work directly; model-dependent strategies
        # fall back to random in this framework (consistent with paper's setup)
        if strategy in ("typiclust", "random", "coreset"):
            new_indices = query(
                strategy=strategy,
                features=features,
                labeled_indices=labeled_indices,
                budget=budget_per_round,
                typicality_fn=typicality_fn,
                random_state=seed + round_idx,
            )
        else:
            from .typicclust import random_query
            new_indices = random_query(
                len(features), labeled_indices, budget_per_round,
                random_state=seed + round_idx,
            )

        labeled_indices.extend(new_indices)

        if verbose:
            print(f"    Queried {len(new_indices)}, total labeled: {len(labeled_indices)}")

        # ── Train linear classifier ──
        acc = train_linear_classifier(
            features=features_t,
            labels=labels_t,
            labeled_indices=labeled_indices,
            test_features=test_features_t,
            test_labels=test_labels_t,
            epochs=linear_epochs,
            device=device,
            verbose=False,
        )

        cumulative_budgets.append(len(labeled_indices))
        accuracies.append(acc)

        if verbose:
            print(f"    -> Accuracy: {acc:.2f}% (Budget: {len(labeled_indices)})")

    return {
        "cumulative_budget": cumulative_budgets,
        "accuracies": accuracies,
    }


# ──────────────────────────────────────────────
# Repeated experiment wrappers
# ──────────────────────────────────────────────

def run_repeated_experiment(
    features: np.ndarray,
    train_dataset: datasets.CIFAR10,
    test_dataset: datasets.CIFAR10,
    strategy: str = "typiclust",
    budget_per_round: int = 10,
    n_rounds: int = 5,
    n_reps: int = 5,
    classifier_epochs: int = 200,
    device: torch.device = None,
    typicality_fn: str = "euclidean",
    base_seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run multiple repetitions of Framework 1 AL experiment."""
    all_accuracies = []

    for rep in range(n_reps):
        if verbose:
            print(f"\n  REP {rep + 1}/{n_reps} | {strategy}")

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

    all_accuracies = np.array(all_accuracies)

    return {
        "strategy": strategy,
        "typicality_fn": typicality_fn,
        "framework": "fully_supervised",
        "cumulative_budget": result["cumulative_budget"],
        "mean_accuracy": all_accuracies.mean(axis=0).tolist(),
        "std_accuracy": all_accuracies.std(axis=0).tolist(),
        "se_accuracy": (all_accuracies.std(axis=0) / np.sqrt(n_reps)).tolist(),
        "all_accuracies": all_accuracies.tolist(),
    }


def run_repeated_experiment_linear(
    features: np.ndarray,
    labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    strategy: str = "typiclust",
    budget_per_round: int = 10,
    n_rounds: int = 5,
    n_reps: int = 5,
    linear_epochs: int = 200,
    device: torch.device = None,
    typicality_fn: str = "euclidean",
    base_seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Run multiple repetitions of Framework 2 AL experiment."""
    all_accuracies = []

    for rep in range(n_reps):
        if verbose:
            print(f"\n  REP {rep + 1}/{n_reps} | {strategy} (linear)")

        result = run_al_experiment_linear(
            features=features,
            labels=labels,
            test_features=test_features,
            test_labels=test_labels,
            strategy=strategy,
            budget_per_round=budget_per_round,
            n_rounds=n_rounds,
            linear_epochs=linear_epochs,
            device=device,
            typicality_fn=typicality_fn,
            seed=base_seed + rep * 1000,
            verbose=verbose,
        )
        all_accuracies.append(result["accuracies"])

    all_accuracies = np.array(all_accuracies)

    return {
        "strategy": strategy,
        "typicality_fn": typicality_fn,
        "framework": "self_supervised_embedding",
        "cumulative_budget": result["cumulative_budget"],
        "mean_accuracy": all_accuracies.mean(axis=0).tolist(),
        "std_accuracy": all_accuracies.std(axis=0).tolist(),
        "se_accuracy": (all_accuracies.std(axis=0) / np.sqrt(n_reps)).tolist(),
        "all_accuracies": all_accuracies.tolist(),
    }
