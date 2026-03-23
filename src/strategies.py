"""
Active Learning query strategies — all baselines from Hacohen et al. (2022).

Model-independent (can be used from round 1):
  - random: Uniform random selection
  - typiclust: TypiClust (our main method)
  - coreset: Greedy k-center on features (Sener & Savarese, 2018)

Model-dependent (need a trained classifier — use random for round 1):
  - uncertainty: Lowest max softmax (Lewis & Gale, 1994)
  - margin: Smallest margin between top-2 softmax
  - entropy: Highest softmax entropy
  - dbal: MC Dropout uncertainty (Gal et al., 2017)
  - bald: Bayesian AL by Disagreement (Kirsch et al., 2019)
  - badge: Gradient embedding diversity (Ash et al., 2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .typicclust import typiclust_query, random_query


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _needs_model(strategy: str) -> bool:
    """Does this strategy require a trained model to compute queries?"""
    return strategy in ("uncertainty", "margin", "entropy", "dbal", "bald", "badge")


@torch.no_grad()
def _get_softmax_outputs(
    model: nn.Module,
    dataset,
    indices: list[int],
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Get softmax probabilities for a subset of the dataset."""
    model.eval()
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []
    for images, _ in loader:
        images = images.to(device)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu())

    return torch.cat(all_probs, dim=0)  # [len(indices), num_classes]


def _get_mc_dropout_outputs(
    model: nn.Module,
    dataset,
    indices: list[int],
    device: torch.device,
    n_forward: int = 10,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Get MC Dropout softmax outputs: run T forward passes with dropout enabled.
    Returns: [len(indices), T, num_classes]
    """
    # Enable dropout during inference
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_outputs = []
    for t in range(n_forward):
        batch_probs = []
        for images, _ in loader:
            images = images.to(device)
            with torch.no_grad():
                logits = model(images)
                probs = F.softmax(logits, dim=1)
            batch_probs.append(probs.cpu())
        all_outputs.append(torch.cat(batch_probs, dim=0))

    # Restore eval mode
    model.eval()

    return torch.stack(all_outputs, dim=1)  # [N, T, C]


# ──────────────────────────────────────────────
# Uncertainty-based strategies
# ──────────────────────────────────────────────

def uncertainty_query(
    model: nn.Module,
    dataset,
    labeled_indices: list[int],
    budget: int,
    device: torch.device,
) -> list[int]:
    """Select examples with lowest max softmax probability."""
    labeled_set = set(labeled_indices)
    unlabeled = [i for i in range(len(dataset)) if i not in labeled_set]

    probs = _get_softmax_outputs(model, dataset, unlabeled, device)
    max_probs, _ = probs.max(dim=1)  # [N_unlabeled]

    # Lowest confidence = most uncertain
    _, sorted_idx = max_probs.sort()
    selected_local = sorted_idx[:budget].tolist()
    return [unlabeled[i] for i in selected_local]


def margin_query(
    model: nn.Module,
    dataset,
    labeled_indices: list[int],
    budget: int,
    device: torch.device,
) -> list[int]:
    """Select examples with smallest margin between top-2 softmax outputs."""
    labeled_set = set(labeled_indices)
    unlabeled = [i for i in range(len(dataset)) if i not in labeled_set]

    probs = _get_softmax_outputs(model, dataset, unlabeled, device)
    top2, _ = probs.topk(2, dim=1)
    margins = top2[:, 0] - top2[:, 1]  # [N_unlabeled]

    # Smallest margin = most uncertain
    _, sorted_idx = margins.sort()
    selected_local = sorted_idx[:budget].tolist()
    return [unlabeled[i] for i in selected_local]


def entropy_query(
    model: nn.Module,
    dataset,
    labeled_indices: list[int],
    budget: int,
    device: torch.device,
) -> list[int]:
    """Select examples with highest softmax entropy."""
    labeled_set = set(labeled_indices)
    unlabeled = [i for i in range(len(dataset)) if i not in labeled_set]

    probs = _get_softmax_outputs(model, dataset, unlabeled, device)
    # Entropy = -sum(p * log(p))
    log_probs = torch.log(probs + 1e-10)
    entropies = -(probs * log_probs).sum(dim=1)  # [N_unlabeled]

    # Highest entropy = most uncertain
    _, sorted_idx = entropies.sort(descending=True)
    selected_local = sorted_idx[:budget].tolist()
    return [unlabeled[i] for i in selected_local]


# ──────────────────────────────────────────────
# CoreSet (Sener & Savarese, 2018)
# ──────────────────────────────────────────────

def coreset_query(
    features: np.ndarray,
    labeled_indices: list[int],
    budget: int,
) -> list[int]:
    """
    Greedy k-center CoreSet selection.
    Pick the unlabeled point farthest from the current labeled set.
    """
    labeled_set = set(labeled_indices)
    unlabeled = np.array([i for i in range(len(features)) if i not in labeled_set])

    if len(labeled_indices) == 0:
        # No labeled points yet — pick a random starting point
        first = np.random.choice(unlabeled)
        selected = [int(first)]
        labeled_set.add(int(first))
        unlabeled = np.array([i for i in unlabeled if i != first])
        remaining_budget = budget - 1
    else:
        selected = []
        remaining_budget = budget

    if remaining_budget == 0 or len(unlabeled) == 0:
        return selected

    # Compute distance from each unlabeled point to nearest labeled point
    labeled_arr = np.array(list(labeled_set))
    labeled_feats = features[labeled_arr]  # [L, D]

    # Initialize min distances
    # Compute all distances at once: [N_unlabeled, L]
    unlabeled_feats = features[unlabeled]  # [U, D]

    # Chunked distance computation to manage memory
    chunk_size = 5000
    min_dists = np.full(len(unlabeled), np.inf)

    for i in range(0, len(labeled_arr), chunk_size):
        chunk = labeled_feats[i:i + chunk_size]  # [chunk, D]
        # Squared Euclidean distance
        dists = (
            np.sum(unlabeled_feats ** 2, axis=1, keepdims=True)
            + np.sum(chunk ** 2, axis=1)
            - 2 * unlabeled_feats @ chunk.T
        )  # [U, chunk]
        min_dists = np.minimum(min_dists, dists.min(axis=1))

    for _ in range(remaining_budget):
        if len(unlabeled) == 0:
            break

        # Select the point farthest from any labeled point
        best_local = np.argmax(min_dists)
        best_global = int(unlabeled[best_local])
        selected.append(best_global)

        # Update min distances with the newly selected point
        new_feat = features[best_global:best_global + 1]  # [1, D]
        new_dists = (
            np.sum(unlabeled_feats ** 2, axis=1)
            + np.sum(new_feat ** 2)
            - 2 * (unlabeled_feats @ new_feat.T).squeeze()
        )
        min_dists = np.minimum(min_dists, new_dists)

        # Mark as selected (set distance to -inf so it's not picked again)
        min_dists[best_local] = -1

    return selected


# ──────────────────────────────────────────────
# DBAL — MC Dropout Uncertainty (Gal et al., 2017)
# ──────────────────────────────────────────────

def dbal_query(
    model: nn.Module,
    dataset,
    labeled_indices: list[int],
    budget: int,
    device: torch.device,
    n_forward: int = 10,
) -> list[int]:
    """
    Deep Bayesian Active Learning.
    Use MC Dropout to estimate uncertainty (variation ratio or predictive entropy).
    Select the most uncertain points.
    """
    labeled_set = set(labeled_indices)
    unlabeled = [i for i in range(len(dataset)) if i not in labeled_set]

    # [N_unlabeled, T, C]
    mc_outputs = _get_mc_dropout_outputs(model, dataset, unlabeled, device, n_forward)

    # Predictive entropy of the mean prediction
    mean_probs = mc_outputs.mean(dim=1)  # [N, C]
    entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)

    _, sorted_idx = entropy.sort(descending=True)
    selected_local = sorted_idx[:budget].tolist()
    return [unlabeled[i] for i in selected_local]


# ──────────────────────────────────────────────
# BALD — Bayesian Active Learning by Disagreement
# (Kirsch et al., 2019)
# ──────────────────────────────────────────────

def bald_query(
    model: nn.Module,
    dataset,
    labeled_indices: list[int],
    budget: int,
    device: torch.device,
    n_forward: int = 10,
) -> list[int]:
    """
    BALD: Mutual information between predictions and model parameters.
    I(y; w | x) = H(y|x) - E_w[H(y|x,w)]
    = predictive_entropy - mean_of_per_sample_entropies
    """
    labeled_set = set(labeled_indices)
    unlabeled = [i for i in range(len(dataset)) if i not in labeled_set]

    mc_outputs = _get_mc_dropout_outputs(model, dataset, unlabeled, device, n_forward)

    # Predictive entropy: H(y|x)
    mean_probs = mc_outputs.mean(dim=1)  # [N, C]
    predictive_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum(dim=1)

    # Expected entropy: E_w[H(y|x,w)]
    per_sample_entropy = -(mc_outputs * torch.log(mc_outputs + 1e-10)).sum(dim=2)  # [N, T]
    expected_entropy = per_sample_entropy.mean(dim=1)  # [N]

    # BALD score = mutual information
    bald_scores = predictive_entropy - expected_entropy

    _, sorted_idx = bald_scores.sort(descending=True)
    selected_local = sorted_idx[:budget].tolist()
    return [unlabeled[i] for i in selected_local]


# ──────────────────────────────────────────────
# BADGE — Batch Active learning by Diverse
# Gradient Embeddings (Ash et al., 2020)
# ──────────────────────────────────────────────

def badge_query(
    model: nn.Module,
    dataset,
    labeled_indices: list[int],
    budget: int,
    device: torch.device,
) -> list[int]:
    """
    BADGE: Compute gradient embeddings for unlabeled points,
    then use k-means++ initialization to select diverse embeddings.
    
    Gradient embedding for point x with predicted class y_hat:
    g_x = (e_{y_hat} - p) ⊗ h  where h is the penultimate layer output,
    p is the softmax output, and e_{y_hat} is the one-hot of the predicted class.
    """
    labeled_set = set(labeled_indices)
    unlabeled = [i for i in range(len(dataset)) if i not in labeled_set]

    model.eval()
    subset = Subset(dataset, unlabeled)
    loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=0)

    # Get gradient embeddings
    grad_embeddings = []

    # Hook to capture penultimate layer output
    penultimate_outputs = []

    def hook_fn(module, input, output):
        penultimate_outputs.append(input[0].detach())

    # Register hook on the final FC layer
    handle = model.fc.register_forward_hook(hook_fn)

    for images, _ in loader:
        images = images.to(device)
        penultimate_outputs.clear()

        with torch.no_grad():
            logits = model(images)
            probs = F.softmax(logits, dim=1)  # [B, C]

        h = penultimate_outputs[0]  # [B, D] penultimate features
        y_hat = logits.argmax(dim=1)  # [B]

        # e_{y_hat} - p
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, y_hat.unsqueeze(1), 1.0)
        diff = one_hot - probs  # [B, C]

        # Gradient embedding: outer product → [B, C*D]
        # (flatten the outer product)
        grad_emb = (diff.unsqueeze(2) * h.unsqueeze(1)).reshape(h.size(0), -1)
        grad_embeddings.append(grad_emb.cpu().numpy())

    handle.remove()

    grad_embeddings = np.concatenate(grad_embeddings, axis=0)  # [U, C*D]

    # k-means++ initialization to select budget points
    selected_local = _kmeans_plus_plus(grad_embeddings, budget)
    return [unlabeled[i] for i in selected_local]


def _kmeans_plus_plus(data: np.ndarray, k: int) -> list[int]:
    """k-means++ initialization: select k diverse points."""
    n = data.shape[0]
    if k >= n:
        return list(range(n))

    # First center: random
    centers = [np.random.randint(n)]
    min_dists = np.full(n, np.inf)

    for _ in range(k - 1):
        # Update distances to nearest center
        new_center = data[centers[-1]:centers[-1] + 1]
        dists = np.sum((data - new_center) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)

        # Don't re-select existing centers
        min_dists_copy = min_dists.copy()
        for c in centers:
            min_dists_copy[c] = 0

        # Select next center proportional to distance^2
        probs = min_dists_copy / (min_dists_copy.sum() + 1e-10)
        next_center = np.random.choice(n, p=probs)
        centers.append(next_center)

    return centers


# ──────────────────────────────────────────────
# Unified query interface
# ──────────────────────────────────────────────

def query(
    strategy: str,
    features: np.ndarray,
    labeled_indices: list[int],
    budget: int,
    model: nn.Module = None,
    dataset=None,
    device: torch.device = None,
    typicality_fn: str = "euclidean",
    random_state: int = 42,
) -> list[int]:
    """
    Unified interface for all query strategies.
    
    Model-dependent strategies fall back to random if no model is provided
    (cold start / round 1).
    """
    # Cold start: model-dependent strategies use random for first round
    if _needs_model(strategy) and model is None:
        return random_query(len(features), labeled_indices, budget, random_state)

    if strategy == "random":
        return random_query(len(features), labeled_indices, budget, random_state)

    elif strategy == "typiclust":
        return typiclust_query(
            features, labeled_indices, budget,
            typicality_fn=typicality_fn, random_state=random_state,
        )

    elif strategy == "uncertainty":
        return uncertainty_query(model, dataset, labeled_indices, budget, device)

    elif strategy == "margin":
        return margin_query(model, dataset, labeled_indices, budget, device)

    elif strategy == "entropy":
        return entropy_query(model, dataset, labeled_indices, budget, device)

    elif strategy == "coreset":
        return coreset_query(features, labeled_indices, budget)

    elif strategy == "dbal":
        return dbal_query(model, dataset, labeled_indices, budget, device)

    elif strategy == "bald":
        return bald_query(model, dataset, labeled_indices, budget, device)

    elif strategy == "badge":
        return badge_query(model, dataset, labeled_indices, budget, device)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
