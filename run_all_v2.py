"""
run_all_v2.py — Second RunPod run: all baselines + Framework 2.

Assumes run_all.py has already been run and produced:
  - models/simclr_cifar10.pth
  - models/simclr_features.npy
  - models/simclr_labels.npy
  - results/ from the first run (TypiClust variants + random)

This script adds:
  - Framework 1: All remaining baselines (uncertainty, margin, entropy,
    coreset, dbal, bald, badge)
  - Framework 2: Linear classifier on SimCLR features for key strategies

Usage:
    python run_all_v2.py
"""

import time
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.simclr import SimCLR, extract_features
from src.augmentations import (
    get_classifier_train_transform, get_classifier_test_transform,
    get_embedding_transform,
)
from src.active_learning import run_repeated_experiment, run_repeated_experiment_linear
from src.utils import get_device, set_seed, MODELS_DIR, DATA_DIR, RESULTS_DIR, FIGURES_DIR, save_results

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
CLASSIFIER_EPOCHS = 200
LINEAR_EPOCHS = 200       # Paper: 2x the supervised epochs
BUDGET_PER_ROUND = 10
N_ROUNDS = 5
N_REPS = 5

# Framework 1 baselines still to run
FW1_STRATEGIES = [
    "uncertainty", "margin", "entropy",
    "coreset", "dbal", "bald", "badge",
]

# Framework 2 strategies (paper Fig. 5)
FW2_STRATEGIES = [
    "typiclust", "random", "coreset",
    "uncertainty", "margin", "entropy",
    "dbal", "bald", "badge",
]
# ══════════════════════════════════════════════


def main():
    total_start = time.time()
    set_seed(42)
    device = get_device()

    # ──────────────────────────────────────────
    # Load precomputed features
    # ──────────────────────────────────────────
    features_path = str(MODELS_DIR / "simclr_features.npy")
    labels_path = str(MODELS_DIR / "simclr_labels.npy")

    if not os.path.exists(features_path):
        print("ERROR: simclr_features.npy not found. Run run_all.py first.")
        sys.exit(1)

    features = np.load(features_path)
    labels = np.load(labels_path)
    print(f"Loaded features: {features.shape}, labels: {labels.shape}")

    # ──────────────────────────────────────────
    # Load datasets
    # ──────────────────────────────────────────
    train_dataset = datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=True,
        transform=get_classifier_train_transform()
    )
    test_dataset = datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=True,
        transform=get_classifier_test_transform()
    )

    # ──────────────────────────────────────────
    # PHASE 1: Framework 1 — remaining baselines
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1: Framework 1 — Additional Baselines")
    print("=" * 70)

    for strategy in FW1_STRATEGIES:
        result_file = f"fw1_{strategy}_b10.json"
        result_path = RESULTS_DIR / result_file

        if result_path.exists():
            print(f"\n  {strategy}: results found, skipping.")
            continue

        print(f"\n{'─'*60}")
        print(f"  Running Framework 1: {strategy}")
        print(f"{'─'*60}")

        t0 = time.time()
        results = run_repeated_experiment(
            features=features,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            strategy=strategy,
            budget_per_round=BUDGET_PER_ROUND,
            n_rounds=N_ROUNDS,
            n_reps=N_REPS,
            classifier_epochs=CLASSIFIER_EPOCHS,
            device=device,
            base_seed=42,
            verbose=True,
        )
        save_results(results, result_file)
        elapsed = (time.time() - t0) / 60
        print(f"  {strategy} took {elapsed:.1f} min")
        for b, a, s in zip(results["cumulative_budget"],
                           results["mean_accuracy"],
                           results["se_accuracy"]):
            print(f"    Budget {b}: {a:.2f}% +/- {s:.2f}%")

    # ──────────────────────────────────────────
    # PHASE 2: Extract test features (for Framework 2)
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: Extract Test Features for Framework 2")
    print("=" * 70)

    test_features_path = str(MODELS_DIR / "simclr_test_features.npy")
    test_labels_path = str(MODELS_DIR / "simclr_test_labels.npy")

    if os.path.exists(test_features_path):
        print("Test features found, loading...")
        test_features = np.load(test_features_path)
        test_labels_np = np.load(test_labels_path)
    else:
        model = SimCLR(feature_dim=512, projection_dim=128)
        model.load_state_dict(
            torch.load(str(MODELS_DIR / "simclr_cifar10.pth"),
                       map_location=device, weights_only=True)
        )
        model.to(device)

        test_dataset_embed = datasets.CIFAR10(
            root=str(DATA_DIR), train=False, download=False,
            transform=get_embedding_transform()
        )
        test_loader = DataLoader(test_dataset_embed, batch_size=256,
                                 shuffle=False, num_workers=4)

        test_feats_t, test_labs_t = extract_features(model, test_loader, device)
        test_features = test_feats_t.numpy()
        test_labels_np = test_labs_t.numpy()

        np.save(test_features_path, test_features)
        np.save(test_labels_path, test_labels_np)
        print(f"Test features saved: {test_features.shape}")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ──────────────────────────────────────────
    # PHASE 3: Framework 2 — linear classifier
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: Framework 2 — Self-Supervised Embedding")
    print("=" * 70)

    for strategy in FW2_STRATEGIES:
        result_file = f"fw2_{strategy}_b10.json"
        result_path = RESULTS_DIR / result_file

        if result_path.exists():
            print(f"\n  {strategy} (linear): results found, skipping.")
            continue

        print(f"\n{'─'*60}")
        print(f"  Running Framework 2: {strategy}")
        print(f"{'─'*60}")

        t0 = time.time()
        results = run_repeated_experiment_linear(
            features=features,
            labels=labels,
            test_features=test_features,
            test_labels=test_labels_np,
            strategy=strategy,
            budget_per_round=BUDGET_PER_ROUND,
            n_rounds=N_ROUNDS,
            n_reps=N_REPS,
            linear_epochs=LINEAR_EPOCHS,
            device=device,
            typicality_fn="euclidean",
            base_seed=42,
            verbose=True,
        )
        save_results(results, result_file)
        elapsed = (time.time() - t0) / 60
        print(f"  {strategy} (linear) took {elapsed:.1f} min")

    # ──────────────────────────────────────────
    # PHASE 4: Generate all plots
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 4: Generating Plots")
    print("=" * 70)

    from src.utils import load_results

    all_results = {}
    for f in RESULTS_DIR.glob("*.json"):
        all_results[f.stem] = load_results(f.name)
    print(f"Loaded {len(all_results)} result files.")

    # ── Plot: Framework 1 — All strategies ──
    fig, ax = plt.subplots(figsize=(10, 6))
    color_cycle = plt.cm.tab10.colors
    strategies_fw1 = {}

    # Collect Framework 1 results
    for key, r in all_results.items():
        if key.startswith("fw1_") or key in ("typiclust_euclidean_b10", "random_b10"):
            if key.startswith("fw1_"):
                name = key.replace("fw1_", "").replace("_b10", "")
            elif "typiclust" in key:
                name = "typiclust"
            else:
                name = "random"
            strategies_fw1[name] = r

    for i, (name, r) in enumerate(sorted(strategies_fw1.items())):
        b = r["cumulative_budget"]
        m = np.array(r["mean_accuracy"])
        s = np.array(r["se_accuracy"])
        color = color_cycle[i % len(color_cycle)]
        ls = "--" if name == "random" else "-"
        lw = 2.0 if name in ("typiclust", "random") else 1.2
        ax.plot(b, m, f"o{ls}", label=name.capitalize(), color=color,
                markersize=5, linewidth=lw)
        ax.fill_between(b, m - s, m + s, alpha=0.1, color=color)

    ax.set_xlabel("Cumulative Budget")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Framework 1: Fully Supervised — All Strategies (CIFAR-10)")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "fig_fw1_all_strategies.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── Plot: Framework 2 — supported feature-based strategies only ──
    fig, ax = plt.subplots(figsize=(10, 6))
    strategies_fw2 = {}
    for key, r in all_results.items():
        if key.startswith("fw2_"):
            name = key.replace("fw2_", "").replace("_b10", "")
            if name in ("typiclust", "random", "coreset"):
                strategies_fw2[name] = r

    for i, (name, r) in enumerate(sorted(strategies_fw2.items())):
        b = r["cumulative_budget"]
        m = np.array(r["mean_accuracy"])
        s = np.array(r["se_accuracy"])
        color = color_cycle[i % len(color_cycle)]
        ls = "--" if name == "random" else "-"
        lw = 2.0 if name in ("typiclust", "random") else 1.2
        ax.plot(b, m, f"o{ls}", label=name.capitalize(), color=color,
                markersize=5, linewidth=lw)
        ax.fill_between(b, m - s, m + s, alpha=0.1, color=color)

    ax.set_xlabel("Cumulative Budget")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Framework 2: Self-Supervised Embedding — Supported Feature-Based Strategies")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "fig_fw2_all_strategies.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── Print full summary ──
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for key in sorted(all_results.keys()):
        r = all_results[key]
        if "se_accuracy" not in r or "cumulative_budget" not in r:
            continue
        m = r["mean_accuracy"]
        s = r["se_accuracy"]
        b = r["cumulative_budget"]
        print(f"\n{key}:")
        for i in range(len(b)):
            print(f"  Budget {b[i]:3d}: {m[i]:.2f}% +/- {s[i]:.2f}%")

    total_min = (time.time() - total_start) / 60
    print(f"\n{'=' * 70}")
    print(f"TOTAL RUNTIME: {total_min:.1f} min")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
