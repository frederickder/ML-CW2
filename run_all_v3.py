"""
run_all_v3.py — Framework 3: Semi-supervised (FlexMatch)

Runs FlexMatch (Zhang et al., 2021) with labeled sets selected by
different AL strategies on CIFAR-10.

Matches paper Section 4.2.3 / Fig. 6a:
  - 10 labeled examples
  - WideResNet-28-2
  - 50k iterations (paper: 400k — reduced for compute, discussed in report)
  - 1 repetition (paper: 3 — reduced for compute, discussed in report)

Requires: models/simclr_features.npy (from run_all.py)

Usage:
    nohup python run_all_v3.py > output.log 2>&1 &
    tail -f output.log
"""

import time
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.strategies import query
from src.flexmatch import train_flexmatch
from src.utils import get_device, set_seed, MODELS_DIR, DATA_DIR, RESULTS_DIR, FIGURES_DIR, save_results

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════
BUDGET = 10                   # 10 labeled examples (1 per class, Fig. 6a)
FLEXMATCH_ITERATIONS = 50_000 # Paper: 400k — reduced for compute
N_REPS = 1                    # Paper: 3 — reduced for compute

# Key comparison: Random vs TypiClust (core finding of the paper)
STRATEGIES = [
    "random",
    "typiclust",
]
# ══════════════════════════════════════════════


def select_labeled_indices(features, strategy, budget, seed=42):
    """Select labeled indices using the given strategy (single round)."""
    return query(
        strategy=strategy,
        features=features,
        labeled_indices=[],
        budget=budget,
        typicality_fn="euclidean",
        random_state=seed,
    )


def main():
    total_start = time.time()
    set_seed(42)
    device = get_device()

    # ── Load features ──
    features_path = str(MODELS_DIR / "simclr_features.npy")
    if not os.path.exists(features_path):
        print("ERROR: simclr_features.npy not found. Run run_all.py first.")
        sys.exit(1)

    features = np.load(features_path)
    print(f"Loaded features: {features.shape}", flush=True)

    # ── Run experiments ──
    print(f"\n{'='*70}", flush=True)
    print(f"Framework 3: Semi-Supervised (FlexMatch)", flush=True)
    print(f"Budget: {BUDGET} labels | Iterations: {FLEXMATCH_ITERATIONS}", flush=True)
    print(f"{'='*70}", flush=True)

    all_results = {}

    for strategy in STRATEGIES:
        result_file = f"fw3_{strategy}_b{BUDGET}.json"
        result_path = RESULTS_DIR / result_file

        if result_path.exists():
            print(f"\n  {strategy}: results found, skipping.", flush=True)
            from src.utils import load_results
            all_results[strategy] = load_results(result_file)
            continue

        print(f"\n{'─'*60}", flush=True)
        print(f"  Strategy: {strategy}", flush=True)
        print(f"{'─'*60}", flush=True)

        rep_accuracies = []

        for rep in range(N_REPS):
            seed = 42 + rep * 1000
            set_seed(seed)

            # Select labeled examples
            labeled_indices = select_labeled_indices(features, strategy, BUDGET, seed=seed)
            print(f"\n  Rep {rep+1}/{N_REPS} | Selected indices: {labeled_indices}", flush=True)

            # Check class coverage
            from torchvision import datasets as tv_datasets
            base = tv_datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True)
            selected_labels = [base.targets[i] for i in labeled_indices]
            n_classes_covered = len(set(selected_labels))
            print(f"    Classes covered: {n_classes_covered}/10 | "
                  f"Labels: {sorted(selected_labels)}", flush=True)

            # Train FlexMatch
            t0 = time.time()
            acc = train_flexmatch(
                labeled_indices=labeled_indices,
                num_classes=10,
                total_iterations=FLEXMATCH_ITERATIONS,
                batch_size=64,
                mu=7,
                lr=0.03,
                threshold=0.95,
                device=device,
                data_dir=str(DATA_DIR),
                verbose=True,
                eval_every=5000,
            )
            elapsed = (time.time() - t0) / 60
            print(f"    Accuracy: {acc:.2f}% | Time: {elapsed:.1f} min", flush=True)
            rep_accuracies.append(acc)

        results = {
            "strategy": strategy,
            "framework": "semi_supervised",
            "budget": BUDGET,
            "iterations": FLEXMATCH_ITERATIONS,
            "accuracies": rep_accuracies,
            "mean_accuracy": float(np.mean(rep_accuracies)),
            "std_accuracy": float(np.std(rep_accuracies)) if len(rep_accuracies) > 1 else 0.0,
        }
        save_results(results, result_file)
        all_results[strategy] = results

    # ── Generate plot (bar chart matching paper Fig. 6a style) ──
    print(f"\n{'='*70}", flush=True)
    print("Generating Framework 3 plot", flush=True)
    print(f"{'='*70}", flush=True)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    strategies_sorted = sorted(all_results.keys())
    x_pos = np.arange(len(strategies_sorted))
    means = [all_results[s]["mean_accuracy"] for s in strategies_sorted]
    stds = [all_results[s].get("std_accuracy", 0) for s in strategies_sorted]

    colors = ["tab:blue" if s == "typiclust" else "tab:gray" for s in strategies_sorted]

    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([s.capitalize() for s in strategies_sorted])
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"Framework 3: FlexMatch with {BUDGET} Labels (CIFAR-10)")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "fig_fw3_flexmatch.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── Summary ──
    print(f"\n{'='*70}", flush=True)
    print("FRAMEWORK 3 RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    for s in strategies_sorted:
        r = all_results[s]
        print(f"  {s:15s}: {r['mean_accuracy']:.2f}%", flush=True)

    total_min = (time.time() - total_start) / 60
    print(f"\nTotal runtime: {total_min:.1f} min", flush=True)


if __name__ == "__main__":
    main()
