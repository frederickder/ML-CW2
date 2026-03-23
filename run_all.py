"""
run_all.py — Run the full TypiClust pipeline on a GPU instance.

Usage on RunPod:
    python run_all.py

Produces:
    models/simclr_cifar10.pth      (~44MB — the trained SimCLR encoder)
    models/simclr_features.npy     (~100MB — precomputed 512-d features)
    models/simclr_labels.npy       (~200KB)
    results/*.json                 (experiment results — small)
    figures/*.png                  (plots — small)
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

# ── Setup ──
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.simclr import SimCLR, train_simclr, extract_features
from src.augmentations import (
    SimCLRTransform, get_embedding_transform,
    get_classifier_train_transform, get_classifier_test_transform,
)
from src.active_learning import run_repeated_experiment
from src.utils import get_device, set_seed, MODELS_DIR, DATA_DIR, RESULTS_DIR, FIGURES_DIR, save_results

# ══════════════════════════════════════════════
# CONFIG — adjust these as needed
# ══════════════════════════════════════════════
SIMCLR_EPOCHS = 200       # Paper uses 500; bump if you have time
SIMCLR_BATCH_SIZE = 512
SIMCLR_LR = 0.4

CLASSIFIER_EPOCHS = 100   # Paper uses 200
BUDGET_PER_ROUND = 10     # B = M (number of classes)
N_ROUNDS = 5
N_REPS = 3                # Paper uses 10; bump to 5 if time allows

TYPICALITY_VARIANTS = ["euclidean", "cosine", "lof", "kde"]
# ══════════════════════════════════════════════

def main():
    total_start = time.time()
    set_seed(42)
    device = get_device()

    # ──────────────────────────────────────────
    # PHASE 1: SimCLR Training
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1: SimCLR Training")
    print("=" * 70)

    simclr_path = str(MODELS_DIR / "simclr_cifar10.pth")

    if os.path.exists(simclr_path):
        print(f"SimCLR checkpoint found at {simclr_path}, skipping training.")
    else:
        simclr_transform = SimCLRTransform(image_size=32)
        train_dataset_simclr = datasets.CIFAR10(
            root=str(DATA_DIR), train=True, download=True, transform=simclr_transform
        )
        simclr_loader = DataLoader(
            train_dataset_simclr,
            batch_size=SIMCLR_BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        model = SimCLR(feature_dim=512, projection_dim=128)
        t0 = time.time()
        losses = train_simclr(
            model=model,
            dataloader=simclr_loader,
            epochs=SIMCLR_EPOCHS,
            lr=SIMCLR_LR,
            weight_decay=1e-4,
            temperature=0.5,
            device=device,
            save_path=simclr_path,
        )
        print(f"SimCLR training took {(time.time() - t0) / 60:.1f} min")

        # Save loss curve
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 4))
        plt.plot(losses)
        plt.xlabel("Epoch"); plt.ylabel("NT-Xent Loss")
        plt.title(f"SimCLR Training Loss ({SIMCLR_EPOCHS} epochs)")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(str(FIGURES_DIR / "simclr_training_loss.png"), dpi=150)
        plt.close()

    # ──────────────────────────────────────────
    # PHASE 2: Feature Extraction
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 2: Feature Extraction")
    print("=" * 70)

    features_path = str(MODELS_DIR / "simclr_features.npy")

    if os.path.exists(features_path):
        print("Features found, loading...")
        features = np.load(features_path)
    else:
        model = SimCLR(feature_dim=512, projection_dim=128)
        model.load_state_dict(torch.load(simclr_path, map_location=device, weights_only=True))
        model.to(device)

        embed_transform = get_embedding_transform()
        train_dataset_embed = datasets.CIFAR10(
            root=str(DATA_DIR), train=True, download=False, transform=embed_transform
        )
        embed_loader = DataLoader(
            train_dataset_embed, batch_size=256, shuffle=False, num_workers=4
        )

        features_tensor, labels_tensor = extract_features(model, embed_loader, device=device)
        features = features_tensor.numpy()
        np.save(features_path, features)
        np.save(str(MODELS_DIR / "simclr_labels.npy"), labels_tensor.numpy())
        print(f"Features saved: {features.shape}")

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ──────────────────────────────────────────
    # PHASE 3: Active Learning Experiments
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: Active Learning Experiments")
    print("=" * 70)

    train_dataset = datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=True,
        transform=get_classifier_train_transform()
    )
    test_dataset = datasets.CIFAR10(
        root=str(DATA_DIR), train=False, download=True,
        transform=get_classifier_test_transform()
    )

    # ── Random baseline ──
    random_results_path = RESULTS_DIR / "random_b10.json"
    if random_results_path.exists():
        print("Random baseline results found, skipping.")
    else:
        print("\n>>> Running Random baseline...")
        t0 = time.time()
        results_random = run_repeated_experiment(
            features=features,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            strategy="random",
            budget_per_round=BUDGET_PER_ROUND,
            n_rounds=N_ROUNDS,
            n_reps=N_REPS,
            classifier_epochs=CLASSIFIER_EPOCHS,
            device=device,
            base_seed=42,
            verbose=False,
        )
        save_results(results_random, "random_b10.json")
        print(f"Random baseline took {(time.time() - t0) / 60:.1f} min")

    # ── TypiClust variants ──
    for typ_fn in TYPICALITY_VARIANTS:
        result_file = f"typiclust_{typ_fn}_b10.json"
        result_path = RESULTS_DIR / result_file

        if result_path.exists():
            print(f"Results for {typ_fn} found, skipping.")
            continue

        print(f"\n>>> Running TypiClust ({typ_fn})...")
        t0 = time.time()
        results = run_repeated_experiment(
            features=features,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            strategy="typiclust",
            budget_per_round=BUDGET_PER_ROUND,
            n_rounds=N_ROUNDS,
            n_reps=N_REPS,
            classifier_epochs=CLASSIFIER_EPOCHS,
            device=device,
            typicality_fn=typ_fn,
            base_seed=42,
            verbose=False,
        )
        save_results(results, result_file)
        print(f"TypiClust ({typ_fn}) took {(time.time() - t0) / 60:.1f} min")

    # ──────────────────────────────────────────
    # PHASE 4: Generate Plots
    # ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 4: Generating Plots")
    print("=" * 70)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils import load_results

    # Load all results
    all_results = {}
    for f in RESULTS_DIR.glob("*.json"):
        all_results[f.stem] = load_results(f.name)

    # ── Plot 1: TypiClust vs Random ──
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, color, marker, label, ls in [
        ("typiclust_euclidean_b10", "tab:blue", "o", "TypiClust", "-"),
        ("random_b10", "tab:gray", "s", "Random", "--"),
    ]:
        if key in all_results:
            r = all_results[key]
            b = r["cumulative_budget"]
            m = np.array(r["mean_accuracy"])
            s = np.array(r["se_accuracy"])
            ax.plot(b, m, f"{marker}{ls}", label=label, color=color, markersize=6)
            ax.fill_between(b, m - s, m + s, alpha=0.2, color=color)
    ax.set_xlabel("Cumulative Budget")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("CIFAR-10: TypiClust vs Random (Fully Supervised)")
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "fig1_main.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── Plot 2: All typicality variants ──
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "euclidean": ("tab:blue", "o"),
        "cosine": ("tab:orange", "^"),
        "lof": ("tab:green", "s"),
        "kde": ("tab:red", "D"),
    }
    for key, r in all_results.items():
        if "typiclust_" in key:
            t = key.replace("typiclust_", "").replace("_b10", "")
            if t in styles:
                c, mk = styles[t]
                b = r["cumulative_budget"]
                m = np.array(r["mean_accuracy"])
                s = np.array(r["se_accuracy"])
                ax.plot(b, m, f"{mk}-", label=f"TypiClust ({t})", color=c, markersize=6)
                ax.fill_between(b, m - s, m + s, alpha=0.15, color=c)
    if "random_b10" in all_results:
        r = all_results["random_b10"]
        ax.plot(r["cumulative_budget"], r["mean_accuracy"], "x--",
                label="Random", color="tab:gray")
    ax.set_xlabel("Cumulative Budget")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Typicality Measure Comparison on CIFAR-10")
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(str(FIGURES_DIR / "fig2_modification.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for key, r in sorted(all_results.items()):
        m = r["mean_accuracy"]; s = r["se_accuracy"]; b = r["cumulative_budget"]
        print(f"\n{key}:")
        for i in range(len(b)):
            print(f"  Budget {b[i]:3d}: {m[i]:.2f}% ± {s[i]:.2f}%")

    total_min = (time.time() - total_start) / 60
    print(f"\n{'=' * 70}")
    print(f"TOTAL RUNTIME: {total_min:.1f} min")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
