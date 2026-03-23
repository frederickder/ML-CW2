# ML-CW2: Active Learning on a Budget — CIFAR-10 Reproduction

Reproduction and extension of the TPC_RP algorithm from:

> Hacohen, G., Dekel, A., & Weinshall, D. (2022). *Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets.* ICML 2022.

## Project Structure

```
typicclust-cifar10/
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_simclr_training.ipynb        # Step 1: Train SimCLR on CIFAR-10
│   ├── 02_typicclust_baseline.ipynb    # Step 2-3: TypiClust implementation + evaluation
│   ├── 03_modification.ipynb           # Task 3: Modified typicality measures
│   └── 04_analysis_and_plots.ipynb     # Report figures and statistical analysis
├── src/
│   ├── __init__.py
│   ├── simclr.py          # SimCLR model, loss, training loop
│   ├── resnet.py           # ResNet-18 adapted for CIFAR-10 (32x32)
│   ├── typicclust.py       # TypiClust algorithm (clustering + typicality)
│   ├── active_learning.py  # AL loop: query → label → train → evaluate
│   ├── classifier.py       # Supervised ResNet-18 classifier training
│   ├── augmentations.py    # SimCLR and classifier augmentation pipelines
│   └── utils.py            # Device detection, seeding, logging, metrics
├── results/                # Saved experiment results (CSVs, JSONs)
├── models/                 # Saved model checkpoints
└── figures/                # Generated plots for the report
```

## Quick Start

```bash
pip install -r requirements.txt

# Option 1: Run notebooks sequentially in Jupyter/Colab
# Option 2: Run from command line
python -m src.simclr --epochs 200 --batch-size 512
python -m src.active_learning --strategy typiclust --budget 10 --rounds 5 --reps 3
```

## Hardware

!!CHANGE THIS!!
Tested on:
- MacBook Air M2 (16GB) via MPS backend
- Google Colab (T4 GPU) via CUDA

Device selection is automatic: CUDA → MPS → CPU.

## Deviations from Paper
!!CHANGE THIS!!
| Parameter | Paper | Ours | Rationale |
|-----------|-------|------|-----------|
| SimCLR epochs | 500 | 200 | Compute constraints; accuracy impact discussed in report |
| AL repetitions | 10 | 3-5 | Compute constraints; wider CIs noted |
| Classifier epochs | 200 | 100 | Compute constraints |

## Modification (Task 3)

We evaluate alternative typicality measures:
1. **Baseline**: Inverse mean Euclidean distance to K-NN (paper's method)
2. **Cosine typicality**: Inverse mean cosine distance to K-NN
3. **Local Outlier Factor (LOF)**: Density-based outlier score (inverted)
4. **Kernel Density Estimation (KDE)**: Gaussian KDE score

See `notebooks/03_modification.ipynb` for implementation and evaluation.
