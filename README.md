# ML-CW2: Active Learning on a Budget

Reproduction and extension of the TPC_RP algorithm from:

> Hacohen, G., Dekel, A., & Weinshall, D. (2022). *Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets.* ICML 2022.

## Project Structure

```
ML-CW2/
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
│   ├── wideresnet.py       # WideResNet backbone used by FlexMatch
│   ├── typicclust.py       # TypiClust algorithm (clustering + typicality)
│   ├── active_learning.py  # AL loop: query → label → train → evaluate
│   ├── strategies.py       # Active learning query strategies and baselines
│   ├── classifier.py       # Supervised ResNet-18 classifier training
│   ├── flexmatch.py        # FlexMatch semi-supervised training loop
│   ├── augmentations.py    # SimCLR and classifier augmentation pipelines
│   └── utils.py            # Device detection, seeding, logging, metrics
├── results/                # Saved experiment results (CSVs, JSONs)
├── models/                 # Saved model checkpoints
└── figures/                # Generated plots for the report
```

## Quick Start

```bash
pip install -r requirements.txt

# Results were generated via these scripts (run sequentially):
python run_all.py      # SimCLR training + FW1 TypiClust variants + Random
python run_all_v2.py   # FW1 all baselines + FW2 all strategies
python run_all_v3.py   # FW3 FlexMatch semi-supervised
```

## Hardware

Experiments were run on:
- RunPod RTX 4090 (Frameworks 1 & 2, SimCLR training)
- RunPod RTX 4000 Ada (Framework 3)

Device selection is automatic: CUDA → MPS → CPU.

## Deviations from Paper
| Parameter | Paper | Ours | Rationale |
|-----------|-------|------|-----------|
| SimCLR epochs | 500 | 500 | Matched |
| Classifier epochs | 200 | 200 | Matched |
| AL repetitions | 10 | 5 | Compute constraints; wider CIs noted in report |
| FlexMatch iterations | 400k | 50k | Compute constraints; discussed in report |

## Modification (Task 3)

We evaluate alternative typicality measures:
1. **Baseline**: Inverse mean Euclidean distance to K-NN (paper's method)
2. **Cosine typicality**: Inverse mean cosine distance to K-NN
3. **Local Outlier Factor (LOF)**: Density-based outlier score (inverted)
4. **Kernel Density Estimation (KDE)**: Gaussian KDE score

See `notebooks/03_modification.ipynb` for implementation and evaluation.
