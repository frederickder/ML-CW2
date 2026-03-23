#!/bin/bash
# ──────────────────────────────────────────────────────────
# runpod_run.sh — One-shot script for RunPod GPU instance
#
# Usage:
#   1. Start a RunPod pod (4090, PyTorch template)
#   2. Open terminal
#   3. Run:
#      export GH_TOKEN="ghp_your_token_here"
#      export GH_REPO="your-username/ML-CW2"
#      bash runpod_run.sh
#
#   4. Once it finishes, terminate the pod.
# ──────────────────────────────────────────────────────────

set -e  # Exit on any error

# ── Validate env vars ──
if [ -z "$GH_TOKEN" ] || [ -z "$GH_REPO" ]; then
    echo "ERROR: Set GH_TOKEN and GH_REPO first."
    echo '  export GH_TOKEN="ghp_..."'
    echo '  export GH_REPO="username/ML-CW2"'
    exit 1
fi

echo "═══════════════════════════════════════════"
echo "RunPod one-shot pipeline"
echo "Repo: $GH_REPO"
echo "═══════════════════════════════════════════"

# ── Clone repo ──
cd /workspace
git clone https://${GH_TOKEN}@github.com/${GH_REPO}.git repo
cd repo

# ── Install deps ──
pip install -r requirements.txt

# ── Run the full pipeline ──
python run_all.py

# ── Commit and push results ──
git config user.email "runpod@experiment.run"
git config user.name "RunPod Experiment"

# Add only the artifacts we need (not the 100MB+ files)
git add results/*.json
git add figures/*.png

# Optionally add the model checkpoint (44MB — within GitHub's 100MB limit)
# Uncomment the next line if you want the model in the repo
# git add models/simclr_cifar10.pth

git commit -m "experiment: full pipeline results ($(date -u '+%Y-%m-%d %H:%M UTC'))"
git push

echo ""
echo "═══════════════════════════════════════════"
echo "DONE. Results pushed to $GH_REPO"
echo "You can now terminate this pod."
echo "═══════════════════════════════════════════"
