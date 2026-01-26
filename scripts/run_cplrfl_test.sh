#!/usr/bin/env bash
# Test C-PLR-FL vs FLARE on MNIST: IID and Non-IID
# Compare: ASR, Test Acc, runtime

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/FedAvg_MNIST_config.yaml"

echo "=== C-PLR-FL vs FLARE Comparison (MNIST, DBA) ==="
echo ""

# Test 1: C-PLR-FL on IID
echo "[1/4] C-PLR-FL on IID..."
python main.py --config "$CONFIG" -def CPLRFL -dtb iid

# Test 2: C-PLR-FL on Non-IID (alpha=0.1, strong heterogeneity)
echo "[2/4] C-PLR-FL on Non-IID alpha=0.1..."
python main.py --config "$CONFIG" -def CPLRFL -dtb non-iid -dirichlet_alpha 0.1

# Test 3: FLARE on Non-IID for comparison
echo "[3/4] FLARE on Non-IID alpha=0.1 (baseline)..."
python main.py --config "$CONFIG" -def FLARE -dtb non-iid -dirichlet_alpha 0.1

# Test 4: Mean on Non-IID (no defense baseline)
echo "[4/4] Mean on Non-IID alpha=0.1 (no defense baseline)..."
python main.py --config "$CONFIG" -def Mean -dtb non-iid -dirichlet_alpha 0.1

echo ""
echo "Done. Compare results:"
echo "  - C-PLR-FL IID:        logs/FedAvg/MNIST_lenet/iid/MNIST_lenet_iid_DBA_CPLRFL_150_20_0.01_FedAvg.txt"
echo "  - C-PLR-FL Non-IID:    logs/FedAvg/MNIST_lenet/non-iid/MNIST_lenet_non-iid_alpha0.1_DBA_CPLRFL_150_20_0.01_FedAvg.txt"
echo "  - FLARE Non-IID:       logs/FedAvg/MNIST_lenet/non-iid/MNIST_lenet_non-iid_alpha0.1_DBA_FLARE_150_20_0.01_FedAvg.txt"
echo "  - Mean Non-IID:        logs/FedAvg/MNIST_lenet/non-iid/MNIST_lenet_non-iid_alpha0.1_DBA_Mean_150_20_0.01_FedAvg.txt"
