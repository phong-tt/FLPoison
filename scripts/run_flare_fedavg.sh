#!/usr/bin/env bash
# Test FLARE + FedAvg on MNIST: IID and Non-IID (various dirichlet_alpha)
# Config: Quick wins — local_epochs 2, num_clients 20, epochs 150, FLARE aux_data_num 100

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/FedAvg_MNIST_config.yaml"

echo "=== FLARE + FedAvg (MNIST, DBA) ==="

# IID (config mặc định đã là iid, không cần override)
echo "[1/4] IID..."
python main.py --config "$CONFIG"

# Non-IID alpha=1.0 (mild)
echo "[2/4] Non-IID alpha=1.0 (mild)..."
python main.py --config "$CONFIG" -dtb non-iid -dirichlet_alpha 1.0

# Non-IID alpha=0.5 (moderate)
echo "[3/4] Non-IID alpha=0.5 (moderate)..."
python main.py --config "$CONFIG" -dtb non-iid -dirichlet_alpha 0.5

# Non-IID alpha=0.1 (strong)
echo "[4/4] Non-IID alpha=0.1 (strong)..."
python main.py --config "$CONFIG" -dtb non-iid -dirichlet_alpha 0.1

echo "Done. Logs:"
echo "  - IID: logs/FedAvg/MNIST_lenet/iid/MNIST_lenet_iid_DBA_FLARE_150_20_0.01_FedAvg.txt"
echo "  - Non-IID alpha=1.0: logs/FedAvg/MNIST_lenet/non-iid/MNIST_lenet_non-iid_alpha1.0_DBA_FLARE_150_20_0.01_FedAvg.txt"
echo "  - Non-IID alpha=0.5: logs/FedAvg/MNIST_lenet/non-iid/MNIST_lenet_non-iid_alpha0.5_DBA_FLARE_150_20_0.01_FedAvg.txt"
echo "  - Non-IID alpha=0.1: logs/FedAvg/MNIST_lenet/non-iid/MNIST_lenet_non-iid_alpha0.1_DBA_FLARE_150_20_0.01_FedAvg.txt"
