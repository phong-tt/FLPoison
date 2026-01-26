# Scripts

## Config: `configs/FedAvg_MNIST_config.yaml`

- **Algorithm:** FedAvg
- **Dataset:** MNIST, LeNet
- **GPU:** `gpu_idx: [1]`
- **Quick wins:** `local_epochs: 2`, `num_clients: 20`, `epochs: 150`, FLARE `aux_data_num: 100`
- **High memory:** `batch_size: 4096`, `num_workers: 16`
- **Attack:** DBA (default)
- **Defense:** FLARE (default)
- **Note:** Log files for non-IID include `dirichlet_alpha` in filename (e.g., `..._non-iid_alpha0.1_...`)

**Chỉ override những gì khác với config:** `-dtb non-iid -dirichlet_alpha <float>`, `-gidx <idx>`, etc.

---

## Commands: FLARE + FedAvg (IID & Non-IID)

**Base config:** `configs/FedAvg_MNIST_config.yaml`

### IID (config mặc định đã là iid)
```bash
python main.py --config configs/FedAvg_MNIST_config.yaml
```

### Non-IID (alpha = 1.0, mild)
```bash
python main.py --config configs/FedAvg_MNIST_config.yaml -dtb non-iid -dirichlet_alpha 1.0
```

### Non-IID (alpha = 0.5, moderate)
```bash
python main.py --config configs/FedAvg_MNIST_config.yaml -dtb non-iid -dirichlet_alpha 0.5
```

### Non-IID (alpha = 0.1, strong)
```bash
python main.py --config configs/FedAvg_MNIST_config.yaml -dtb non-iid -dirichlet_alpha 0.1
```

### Override GPU (nếu khác config)
```bash
python main.py --config configs/FedAvg_MNIST_config.yaml -gidx 0
```

### Run all (IID + 3 Non-IID) via script
```bash
./scripts/run_flare_fedavg.sh
```

### Override batch size (nếu muốn tăng)
```bash
python main.py --config configs/FedAvg_MNIST_config.yaml -bs 8192
```

**Note:** Quick wins: `local_epochs` 2, `num_clients` 20, `epochs` 150, FLARE `aux_data_num` 100. Có thể tăng thêm `batch_size` nếu GPU còn trống.

### Batchrun (FedAvg + FLARE, IID & Non-IID)
```bash
python batchrun.py -data MNIST -model lenet -algorithms FedAvg \
  -distributions iid non-iid -attacks DBA -defenses FLARE -gidx 1 -maxp 2
```

**Log files:**
- IID: `logs/FedAvg/MNIST_lenet/iid/MNIST_lenet_iid_DBA_FLARE_150_20_0.01_FedAvg.txt`
- Non-IID alpha=1.0: `logs/FedAvg/MNIST_lenet/non-iid/MNIST_lenet_non-iid_alpha1.0_DBA_FLARE_150_20_0.01_FedAvg.txt`
- Non-IID alpha=0.5: `logs/FedAvg/MNIST_lenet/non-iid/MNIST_lenet_non-iid_alpha0.5_DBA_FLARE_150_20_0.01_FedAvg.txt`
- Non-IID alpha=0.1: `logs/FedAvg/MNIST_lenet/non-iid/MNIST_lenet_non-iid_alpha0.1_DBA_FLARE_150_20_0.01_FedAvg.txt`

**Quick wins (no code changes):**
- ✅ `local_epochs`: 5 → 2 (giảm ~40–60% thời gian / round)
- ✅ `num_clients`: 50 → 20 (giảm overhead aggregation & training passes)
- ✅ `epochs`: 100 → 150 (bù do ít local updates / round)
- ✅ FLARE `aux_data_num`: 200 → 100 (giảm ~50% thời gian PLR extraction)
