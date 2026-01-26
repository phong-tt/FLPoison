# C-PLR-FL: Class-wise Penultimate Layer Representation for Federated Learning

## Overview

C-PLR-FL is a novel defense method against poisoning attacks in Non-IID Federated Learning. Unlike FLARE which uses centralized auxiliary dataset, C-PLR-FL:

- Extracts **per-class PLR** from client **local data**
- Performs **cross-client validation** by grouping clients with shared classes
- Computes **class-wise similarity** (MMD or Cosine)
- No centralized auxiliary dataset required

## Key Advantages

| Feature | FLARE | C-PLR-FL |
|---------|-------|----------|
| Auxiliary data | Server test set required | Not required (uses client local data) |
| PLR scope | Global (single PLR per client) | Per-class (multiple PLRs per client) |
| Non-IID handling | Works but suboptimal | Designed for Non-IID |
| Detection | All-to-all comparison | Class-wise cross-validation |
| Privacy | Needs centralized test data | Uses only local data |

## Configuration

Default parameters in `configs/FedAvg_MNIST_config.yaml`:

```yaml
- defense: CPLRFL
  defense_params:
    samples_per_class: 50        # Samples per class for PLR extraction
    similarity_metric: "mmd"     # "mmd" or "cosine"
    tau: 1.0                     # Exponential coefficient for trust scores
    mmd_kernel: "rbf"            # Kernel for MMD (if using MMD)
    mmd_gamma: null              # Gamma for RBF kernel
```

## Commands

### Test C-PLR-FL on IID
```bash
python main.py --config configs/FedAvg_MNIST_config.yaml -def CPLRFL
```

### Test C-PLR-FL on Non-IID (various alpha)
```bash
# Strong heterogeneity (alpha=0.1)
python main.py --config configs/FedAvg_MNIST_config.yaml -def CPLRFL -dtb non-iid -dirichlet_alpha 0.1

# Moderate heterogeneity (alpha=0.5)
python main.py --config configs/FedAvg_MNIST_config.yaml -def CPLRFL -dtb non-iid -dirichlet_alpha 0.5

# Mild heterogeneity (alpha=1.0)
python main.py --config configs/FedAvg_MNIST_config.yaml -def CPLRFL -dtb non-iid -dirichlet_alpha 1.0
```

### Compare C-PLR-FL vs FLARE
```bash
./scripts/run_cplrfl_test.sh
```

### Override similarity metric (use Cosine instead of MMD)
```bash
python main.py --config configs/FedAvg_MNIST_config.yaml -def CPLRFL \
  -defense_params '{"similarity_metric": "cosine"}'
```

### Batchrun (C-PLR-FL vs multiple defenses)
```bash
python batchrun.py -data MNIST -model lenet -algorithms FedAvg \
  -distributions iid non-iid -attacks DBA -defenses Mean FLARE CPLRFL -gidx 1 -maxp 2
```

## How It Works

### Client-side (during aggregation)
1. Server accesses client's `train_dataset` (local data)
2. Identifies classes present in local data: `C_i = {c1, c2, ..., ck}`
3. For each class `c`:
   - Samples `n` samples from class `c`
   - Extracts PLR via forward pass: `PLR_{i,c}`
4. Result: `{client_i: {class_c: PLR_{i,c}}}`

### Server-side
1. **Group by shared classes**: `G_c = {clients with class c}`
2. **Compute class-wise similarity**:
   - For each pair `(i, j)` in `G_c`: `sim_{i,j}^c = Similarity(PLR_{i,c}, PLR_{j,c})`
3. **Aggregate to trust scores**:
   - For each client: `trust_i = median(similarities on classes in C_i)`
   - Convert to exponential weights
4. **Weighted aggregation**: `θ_{t+1} = Σ_i (trust_i × θ_i)`

### Malicious Detection
- Malicious clients have **different PLR** on poisoned classes
- Cross-client validation: benign clients with same class have similar PLR
- Malicious clients: PLR differs → low similarity → low trust score → reduced weight

## Expected Results

Compared to FLARE on Non-IID:
- **Similar or better ASR** (attack success rate should be low ~0.1)
- **No auxiliary dataset overhead**
- **Faster** (no centralized test data loading)
- **More robust to Non-IID** (designed for heterogeneous settings)

## Log Files

- IID: `logs/FedAvg/MNIST_lenet/iid/MNIST_lenet_iid_DBA_CPLRFL_150_20_0.01_FedAvg.txt`
- Non-IID: `logs/FedAvg/MNIST_lenet/non-iid/MNIST_lenet_non-iid_alpha{X}_DBA_CPLRFL_150_20_0.01_FedAvg.txt`

## Implementation Details

- **File**: `aggregators/cplrfl.py`
- **Base class**: `AggregatorBase`
- **Registry**: Auto-registered via `@aggregator_registry`
- **Dependencies**: 
  - `sklearn.metrics.pairwise` (MMD, Cosine)
  - `datapreprocessor.data_utils` (class identification)
  - PyTorch forward hooks (PLR extraction)

## Future Enhancements

1. Adaptive `tau` based on round number
2. Multi-layer PLR (combine representations from multiple layers)
3. Differential privacy for PLR
4. Class imbalance weighting
5. Temporal consistency tracking across rounds
