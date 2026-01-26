import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import pairwise_kernels

from aggregators.aggregatorbase import AggregatorBase
from aggregators.aggregator_utils import prepare_updates, wrapup_aggregated_grads
from aggregators import aggregator_registry


@aggregator_registry
class FLARE(AggregatorBase):
    """
    [FLARE: Defending Federated Learning against Model Poisoning Attacks via Latent Space Representations](https://dl.acm.org/doi/10.1145/3488932.3497760) - AsiaCCS '22
    FLARE detects malicious clients by extracting Penultimate Layer Representations (PLR) from each client's model, computing Maximum Mean Discrepancy (MMD) between PLRs, and calculating trust scores based on nearest-neighbor counts. Client updates are then weighted by trust scores for aggregation.

    This implementation is adapted to this repo's PyTorch framework while matching the
    original FLARE logic (penultimate representation + MMD + trust scores).
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        aux_data_num (int): the number of auxiliary samples to use for PLR extraction
        selected_class (int): the class index to filter test data for PLR extraction
        tau (float): exponential coefficient for trust score computation
        mmd_kernel (str): kernel type for MMD computation ('rbf', 'linear', etc.)
        mmd_gamma (float): gamma parameter for RBF kernel (None -> follow sklearn default, matching original)
        """
        self.default_defense_params = {
            "aux_data_num": 200,
            "selected_class": 1,
            "tau": 1.0,
            "mmd_kernel": "rbf",
            "mmd_gamma": None,
        }
        self.update_and_set_attr()

        # Get test dataset for PLR extraction
        self.test_dataset = kwargs.get('test_dataset', None)
        if self.test_dataset is None:
            raise ValueError("FLARE requires test_dataset in kwargs. Please ensure test_dataset is passed to aggregator constructor.")

    def _select_aux_dataset(self):
        """
        Match the original FLARE logic:
        - pick first `aux_data_num` samples from `selected_class`
        - fallback to the first `aux_data_num` samples overall if class not found
        """
        xs, ys = [], []
        fallback_xs, fallback_ys = [], []

        for x, y in self.test_dataset:
            y_int = y.item() if torch.is_tensor(y) else int(y)

            if len(fallback_xs) < self.aux_data_num:
                fallback_xs.append(x)
                fallback_ys.append(y_int)

            if y_int == self.selected_class and len(xs) < self.aux_data_num:
                xs.append(x)
                ys.append(y_int)

            if len(xs) >= self.aux_data_num:
                break

        if len(xs) == 0:
            xs, ys = fallback_xs, fallback_ys

        if len(xs) == 0:
            raise ValueError("FLARE: empty test_dataset, cannot select aux data.")

        x_tensor = torch.stack(xs)
        y_tensor = torch.tensor(ys, dtype=torch.long)
        return torch.utils.data.TensorDataset(x_tensor, y_tensor)

    def _extract_plr(self, model, test_data_loader, device):
        """
        Extract Penultimate Layer Representation (PLR) using forward hooks.

        To best match the original TF/Keras implementation (a fixed penultimate layer output),
        we use the **input to the last nn.Linear** as PLR when available. This is the typical
        “penultimate representation” for classifier architectures.
        """
        model = model.to(device)
        model.eval()
        
        plr_list = []

        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]

        if len(linear_layers) > 0:
            hook_layer = linear_layers[-1]

            def hook_fn(_module, inputs, _output):
                plr = inputs[0]
                if len(plr.shape) > 2:
                    plr = plr.view(plr.size(0), -1)
                plr_list.append(plr.detach().cpu().numpy())

        elif len(conv_layers) > 0:
            hook_layer = conv_layers[-1]

            def hook_fn(_module, _inputs, output):
                if len(output.shape) > 2:
                    output = output.view(output.size(0), -1)
                plr_list.append(output.detach().cpu().numpy())

        else:
            raise ValueError("FLARE: No Linear/Conv2d layers found; cannot extract PLR.")

        hook_handle = hook_layer.register_forward_hook(hook_fn)
        
        # Forward pass to extract PLR
        with torch.no_grad():
            for batch_x, _ in test_data_loader:
                batch_x = batch_x.to(device)
                _ = model(batch_x)  # Forward pass triggers hook
        
        # Remove hook
        hook_handle.remove()
        
        if len(plr_list) == 0:
            raise ValueError("No PLR extracted. Check test_data_loader and selected_class.")
        
        plr = np.concatenate(plr_list, axis=0)
        return plr

    def _compute_mmd(self, X, Y):
        """
        Compute Maximum Mean Discrepancy (MMD) between two distributions.
        Uses unbiased MMD^2_u estimator.
        
        Args:
            X: First distribution samples [m, dim]
            Y: Second distribution samples [n, dim]
            
        Returns:
            float: MMD^2_u value
        """
        m, n = len(X), len(Y)
        if m == 0 or n == 0:
            return 0.0
        
        XY = np.vstack([X, Y])
        
        # Match original FLARE MMD:
        # use pairwise_kernels, and only pass gamma if explicitly provided.
        kernel_kwargs = {}
        if self.mmd_kernel == "rbf" and self.mmd_gamma is not None:
            kernel_kwargs["gamma"] = self.mmd_gamma

        K = pairwise_kernels(XY, metric=self.mmd_kernel, **kernel_kwargs)
        
        # MMD^2_u (unbiased estimator)
        Kx = K[:m, :m]
        Ky = K[m:, m:]
        Kxy = K[:m, m:]
        
        # Handle edge case when m=1 or n=1
        if m == 1:
            term1 = 0.0
        else:
            term1 = 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum())
        
        if n == 1:
            term2 = 0.0
        else:
            term2 = 1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum())
        
        term3 = 2.0 / (m * n) * Kxy.sum()
        
        mmd2u = term1 + term2 - term3
        return max(0, mmd2u)  # Ensure non-negative

    def _calculate_nearest_neighbors(self, plr_list):
        """
        Calculate nearest neighbor counts for each client.
        For each client, count how many times it is selected as a nearest neighbor by other clients.
        
        Args:
            plr_list: List of PLR arrays, one per client
            
        Returns:
            list: Count of times each client is selected as nearest neighbor
        """
        n = len(plr_list)
        if n < 2:
            # If only one client, return uniform count
            return [1] * n
        
        mmd_matrix = np.zeros([n, n])
        
        # Compute pairwise MMD
        for i in range(n):
            for j in range(i + 1, n):
                mmd_val = self._compute_mmd(plr_list[i], plr_list[j])
                mmd_matrix[i, j] = mmd_val
                mmd_matrix[j, i] = mmd_val
        
        # Find top 50% nearest neighbors for each client
        sorted_indices = np.argsort(mmd_matrix, axis=1)
        k = int(0.5 * n)  # Top 50% (match original implementation)
        top_k_neighbors = sorted_indices[:, 1:k]  # Exclude self (index 0), match original: ind[:, 1:k]
        
        # Count how many times each client is selected
        count_list = []
        for i in range(n):
            count = np.sum(top_k_neighbors == i)
            count_list.append(int(count))
        
        return count_list

    def _count_to_trustscore(self, count_list):
        """
        Convert nearest neighbor counts to trust scores using exponential weighting.
        
        Args:
            count_list: List of nearest neighbor counts
            
        Returns:
            np.ndarray: Trust scores (normalized, sum to 1)
        """
        if len(count_list) == 0:
            return np.array([])
        
        count_avg = sum(count_list) / len(count_list) if len(count_list) > 0 else 1.0
        
        if count_avg == 0:
            # Fallback to uniform if all counts are 0
            return np.ones(len(count_list)) / len(count_list)
        
        # Exponential weighting
        exp_scores = [math.exp(c / (self.tau * count_avg)) for c in count_list]
        exp_sum = sum(exp_scores)
        
        if exp_sum == 0:
            # Fallback to uniform
            return np.ones(len(count_list)) / len(count_list)
        
        trust_scores = np.array([s / exp_sum for s in exp_scores])
        return trust_scores

    def aggregate(self, updates, **kwargs):
        """
        Main aggregation function implementing FLARE defense.
        
        Args:
            updates: 2D numpy array [num_clients, model_params]
            **kwargs:
                - last_global_model: Global model (PyTorch)
                - global_weights_vec: Global weights vector
                - global_epoch: Current epoch
                
        Returns:
            Aggregated update (vector form)
        """
        self.global_model = kwargs['last_global_model']
        self.global_weights_vec = kwargs['global_weights_vec']
        device = self.args.device
        
        # Prepare model updates (convert vector to model form)
        client_updated_models, gradient_updates = prepare_updates(
            self.args.algorithm, updates, self.global_model, vector_form=False
        )
        
        # Prepare auxiliary dataset (filtered by selected_class, like original)
        aux_dataset = self._select_aux_dataset()
        test_data_loader = torch.utils.data.DataLoader(
            aux_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers
        )
        
        # Extract PLR for each client
        plr_list = []
        for client_model in client_updated_models:
            client_model = client_model.to(device)
            plr = self._extract_plr(client_model, test_data_loader, device)
            plr_list.append(plr)
        
        # Calculate nearest neighbor counts
        count_list = self._calculate_nearest_neighbors(plr_list)
        
        # Convert to trust scores
        trust_scores = self._count_to_trustscore(count_list)
        
        # Weighted aggregation
        # trust_scores shape: [num_clients]
        # gradient_updates shape: [num_clients, num_params]
        weighted_updates = gradient_updates * trust_scores[:, np.newaxis]
        aggregated_gradient = np.sum(weighted_updates, axis=0)
        
        # IMPORTANT: return a 1D vector for FedSGD/FedOpt, and model-vec for FedAvg
        return wrapup_aggregated_grads(
            aggregated_gradient,
            self.args.algorithm,
            self.global_model,
            aggregated=True
        )

