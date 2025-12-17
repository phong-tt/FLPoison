import numpy as np
import torch
import torch.nn as nn
import math
from sklearn.metrics.pairwise import pairwise_kernels
from aggregators.aggregatorbase import AggregatorBase
from aggregators.aggregator_utils import prepare_updates, wrapup_aggregated_grads
from aggregators import aggregator_registry
from copy import deepcopy


@aggregator_registry
class FLARE(AggregatorBase):
    """
    [FLARE: Defending Federated Learning against Model Poisoning Attacks via Latent Space Representations](https://dl.acm.org/doi/10.1145/3488932.3497760) - AsiaCCS '22
    FLARE detects malicious clients by extracting Penultimate Layer Representations (PLR) from each client's model, computing Maximum Mean Discrepancy (MMD) between PLRs, and calculating trust scores based on nearest-neighbor counts. Client updates are then weighted by trust scores for aggregation.
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        aux_data_num (int): the number of auxiliary samples to use for PLR extraction
        selected_class (int): the class index to filter test data for PLR extraction
        tau (float): exponential coefficient for trust score computation
        mmd_kernel (str): kernel type for MMD computation ('rbf', 'linear', etc.)
        mmd_gamma (float): gamma parameter for RBF kernel (None for auto-selection via median heuristic)
        penultimate_layer_idx (int): index of penultimate layer (None for auto-detection)
        """
        self.default_defense_params = {
            "aux_data_num": 200,
            "selected_class": 1,
            "tau": 1.0,
            "mmd_kernel": "rbf",
            "mmd_gamma": None,
            "penultimate_layer_idx": None,
        }
        self.update_and_set_attr()

        # Get test dataset for PLR extraction
        self.test_dataset = kwargs.get('test_dataset', None)
        if self.test_dataset is None:
            raise ValueError("FLARE requires test_dataset in kwargs. Please ensure test_dataset is passed to aggregator constructor.")

        # Auto-detect penultimate layer if not specified
        if self.penultimate_layer_idx is None:
            # We'll detect it when we have access to the model in aggregate()
            self._penultimate_layer_idx = None
        else:
            self._penultimate_layer_idx = self.penultimate_layer_idx

    def _auto_detect_penultimate_layer(self, model):
        """
        Auto-detect the penultimate layer index by finding the layer before the output layer.
        
        Args:
            model: PyTorch model
            
        Returns:
            int: Index of penultimate layer
        """
        layers = list(model.children())
        
        # Find the last Linear or Conv2d layer (output layer)
        output_layer_idx = None
        for i in range(len(layers) - 1, -1, -1):
            if isinstance(layers[i], (nn.Linear, nn.Conv2d)):
                output_layer_idx = i
                break
        
        if output_layer_idx is None:
            # Fallback: use second-to-last layer
            return max(0, len(layers) - 2)
        
        # Return the layer before output layer
        if output_layer_idx > 0:
            return output_layer_idx - 1
        else:
            # If output layer is first layer (unlikely), use first layer
            return 0

    def _extract_plr(self, model, test_data_loader, device):
        """
        Extract Penultimate Layer Representation (PLR) from a model using forward hooks.
        
        Args:
            model: PyTorch model
            test_data_loader: DataLoader for test data (filtered by selected_class)
            device: torch device
            
        Returns:
            np.ndarray: PLR matrix of shape [num_samples, plr_dim]
        """
        model = model.to(device)
        model.eval()
        
        # Find all Linear and Conv2d layers
        layers = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers.append(module)
        
        if len(layers) < 2:
            # If less than 2 layers, use input to last layer as PLR
            if len(layers) == 0:
                raise ValueError("No Linear or Conv2d layers found in model.")
            
            # Use input to last layer
            plr_list = []
            
            def hook_fn(module, input, output):
                # Extract input (before the layer)
                plr = input[0]
                # Flatten if needed
                if len(plr.shape) > 2:
                    plr = plr.view(plr.size(0), -1)
                plr_list.append(plr.detach().cpu().numpy())
            
            hook_handle = layers[-1].register_forward_hook(hook_fn)
        else:
            # Use output of penultimate layer
            penultimate_layer = layers[-2]
            plr_list = []
            
            def hook_fn(module, input, output):
                # Flatten if needed
                if len(output.shape) > 2:
                    output = output.view(output.size(0), -1)
                plr_list.append(output.detach().cpu().numpy())
            
            hook_handle = penultimate_layer.register_forward_hook(hook_fn)
        
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
        
        # Compute kernel matrix
        kwargs = {}
        if self.mmd_kernel == 'rbf':
            if self.mmd_gamma is None:
                # Auto-select gamma using median heuristic
                if len(XY) > 1:
                    pairwise_distances = np.sqrt(((XY[:, None, :] - XY[None, :, :]) ** 2).sum(axis=2))
                    # Get upper triangle (excluding diagonal)
                    mask = np.triu(np.ones_like(pairwise_distances), k=1).astype(bool)
                    distances = pairwise_distances[mask]
                    if len(distances) > 0:
                        median_dist = np.median(distances[distances > 0])
                        kwargs['gamma'] = 1.0 / (2 * median_dist ** 2) if median_dist > 0 else 1.0
                    else:
                        kwargs['gamma'] = 1.0
                else:
                    kwargs['gamma'] = 1.0
            else:
                kwargs['gamma'] = self.mmd_gamma
        
        K = pairwise_kernels(XY, metric=self.mmd_kernel, **kwargs)
        
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
        
        # Prepare test data filtered by selected_class
        # Collect all test data first
        test_data_list = []
        test_labels_list = []
        for data, label in self.test_dataset:
            test_data_list.append(data)
            test_labels_list.append(label.item() if torch.is_tensor(label) else label)
        
        test_data_tensor = torch.stack(test_data_list)
        test_labels_tensor = torch.tensor(test_labels_list)
        
        # Filter by selected_class
        class_indices = torch.where(test_labels_tensor == self.selected_class)[0]
        
        if len(class_indices) == 0:
            # Fallback: use all classes if selected_class not found
            print(f"Warning: selected_class {self.selected_class} not found in test data. Using all classes.")
            class_indices = torch.arange(min(self.aux_data_num, len(test_data_tensor)))
        else:
            # Take first aux_data_num samples
            class_indices = class_indices[:self.aux_data_num]
        
        filtered_test_data = test_data_tensor[class_indices]
        filtered_test_labels = test_labels_tensor[class_indices]
        
        # Create DataLoader for filtered test data
        filtered_dataset = torch.utils.data.TensorDataset(filtered_test_data, filtered_test_labels)
        test_data_loader = torch.utils.data.DataLoader(
            filtered_dataset,
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
        
        # Wrap up based on algorithm type
        return wrapup_aggregated_grads(
            aggregated_gradient.reshape(1, -1),
            self.args.algorithm,
            self.global_model,
            aggregated=True
        )

