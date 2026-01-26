import math
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import pairwise_kernels, cosine_similarity

from aggregators.aggregatorbase import AggregatorBase
from aggregators.aggregator_utils import prepare_updates, wrapup_aggregated_grads
from aggregators import aggregator_registry
from datapreprocessor.data_utils import subset_by_idx


@aggregator_registry
class CPLRFL(AggregatorBase):
    """
    C-PLR-FL: Class-wise Penultimate Layer Representation for Federated Learning Defense
    
    This defense extracts per-class PLR from client local data, performs cross-client validation
    by grouping clients with shared classes, computes class-wise similarity (MMD or Cosine),
    and aggregates to trust scores for weighted model aggregation.
    
    Key advantages over FLARE:
    - No centralized auxiliary dataset required
    - Designed for Non-IID: leverages class overlap between clients
    - Class-specific detection: identifies which classes are poisoned
    - Cross-client validation: malicious clients have different PLR on shared classes
    """

    def __init__(self, args, **kwargs):
        super().__init__(args)
        """
        samples_per_class (int): number of samples to extract per class from client local data
        similarity_metric (str): 'mmd' or 'cosine' for computing PLR similarity
        tau (float): exponential coefficient for trust score computation
        mmd_kernel (str): kernel type for MMD computation ('rbf', 'linear', etc.)
        mmd_gamma (float): gamma parameter for RBF kernel (None -> sklearn default)
        """
        self.default_defense_params = {
            "samples_per_class": 50,
            "similarity_metric": "mmd",  # "mmd" or "cosine"
            "tau": 1.0,
            "mmd_kernel": "rbf",
            "mmd_gamma": None,
        }
        self.update_and_set_attr()

    def _identify_client_classes(self, client_dataset):
        """
        Identify which classes exist in a client's local dataset.
        
        Args:
            client_dataset: Client's train_dataset (Subset or Dataset)
            
        Returns:
            list: Class labels present in dataset (sorted)
        """
        if hasattr(client_dataset, 'indices'):
            # Subset case (most common in FL)
            full_dataset = client_dataset.dataset
            indices = client_dataset.indices
            if isinstance(indices, torch.Tensor):
                indices = indices.numpy()
            labels = full_dataset.targets[indices]
        else:
            # Direct dataset
            labels = client_dataset.targets
        
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        
        unique_classes = np.unique(labels).tolist()
        return sorted(unique_classes)

    def _sample_class_data(self, client_dataset, class_label, n_samples):
        """
        Sample n_samples from a specific class in client's local dataset.
        
        Args:
            client_dataset: Client's train_dataset
            class_label: Target class to sample
            n_samples: Number of samples to extract
            
        Returns:
            TensorDataset with sampled data
        """
        xs, ys = [], []
        
        if hasattr(client_dataset, 'indices'):
            # Subset case
            full_dataset = client_dataset.dataset
            indices = client_dataset.indices
            if isinstance(indices, torch.Tensor):
                indices = indices.numpy()
            labels = full_dataset.targets[indices]
            
            # Find indices of target class
            class_mask = (labels == class_label)
            if isinstance(labels, torch.Tensor):
                class_mask = class_mask.numpy()
            class_local_indices = np.where(class_mask)[0]
            
            # Sample
            sample_count = min(n_samples, len(class_local_indices))
            sampled_local_indices = np.random.choice(
                class_local_indices, size=sample_count, replace=False
            )
            
            for local_idx in sampled_local_indices:
                global_idx = indices[local_idx]
                x, y = full_dataset[global_idx]
                xs.append(x)
                ys.append(class_label)
        else:
            # Direct dataset (rare)
            for i, (x, y) in enumerate(client_dataset):
                y_int = y.item() if torch.is_tensor(y) else int(y)
                if y_int == class_label and len(xs) < n_samples:
                    xs.append(x)
                    ys.append(class_label)
                if len(xs) >= n_samples:
                    break
        
        if len(xs) == 0:
            return None
        
        x_tensor = torch.stack(xs)
        y_tensor = torch.tensor(ys, dtype=torch.long)
        return torch.utils.data.TensorDataset(x_tensor, y_tensor)

    def _extract_plr(self, model, data_loader, device):
        """
        Extract Penultimate Layer Representation (PLR) using forward hooks.
        Reused from FLARE implementation.
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
            raise ValueError("C-PLR-FL: No Linear/Conv2d layers found; cannot extract PLR.")

        hook_handle = hook_layer.register_forward_hook(hook_fn)
        
        # Forward pass to extract PLR
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(device)
                _ = model(batch_x)  # Forward pass triggers hook
        
        # Remove hook
        hook_handle.remove()
        
        if len(plr_list) == 0:
            return None
        
        plr = np.concatenate(plr_list, axis=0)
        return plr

    def _compute_similarity(self, plr_a, plr_b, metric="mmd"):
        """
        Compute similarity between two PLR distributions.
        Lower value = more similar.
        
        Args:
            plr_a: PLR from client A [m, dim]
            plr_b: PLR from client B [n, dim]
            metric: "mmd" or "cosine"
            
        Returns:
            float: Similarity value (lower = more similar for MMD, higher = more similar for cosine)
        """
        if plr_a is None or plr_b is None:
            return float('inf') if metric == "mmd" else 0.0
        
        if metric == "mmd":
            return self._compute_mmd(plr_a, plr_b)
        elif metric == "cosine":
            # For cosine: average over all pairs, return 1 - similarity (so lower = more similar)
            # Flatten PLRs
            plr_a_flat = plr_a.reshape(plr_a.shape[0], -1)
            plr_b_flat = plr_b.reshape(plr_b.shape[0], -1)
            
            # Average representations
            mean_a = np.mean(plr_a_flat, axis=0, keepdims=True)
            mean_b = np.mean(plr_b_flat, axis=0, keepdims=True)
            
            # Cosine similarity
            cos_sim = cosine_similarity(mean_a, mean_b)[0, 0]
            # Return distance (1 - similarity) so lower = more similar
            return 1.0 - cos_sim
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")

    def _compute_mmd(self, X, Y):
        """
        Compute Maximum Mean Discrepancy (MMD) between two distributions.
        Uses unbiased MMD^2_u estimator. Reused from FLARE.
        """
        m, n = len(X), len(Y)
        if m == 0 or n == 0:
            return 0.0
        
        XY = np.vstack([X, Y])
        
        kernel_kwargs = {}
        if self.mmd_kernel == "rbf" and self.mmd_gamma is not None:
            kernel_kwargs["gamma"] = self.mmd_gamma

        K = pairwise_kernels(XY, metric=self.mmd_kernel, **kernel_kwargs)
        
        # MMD^2_u (unbiased estimator)
        Kx = K[:m, :m]
        Ky = K[m:, m:]
        Kxy = K[:m, m:]
        
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
        return max(0, mmd2u)

    def _extract_classwise_plr(self, clients, client_updated_models, device):
        """
        Extract class-wise PLR for each client from their local data.
        
        Args:
            clients: List of client objects (from kwargs)
            client_updated_models: List of updated models for each client
            device: GPU/CPU device
            
        Returns:
            client_plr_dict: {client_id: {class_c: PLR_{i,c}}}
            client_classes: {client_id: [list of classes]}
        """
        client_plr_dict = {}
        client_classes = {}
        
        for client_id, (client, client_model) in enumerate(zip(clients, client_updated_models)):
            # Identify classes in client's local data
            local_classes = self._identify_client_classes(client.train_dataset)
            client_classes[client_id] = local_classes
            
            client_plr_dict[client_id] = {}
            
            # Extract PLR for each class
            for class_label in local_classes:
                # Sample data from this class
                class_dataset = self._sample_class_data(
                    client.train_dataset, class_label, self.samples_per_class
                )
                
                if class_dataset is None or len(class_dataset) == 0:
                    # Skip if no samples found
                    continue
                
                # Create dataloader
                class_loader = torch.utils.data.DataLoader(
                    class_dataset,
                    batch_size=min(32, len(class_dataset)),
                    shuffle=False,
                    num_workers=0  # Use 0 to avoid issues with multiprocessing
                )
                
                # Extract PLR
                client_model = client_model.to(device)
                plr = self._extract_plr(client_model, class_loader, device)
                
                if plr is not None:
                    client_plr_dict[client_id][class_label] = plr
        
        return client_plr_dict, client_classes

    def _calculate_classwise_similarities(self, client_plr_dict, client_classes):
        """
        Compute pairwise similarities for each class across clients.
        
        Args:
            client_plr_dict: {client_id: {class_c: PLR_{i,c}}}
            client_classes: {client_id: [list of classes]}
            
        Returns:
            class_similarities: {class_c: {(i, j): similarity_value}}
        """
        class_similarities = defaultdict(dict)
        
        # Group clients by class
        class_to_clients = defaultdict(list)
        for client_id, classes in client_classes.items():
            for cls in classes:
                if cls in client_plr_dict[client_id]:  # Only if PLR exists
                    class_to_clients[cls].append(client_id)
        
        # For each class, compute pairwise similarities
        for cls, client_ids in class_to_clients.items():
            if len(client_ids) < 2:
                # Skip if only one client has this class
                continue
            
            for i in range(len(client_ids)):
                for j in range(i + 1, len(client_ids)):
                    client_i = client_ids[i]
                    client_j = client_ids[j]
                    
                    plr_i = client_plr_dict[client_i][cls]
                    plr_j = client_plr_dict[client_j][cls]
                    
                    sim = self._compute_similarity(plr_i, plr_j, self.similarity_metric)
                    
                    # Store symmetric similarity
                    class_similarities[cls][(client_i, client_j)] = sim
                    class_similarities[cls][(client_j, client_i)] = sim
        
        return class_similarities

    def _aggregate_to_trust_scores(self, class_similarities, client_classes, num_clients):
        """
        Aggregate class-wise similarities to per-client trust scores.
        
        Strategy:
        1. For each client, collect all similarities on classes they have
        2. Compute median similarity (robust to outliers)
        3. Convert to trust scores using exponential weighting
        
        Args:
            class_similarities: {class_c: {(i, j): similarity}}
            client_classes: {client_id: [list of classes]}
            num_clients: Total number of clients
            
        Returns:
            np.ndarray: Trust scores [num_clients], normalized to sum=1
        """
        # Collect similarities for each client
        client_similarities = defaultdict(list)
        
        for cls, sim_dict in class_similarities.items():
            for (client_i, client_j), sim_value in sim_dict.items():
                # Add similarity for both clients
                client_similarities[client_i].append(sim_value)
        
        # Compute aggregated similarity score for each client
        client_scores = np.zeros(num_clients)
        
        for client_id in range(num_clients):
            if client_id in client_similarities and len(client_similarities[client_id]) > 0:
                # Use median (robust to outliers)
                # Lower similarity = better (more similar to others)
                # So we invert: higher score = lower similarity
                median_sim = np.median(client_similarities[client_id])
                
                # Invert: lower similarity -> higher score
                # Use negative exponential to convert distance to similarity
                client_scores[client_id] = math.exp(-median_sim / (self.tau + 1e-9))
            else:
                # Client has no shared classes or no similarities computed
                # Give neutral score
                client_scores[client_id] = 1.0
        
        # Normalize to sum=1
        score_sum = np.sum(client_scores)
        if score_sum > 0:
            trust_scores = client_scores / score_sum
        else:
            # Fallback to uniform
            trust_scores = np.ones(num_clients) / num_clients
        
        return trust_scores

    def aggregate(self, updates, **kwargs):
        """
        Main aggregation function implementing C-PLR-FL defense.
        
        Args:
            updates: 2D numpy array [num_clients, model_params]
            **kwargs:
                - last_global_model: Global model (PyTorch)
                - global_weights_vec: Global weights vector
                - global_epoch: Current epoch
                - clients: List of client objects (needed for local data access)
                
        Returns:
            Aggregated update (vector form)
        """
        self.global_model = kwargs['last_global_model']
        self.global_weights_vec = kwargs['global_weights_vec']
        device = self.args.device
        
        # Get clients from server (need to access local data)
        clients = kwargs.get('clients', None)
        if clients is None:
            raise ValueError("C-PLR-FL requires 'clients' in kwargs to access local data. "
                           "Please ensure Server passes clients to aggregator.")
        
        num_clients = len(clients)
        
        # Prepare model updates (convert vector to model form)
        client_updated_models, gradient_updates = prepare_updates(
            self.args.algorithm, updates, self.global_model, vector_form=False
        )
        
        # Step 1: Extract class-wise PLR from client local data
        client_plr_dict, client_classes = self._extract_classwise_plr(
            clients, client_updated_models, device
        )
        
        # Step 2: Compute class-wise similarities
        class_similarities = self._calculate_classwise_similarities(
            client_plr_dict, client_classes
        )
        
        # Step 3: Aggregate to trust scores
        trust_scores = self._aggregate_to_trust_scores(
            class_similarities, client_classes, num_clients
        )
        
        # Step 4: Weighted aggregation
        # trust_scores shape: [num_clients]
        # gradient_updates shape: [num_clients, num_params]
        weighted_updates = gradient_updates * trust_scores[:, np.newaxis]
        aggregated_gradient = np.sum(weighted_updates, axis=0)
        
        # Return 1D vector for FedSGD/FedOpt, model-vec for FedAvg
        return wrapup_aggregated_grads(
            aggregated_gradient,
            self.args.algorithm,
            self.global_model,
            aggregated=True
        )
