import torch
import torch.nn.functional as F
import math

class DSERegularizer:
    def __init__(
        self,
        normalize=True,
        local_clusters=3,
        global_clusters=24
    ):
        self.normalize = normalize
        self.local_clusters = local_clusters
        self.global_clusters = global_clusters
        self.inter_class_images = global_clusters // local_clusters

    def _kmeans_plus_plus_init(self, X, k, seed=None):
        """
        X: (N, d) data tensor
        k: number of clusters
        seed: optional integer for reproducibility

        Returns:
            A (k, d) tensor of initial centroids.
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        N, d = X.shape
        # Choose first centroid randomly
        idx = torch.randint(0, N, (1,))
        centroids = X[idx, :]  # shape (1, d)
        
        # Choose each subsequent centroid
        for _ in range(k - 1):
            # Compute distances from each point to the nearest centroid
            dists = torch.cdist(X, centroids).min(dim=1).values
            # Weighted probability ~ distance^2
            prob = dists / (torch.sum(dists) + 1e-12) # Added epsilon for numerical stability
            if torch.sum(prob) == 0: # Handle case where all probabilities are zero
                prob = torch.ones_like(prob) / len(prob)
            chosen_idx = torch.multinomial(prob, 1)
            centroids = torch.cat((centroids, X[chosen_idx, :]), dim=0)
        
        return centroids

    def _kmeans_pp_with_early_stop(
        self, 
        X, 
        k, 
        max_iters=50, 
        tol=1e-4, 
        seed=None
    ):
        """
        X: (N, d) data
        k: number of clusters
        max_iters: maximum number of iterations
        tol: early stopping threshold on centroid shift
        seed: integer or None, for reproducible initialization

        Returns:
            (centroids, labels, final_inertia)
        """
        device = X.device
        N, d = X.shape
        
        # 1. K-Means++ initialization
        centroids = self._kmeans_plus_plus_init(X, k, seed=seed).to(device)
        
        # 2. Iterative refinement
        for _ in range(max_iters):
            # Compute distances (N, k)
            dists = torch.cdist(X, centroids)
            # Assign each point to the nearest centroid
            labels = torch.argmin(dists, dim=1)
            
            # Recompute centroids
            new_centroids = []
            for i in range(k):
                cluster_points = X[labels == i]
                if cluster_points.shape[0] > 0:
                    new_centroids.append(cluster_points.mean(dim=0))
                else:
                    # If cluster is empty, re-initialize this centroid randomly from X
                    # This handles the case of empty clusters better than keeping the old centroid.
                    idx = torch.randint(0, N, (1,), device=device)
                    new_centroids.append(X[idx, :].squeeze())
            new_centroids = torch.stack(new_centroids, dim=0)
            
            # Check shift (L2 distance between old and new centroids)
            shift = (new_centroids - centroids).pow(2).sum().sqrt().item()
            centroids = new_centroids
            
            if shift < tol:
                # Early stopping if centroids don't move much
                break
        
        # Final calculation of distances and inertia
        dists = torch.cdist(X, centroids)
        labels = torch.argmin(dists, dim=1) # Re-assign labels with final centroids
        final_inertia = 0.0
        for i in range(N): # Use N from X.shape
            final_inertia += (X[i] - centroids[labels[i]]).pow(2).sum().item()
        
        return centroids, labels, final_inertia

    def _kmeans(
        self, 
        X, 
        k, 
        n_runs=1, 
        max_iters=20, 
        tol=1e-4
    ):
        """
        X: (N, d) data tensor
        k: number of clusters
        n_runs: number of times to run k-means
        max_iters: max iterations per run
        tol: early-stopping threshold

        Returns:
            best_centroids, best_labels (inertia is not returned by this wrapper)
        """
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None
        
        for run_id in range(n_runs):
            # Use different seed each run
            current_seed = torch.initial_seed() + run_id
            
            centroids, labels, inertia = self._kmeans_pp_with_early_stop(
                X, 
                k, 
                max_iters=max_iters, 
                tol=tol, 
                seed=current_seed 
            )
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.clone()
                best_labels = labels.clone()
        
        return best_centroids, best_labels # Removed best_inertia from return as it's not used by calling function

    def estimator(self, representations, method_type):
        # Core calculation function that preserves gradient propagation
        assert len(representations.shape) == 2, "Input should be a 2D tensor"
        N, d = representations.shape
        if N == 0 or d == 0: # Handle empty input
            return torch.tensor(0.0, device=representations.device)

        if method_type == 'effective_rank':
            singular_values = torch.linalg.svdvals(representations)
            # Normalize singular values to get probabilities
            p = singular_values / (singular_values.sum() + 1e-12) # Epsilon for stability
            p = p + 1e-12  # Avoid log(0)
            # Shannon entropy
            H = -torch.sum(p * torch.log(p))
            # Effective rank normalized by min(N,d)
            rank = torch.exp(H) / min(N, d) if min(N,d) > 0 else torch.tensor(0.0, device=representations.device)


        elif method_type == 'centered_singular_sum':
            representations_centered = representations - representations.mean(dim=0, keepdim=True)
            singular_values = torch.linalg.svdvals(representations_centered)
            # Sum of singular values normalized by sqrt(max(N-1, d))
            denominator = math.sqrt(max(N - 1, d)) if max(N-1, d) > 0 else 1.0 # Avoid sqrt(0) or division by zero
            rank = singular_values.sum() / denominator


        return rank

    def compute_DSE_loss(self, representations):
        # Input representations shape: (B, N, d) - B: batch, N: patches/tokens per image, d: feature_dim
        B, N_patches, d_feat = representations.shape # Using more descriptive names
        
        if self.normalize:
            representations = F.normalize(representations, dim=-1) # Normalize along the feature dimension
        
        # Calculate average L2 norm of representations (after centering)
        # This avg_l2 is used for normalizing the DSE components
        avg_l2 = torch.norm(representations - representations.mean(dim=(0, 1), keepdim=True), dim=-1, p=2).mean()
        device = representations.device

        # Helper function: Run K-means in a no-gradient context for clustering
        def run_kmeans_no_grad(data_for_kmeans, num_clusters_k):
            with torch.no_grad():
                # Ensure data_for_kmeans is 2D (num_samples, num_features)
                if data_for_kmeans.ndim > 2 :
                     data_for_kmeans = data_for_kmeans.reshape(-1, d_feat)
                if data_for_kmeans.shape[0] < num_clusters_k : # Not enough samples for k clusters
                    if data_for_kmeans.shape[0] == 0:
                         return None, None # No data to cluster
                    actual_k = min(num_clusters_k, data_for_kmeans.shape[0])
                    if actual_k == 0: # Still no data
                         return None, None
                    centroids, labels = self._kmeans(data_for_kmeans, actual_k)

                else:
                    centroids, labels = self._kmeans(data_for_kmeans, num_clusters_k)
            return centroids.detach() if centroids is not None else None, labels.detach() if labels is not None else None
        
        # 1. Calculate M_intra_image: Average effective rank within local clusters of each image
        intra_image_terms = []
        for b_idx in range(B):
            # Representations for the current image: (N_patches, d_feat)
            current_image_repr = representations[b_idx]
            
            # Cluster patches within the current image
            _, local_cluster_labels = run_kmeans_no_grad(current_image_repr, self.local_clusters)
            
            if local_cluster_labels is None:
                continue

            cluster_ranks_for_image = []
            for cluster_idx in range(self.local_clusters):
                # Get data for the current local cluster
                cluster_mask = (local_cluster_labels == cluster_idx)
                if cluster_mask.sum() > 1:
                    cluster_data = current_image_repr[cluster_mask]
                    # Estimate rank for this local cluster's representations
                    effective_rank = self.estimator(cluster_data, 'centered_singular_sum')
                    cluster_ranks_for_image.append(effective_rank)
            
            if cluster_ranks_for_image: # If any valid clusters were found and ranks computed
                intra_image_terms.append(torch.mean(torch.stack(cluster_ranks_for_image)))
        
        # Average M_intra_image across the batch, normalized by avg_l2
        M_intra_image = torch.mean(torch.stack(intra_image_terms)) / (avg_l2 + 1e-12) if intra_image_terms else torch.tensor(0.0, device=device)

        # 2. Calculate M_inter (inter-cluster dissimilarity) and M_intra_batch (intra-cluster similarity for global clusters)
        inter_cluster_terms = []
        intra_batch_cluster_terms = []
        
        # Process in groups of 'inter_class_images'
        for group_start_idx in range(0, B, self.inter_class_images):
            # Representations for the current global group: (group_size * N_patches, d_feat)
            global_group_repr = representations[group_start_idx : group_start_idx + self.inter_class_images].reshape(-1, d_feat)
            
            if global_group_repr.shape[0] == 0:
                continue # Skip if group is empty

            # Cluster all representations in this global group
            global_centroids, global_cluster_labels = run_kmeans_no_grad(global_group_repr, self.global_clusters)

            if global_centroids is None or global_cluster_labels is None:
                continue 
            dist_matrix_to_centroids = torch.cdist(global_group_repr, global_centroids)
            
            # Create a mask to ignore distances to a point's own assigned cluster centroid
            # True where point_idx is assigned to centroid_idx
            assigned_centroid_mask = torch.zeros_like(dist_matrix_to_centroids, dtype=torch.bool, device=device)
            assigned_centroid_mask[torch.arange(global_group_repr.shape[0]), global_cluster_labels] = True
            
            # Add a large value to distances to own centroid to ensure they are not chosen as min
            # Then find the minimum distance to any *other* centroid for each point
            min_dist_to_other_centroids = torch.min(dist_matrix_to_centroids + assigned_centroid_mask.float() * 1e9, dim=1).values
            inter_cluster_terms.append(torch.mean(min_dist_to_other_centroids))
            
            # Calculate intra-batch cluster rank (M_intra_batch component for this group)
            for cluster_idx in range(self.global_clusters):
                cluster_mask = (global_cluster_labels == cluster_idx)
                if cluster_mask.sum() > 1: # Need at least 2 points
                    cluster_data = global_group_repr[cluster_mask]
                    effective_rank = self.estimator(cluster_data, 'centered_singular_sum')
                    intra_batch_cluster_terms.append(effective_rank)
        
        # Average M_inter across groups, normalized
        M_inter = torch.mean(torch.stack(inter_cluster_terms)) / (avg_l2 + 1e-12) if inter_cluster_terms else torch.tensor(0.0, device=device) 
        M_intra_batch = torch.mean(torch.stack(intra_batch_cluster_terms)) / (avg_l2 + 1e-12) if intra_batch_cluster_terms else torch.tensor(1.0, device=device) 

        # 3. Calculate M_dim: Effective rank of globally shuffled representations
        # Flatten all representations from the batch: (B * N_patches, d_feat)
        all_representations_flat = representations.reshape(-1, d_feat)
        if all_representations_flat.shape[0] > 0 :
            # Shuffle these flat representations
            shuffled_flat_representations = all_representations_flat[torch.randperm(all_representations_flat.size(0), device=device)]
            M_dim = self.estimator(shuffled_flat_representations, 'effective_rank')
        else:
            M_dim = torch.tensor(0.0, device=device)

        DSE = M_inter + M_dim - 0.5 * (M_intra_batch + M_intra_image)
        return DSE