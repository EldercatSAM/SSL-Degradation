import torch
import torch.nn.functional as F 
import math

def estimator(representations, method_type):
    assert len(representations.shape) == 2
    N, d = representations.shape
    if method_type == 'effective_rank':
        singular_values = torch.linalg.svdvals(representations)
        p = singular_values / singular_values.sum()
        # Avoid log(0) by adding a small epsilon
        epsilon = 1e-12
        p = p + epsilon
        # Compute the Shannon entropy
        H = -torch.sum(p * torch.log(p))
        # Compute the effective rank
        rank = torch.exp(H)
        rank = rank / min(N, d)

    elif method_type == 'centered_singular_sum':
        representations = representations - representations.mean(dim = 0)
        singular_values = torch.linalg.svdvals(representations)
        rank = singular_values.sum() / math.sqrt(max(N - 1, d))

    return rank

def kmeans_plus_plus_init(X, k, seed=None):
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
        prob = dists / torch.sum(dists)
        chosen_idx = torch.multinomial(prob, 1)
        centroids = torch.cat((centroids, X[chosen_idx, :]), dim=0)
    
    return centroids

def kmeans_pp_with_early_stop(
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
    centroids = kmeans_plus_plus_init(X, k, seed=seed).to(device)
    
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
                # If cluster is empty, keep the old centroid (or re-init)
                new_centroids.append(centroids[i])
        new_centroids = torch.stack(new_centroids, dim=0)
        
        # Check shift (L2 distance between old and new centroids)
        shift = (new_centroids - centroids).pow(2).sum().sqrt().item()
        centroids = new_centroids
        
        if shift < tol:
            # Early stopping if centroids don't move much
            break
    
    dists = torch.cdist(X, centroids)
    final_inertia = 0.0
    for i in range(N):
        final_inertia += (X[i] - centroids[labels[i]]).pow(2).sum().item()
    
    return centroids, labels, final_inertia

def kmeans(
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
        best_centroids, best_labels, best_inertia
        among all repeated runs.
    """
    best_inertia = float('inf')
    best_centroids = None
    best_labels = None
    
    for run_id in range(n_runs):
        # Use different seed each run (optional)
        seed = torch.initial_seed() + run_id
        
        centroids, labels, inertia = kmeans_pp_with_early_stop(
            X, 
            k, 
            max_iters=max_iters, 
            tol=tol, 
            seed=seed
        )
        
        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids.clone()
            best_labels = labels.clone()
    
    return best_centroids, best_labels

def compute_DSE(
    representations,
    inter_class_distance_images=8,
    normalize=False,
    local_clusters=3,
    global_clusters=24, 
):
    with torch.no_grad():
        B, N, d = representations.shape
        
        if normalize:
            representations = F.normalize(representations, dim=-1)
        
        avg_l2 = torch.norm(representations - representations.mean(dim=(0, 1), keepdim=True), dim=-1, p=2).mean()
        device = representations.device

        inter_class_distances, intra_class_images = [], []
        ers_global = []
        effective_dimensionalitys = []


        #Compute M_{intra} with the case B=1
        for b in range(B):
            R = representations[b].view(-1, d)
            centroids, labels = kmeans(R, local_clusters)
            ers_local = []
            for i in range(local_clusters):
                cluster_points = R[labels == i]
                if cluster_points.shape[0] > 1:
                    er_local = estimator(cluster_points, method_type='centered_singular_sum')
                    ers_local.append(er_local)
            intra_class_images.append(sum(ers_local)/len(ers_local) if ers_local else torch.tensor(1.0))

        #Compute M_{inter} and M_{intra} with the case B=8
        for b in range(0, B, inter_class_distance_images):
            interval = min(B, b + inter_class_distance_images)
            R_global = representations[b:interval].reshape(-1, d)
            centroids, global_labels = kmeans(R_global, global_clusters)
            
            for i in range(global_clusters):
                cluster_points = R_global[global_labels == i]
                if cluster_points.shape[0] > 1:
                    er_global = estimator(cluster_points, method_type='centered_singular_sum')
                    ers_global.append(er_global)
         
            dist_to_global_centroids = torch.cdist(R_global, centroids)  # Shape: [B*N, global_clusters]
            # Create a tensor of indices for the batch
            batch_indices = torch.arange(R_global.shape[0], device = R_global.device)
            # Clone the distance tensor to avoid modifying the original
            masked_distances = dist_to_global_centroids.clone()
            
            # Set the distance to the own cluster centroid to infinity to exclude it
            masked_distances[batch_indices, global_labels] = float('inf')
            
            # Compute the minimum distance to any other cluster centroid
            min_distances = masked_distances.min(dim=1).values  # Shape: [B*N]
            
            # Compute the average of the minimum distances
            inter_class_distance = min_distances.mean()
            inter_class_distances.append(inter_class_distance)

        if ers_global:
            intra_class_batch = sum(ers_global) / len(ers_global)
        else:
            intra_class_batch = 1.0 
        
        inter_class_distance = torch.tensor(inter_class_distances).float().mean() / avg_l2 if inter_class_distances else torch.tensor(0.0)
        intra_class_image = torch.tensor(intra_class_images).float().mean() / avg_l2 if intra_class_images else torch.tensor(0.0)
        
        intra_class_batch = torch.tensor(intra_class_batch).float().mean() / avg_l2
        #Compute M_{dim}, shuffle the representations to ensure i.i.d. 
        perm = torch.stack([torch.randperm(N) for _ in range(B)]) 
        shuffled = representations[torch.arange(B).unsqueeze(1), perm]
        shuffled = shuffled.permute(1,0,2)

        for n in range(N):
            R = shuffled[n].reshape(-1, d)
            effective_dimensionalitys.append(estimator(R, method_type='effective_rank'))
        effective_dimensionality = torch.tensor(effective_dimensionalitys).float().mean()

        DSE = inter_class_distance + effective_dimensionality - (intra_class_batch +  intra_class_image) / 2
        
        return (
            inter_class_distance.item(), 
            intra_class_image.item(), 
            intra_class_batch.item(), 
            effective_dimensionality.item(),
            DSE.item()
        )