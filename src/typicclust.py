"""
TypiClust: Typical Clustering for Active Learning in the Low-Budget Regime.

Implements Algorithm 1 from the paper (TPC_RP variant):
  1. Compute embeddings (done externally via SimCLR)
  2. Cluster embeddings via K-means for diversity
  3. Select the most typical (highest density) point from each uncovered cluster

Also includes alternative typicality measures for Task 3 (modification).
"""

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.preprocessing import normalize
from scipy.stats import gaussian_kde


# Typicality Measures

def typicality_euclidean(features: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Paper's original typicality measure (Eq. 4):
    Typicality(x) = (1/K * sum ||x - x_i||_2 for x_i in K-NN(x))^{-1}
    
    Higher value = more typical (denser region).
    """
    n = features.shape[0]
    k_actual = min(k, n - 1)
    if k_actual < 1:
        return np.ones(n)

    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="euclidean")
    nn.fit(features)
    distances, _ = nn.kneighbors(features)

    # Exclude self (first column is distance to self = 0)
    distances = distances[:, 1:]

    mean_dist = distances.mean(axis=1)
    # Avoid division by zero
    mean_dist = np.maximum(mean_dist, 1e-10)
    typicality = 1.0 / mean_dist
    return typicality


def typicality_cosine(features: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Modification: Cosine-distance based typicality.
    Since SimCLR features are L2-normalised, cosine distance = 1 - dot product.
    This may better capture semantic similarity in the normalised embedding space.
    """
    n = features.shape[0]
    k_actual = min(k, n - 1)
    if k_actual < 1:
        return np.ones(n)

    # Ensure L2 normalised
    features_norm = normalize(features, norm="l2")

    nn = NearestNeighbors(n_neighbors=k_actual + 1, metric="cosine")
    nn.fit(features_norm)
    distances, _ = nn.kneighbors(features_norm)

    distances = distances[:, 1:]  # exclude self
    mean_dist = distances.mean(axis=1)
    mean_dist = np.maximum(mean_dist, 1e-10)
    typicality = 1.0 / mean_dist
    return typicality


def typicality_lof(features: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Modification: Local Outlier Factor (LOF) based typicality.
    LOF < 1 means the point is in a denser region than its neighbors (typical).
    LOF > 1 means the point is in a sparser region (atypical).
    We return the negative LOF score so that higher = more typical.
    """
    n = features.shape[0]
    k_actual = min(k, n - 1)
    if k_actual < 1:
        return np.ones(n)

    lof = LocalOutlierFactor(n_neighbors=k_actual, metric="euclidean", novelty=False)
    lof.fit(features)

    # negative_outlier_factor_ is already negated (more negative = more outlier)
    # So we just return it as-is: higher value = more typical
    return lof.negative_outlier_factor_


def typicality_kde(features: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Modification: Kernel Density Estimation based typicality.
    Estimates the density at each point using Gaussian KDE.
    
    Note: Full KDE on high-dimensional data is expensive. We use a
    subsampled approach or PCA reduction if needed.
    """
    n, d = features.shape

    # KDE is impractical in high dimensions; use PCA to reduce first
    if d > 50:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)

    # For large N, subsample the KDE fitting set
    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False)
        kde = gaussian_kde(features[idx].T, bw_method="scott")
    else:
        kde = gaussian_kde(features.T, bw_method="scott")

    # Evaluate density at all points (in batches to manage memory)
    batch_size = 1000
    densities = np.zeros(n)
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        densities[i:end] = kde(features[i:end].T)

    return densities


# Registry of typicality functions
TYPICALITY_FUNCTIONS = {
    "euclidean": typicality_euclidean,
    "cosine": typicality_cosine,
    "lof": typicality_lof,
    "kde": typicality_kde,
}


# Clustering

def cluster_features(features: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """
    Cluster features using K-means.
    
    Paper uses scikit-learn KMeans for K <= 50, MiniBatchKMeans otherwise
    (for runtime efficiency).
    
    Args:
        features: [N, D] feature array.
        n_clusters: Number of clusters.
        random_state: Seed for reproducibility.
    
    Returns:
        cluster_labels: [N] array of cluster assignments.
    """
    if n_clusters <= 50:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    else:
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)

    cluster_labels = kmeans.fit_predict(features)
    return cluster_labels


# TypiClust Query Selection

def typiclust_query(
    features: np.ndarray,
    labeled_indices: list[int],
    budget: int,
    max_clusters: int = 500,
    k_typicality: int = 20,
    min_cluster_size: int = 5,
    typicality_fn: str = "euclidean",
    random_state: int = 42,
) -> list[int]:
    """
    TypiClust query selection (Algorithm 1 from the paper).
    
    Selects `budget` indices from the unlabeled pool that are both
    diverse (from different clusters) and typical (high density).
    
    Args:
        features: [N, D] feature array for ALL data points.
        labeled_indices: List of already-labeled indices.
        budget: Number of new examples to query (B).
        max_clusters: Maximum number of clusters (paper uses 500 for CIFAR-10).
        k_typicality: K for K-NN typicality (paper uses 20).
        min_cluster_size: Drop clusters smaller than this (paper uses 5).
        typicality_fn: Which typicality measure to use.
        random_state: Random seed.
    
    Returns:
        List of `budget` indices to query.
    """
    n = features.shape[0]
    labeled_set = set(labeled_indices)
    typ_fn = TYPICALITY_FUNCTIONS[typicality_fn]

    # Number of clusters: |L| + B, capped at max_clusters
    n_clusters = min(len(labeled_indices) + budget, max_clusters)
    n_clusters = max(n_clusters, budget)  # At least budget clusters

    # Step 2: Cluster for diversity
    cluster_labels = cluster_features(features, n_clusters, random_state=random_state)

    # Precompute cluster membership
    cluster_members = {}
    for c in range(n_clusters):
        cluster_members[c] = np.where(cluster_labels == c)[0]

    # Find uncovered clusters (those with fewest labeled points)
    # and select the most typical point from each
    queries = []
    exhausted_clusters = set()  # Track fully-labeled clusters

    while len(queries) < budget:
        # Count labeled points per cluster
        cluster_label_counts = {}
        cluster_sizes = {}
        for c in range(n_clusters):
            if c in exhausted_clusters:
                continue
            indices = cluster_members[c]
            cluster_sizes[c] = len(indices)
            cluster_label_counts[c] = sum(1 for idx in indices if idx in labeled_set)

        # Filter: only clusters with size >= min_cluster_size and not exhausted
        valid_clusters = [
            c for c in cluster_sizes
            if cluster_sizes[c] >= min_cluster_size and c not in exhausted_clusters
        ]

        if not valid_clusters:
            # Fallback: pick any unlabeled point
            unlabeled = [i for i in range(n) if i not in labeled_set]
            if not unlabeled:
                break  # No unlabeled points left at all
            queries.append(unlabeled[0])
            labeled_set.add(unlabeled[0])
            continue

        # Among valid clusters, find those with fewest labeled points
        min_labeled = min(cluster_label_counts[c] for c in valid_clusters)
        candidate_clusters = [
            c for c in valid_clusters if cluster_label_counts[c] == min_labeled
        ]

        # Among candidates, pick the largest cluster
        best_cluster = max(candidate_clusters, key=lambda c: cluster_sizes[c])

        # Get unlabeled points in this cluster
        cluster_indices = cluster_members[best_cluster]
        unlabeled_in_cluster = np.array(
            [idx for idx in cluster_indices if idx not in labeled_set]
        )

        if len(unlabeled_in_cluster) == 0:
            # FIX #2: Mark cluster as exhausted and retry (don't advance)
            exhausted_clusters.add(best_cluster)
            continue

        # FIX #1: Compute typicality WITHIN the cluster, using
        # min{k_typicality, cluster_size} neighbors (Appendix F.1)
        cluster_features_local = features[cluster_indices]
        k_local = min(k_typicality, len(cluster_indices) - 1)
        if k_local < 1:
            # Cluster too small for meaningful typicality; just pick first unlabeled
            best_idx = unlabeled_in_cluster[0]
        else:
            local_typicality = typ_fn(cluster_features_local, k=k_local)

            # Map local typicality back to global indices
            # Build a lookup: global_idx -> local_typicality_score
            local_typ_map = {
                int(cluster_indices[i]): local_typicality[i]
                for i in range(len(cluster_indices))
            }

            # Select the unlabeled point with highest cluster-local typicality
            best_idx = max(unlabeled_in_cluster, key=lambda idx: local_typ_map[int(idx)])

        queries.append(int(best_idx))
        labeled_set.add(int(best_idx))

    return queries


# Random baseline query

def random_query(
    n_total: int,
    labeled_indices: list[int],
    budget: int,
    random_state: int = 42,
) -> list[int]:
    """Random query selection baseline."""
    rng = np.random.RandomState(random_state)
    unlabeled = [i for i in range(n_total) if i not in set(labeled_indices)]
    selected = rng.choice(unlabeled, size=min(budget, len(unlabeled)), replace=False)
    return selected.tolist()