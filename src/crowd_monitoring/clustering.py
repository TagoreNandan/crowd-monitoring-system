"""DBSCAN spatial clustering over localization points.

Same idea as the Crowd-Cluster-Monitoring-Heatmap reference project: group
nearby points into density-based clusters to find where people are
actually concentrating, rather than just the frame-wide average.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from sklearn.cluster import DBSCAN
    _HAS_SKLEARN = True
except Exception:
    DBSCAN = None
    _HAS_SKLEARN = False


def _numpy_dbscan(points_xy: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Pure-NumPy vectorized DBSCAN fallback when scikit-learn is blocked or unavailable."""
    n = len(points_xy)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    diff = points_xy[:, np.newaxis, :] - points_xy[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.where(dist[i] <= eps)[0]
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            labels[i] = cluster_id
            queue = [int(idx) for idx in neighbors if idx != i]
            while queue:
                current = queue.pop(0)
                if not visited[current]:
                    visited[current] = True
                    curr_neighbors = np.where(dist[current] <= eps)[0]
                    if len(curr_neighbors) >= min_samples:
                        for nbr in curr_neighbors:
                            nbr_int = int(nbr)
                            if not visited[nbr_int] and nbr_int not in queue:
                                queue.append(nbr_int)
                if labels[current] == -1:
                    labels[current] = cluster_id
            cluster_id += 1

    return labels


@dataclass
class ClusterStats:
    n_clusters: int
    n_noise: int
    cluster_sizes: list = field(default_factory=list)
    largest_cluster_size: int = 0
    largest_cluster_centroid: tuple | None = None
    largest_cluster_radius_px: float = 0.0
    average_cluster_size: float = 0.0
    labels: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))
    centroids: list = field(default_factory=list)


def cluster_points(points_xy: np.ndarray, eps_px: float = 40.0, min_samples: int = 3) -> ClusterStats:
    """Run DBSCAN over 2D points and summarize the resulting clusters.

    eps_px: max distance (pixels) between two points for them to be
        considered neighbors -- i.e. how "tight" a cluster must be.
    min_samples: minimum points to form a dense region (DBSCAN core point rule).
    """
    n = len(points_xy)
    if n == 0:
        return ClusterStats(n_clusters=0, n_noise=0)

    if n < min_samples:
        # Too few points for DBSCAN to find any dense region; everything is noise.
        return ClusterStats(n_clusters=0, n_noise=n, labels=np.full(n, -1))

    labels = None
    if _HAS_SKLEARN and DBSCAN is not None:
        try:
            db = DBSCAN(eps=eps_px, min_samples=min_samples).fit(points_xy)
            labels = db.labels_
        except Exception:
            labels = None

    if labels is None:
        labels = _numpy_dbscan(points_xy, eps=eps_px, min_samples=min_samples)

    unique_labels = sorted(set(labels) - {-1})
    n_clusters = len(unique_labels)
    n_noise = int(np.sum(labels == -1))

    cluster_sizes = []
    centroids = []
    for lbl in unique_labels:
        member_pts = points_xy[labels == lbl]
        cluster_sizes.append(len(member_pts))
        centroids.append(tuple(member_pts.mean(axis=0)))

    largest_size = 0
    largest_centroid = None
    largest_radius = 0.0
    if cluster_sizes:
        idx = int(np.argmax(cluster_sizes))
        largest_size = cluster_sizes[idx]
        largest_centroid = centroids[idx]
        member_pts = points_xy[labels == unique_labels[idx]]
        dists = np.linalg.norm(member_pts - np.array(largest_centroid), axis=1)
        largest_radius = float(dists.max()) if len(dists) else 0.0

    avg_size = float(np.mean(cluster_sizes)) if cluster_sizes else 0.0

    return ClusterStats(
        n_clusters=n_clusters,
        n_noise=n_noise,
        cluster_sizes=cluster_sizes,
        largest_cluster_size=largest_size,
        largest_cluster_centroid=largest_centroid,
        largest_cluster_radius_px=largest_radius,
        average_cluster_size=avg_size,
        labels=labels,
        centroids=centroids,
    )
