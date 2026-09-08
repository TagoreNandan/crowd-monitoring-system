"""DBSCAN spatial clustering over localization points.

Same idea as the Crowd-Cluster-Monitoring-Heatmap reference project: group
nearby points into density-based clusters to find where people are
actually concentrating, rather than just the frame-wide average.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import DBSCAN


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

    db = DBSCAN(eps=eps_px, min_samples=min_samples).fit(points_xy)
    labels = db.labels_

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
