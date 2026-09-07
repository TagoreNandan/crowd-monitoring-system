import numpy as np

from crowdmon.clustering import cluster_points


def test_two_tight_groups_form_two_clusters():
    rng = np.random.default_rng(2)
    group_a = rng.normal(loc=(10, 10), scale=1.0, size=(20, 2))
    group_b = rng.normal(loc=(200, 200), scale=1.0, size=(15, 2))
    points = np.vstack([group_a, group_b])

    stats = cluster_points(points, eps_px=10, min_samples=3)
    assert stats.n_clusters == 2
    assert stats.largest_cluster_size == 20
    assert stats.n_noise <= 2


def test_sparse_points_are_mostly_noise():
    rng = np.random.default_rng(3)
    points = rng.uniform(0, 1000, size=(10, 2))
    stats = cluster_points(points, eps_px=1.0, min_samples=5)
    assert stats.n_clusters == 0
    assert stats.n_noise == 10


def test_empty_points():
    stats = cluster_points(np.zeros((0, 2)))
    assert stats.n_clusters == 0
    assert stats.n_noise == 0
    assert stats.largest_cluster_size == 0
