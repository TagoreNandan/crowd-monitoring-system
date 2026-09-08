from crowd_monitoring.risk import RiskConfig, STATUS_LEVELS, assess_risk


def test_low_count_is_safe():
    r = assess_risk(count=2, frame_area_px=1000 * 1000, largest_cluster_size=1, n_points=2)
    assert r.status == "SAFE"


def test_high_count_and_concentration_is_critical():
    r = assess_risk(
        count=500,
        frame_area_px=200 * 200,  # small frame -> very high relative density
        largest_cluster_size=480,
        n_points=500,
        growth_ratio=0.9,
    )
    assert r.status == "CRITICAL"
    assert r.score > 75


def test_status_monotonic_with_count():
    scores = []
    for count in [0, 10, 50, 100, 300]:
        r = assess_risk(count=count, frame_area_px=1000 * 1000, largest_cluster_size=count // 2, n_points=max(count, 1))
        scores.append(r.score)
    assert all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1))


def test_missing_growth_is_neutral_not_penalized():
    r1 = assess_risk(count=50, frame_area_px=1000 * 1000, largest_cluster_size=20, n_points=50, growth_ratio=None)
    r2 = assess_risk(count=50, frame_area_px=1000 * 1000, largest_cluster_size=20, n_points=50, growth_ratio=0.0)
    assert abs(r1.score - r2.score) < 1e-6


def test_all_status_levels_reachable_with_custom_thresholds():
    cfg = RiskConfig(count_caution=10, count_warning=20, count_critical=30, weight_count=1, weight_density=0, weight_concentration=0, weight_growth=0)
    statuses = set()
    for count in [1, 12, 22, 40]:
        r = assess_risk(count=count, frame_area_px=1e9, largest_cluster_size=0, n_points=0, config=cfg)
        statuses.add(r.status)
    assert statuses <= set(STATUS_LEVELS)
    assert len(statuses) > 1
