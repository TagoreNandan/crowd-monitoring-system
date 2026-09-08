import numpy as np

from crowd_monitoring.evaluation import build_gt_density_map, evaluate_count_only, evaluate_density_maps, evaluate_with_gt_points


def test_build_gt_density_map_integrates_to_point_count():
    points = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
    density = build_gt_density_map(points, image_size=(50, 50), sigma=3.0)
    assert abs(float(density.sum()) - 3.0) < 0.05


def test_identical_maps_have_high_psnr_and_ssim():
    d = np.zeros((40, 40), dtype=np.float32)
    d[10:15, 10:15] = 1.0
    result = evaluate_density_maps(d, d.copy())
    assert result.mae < 1e-6
    assert result.psnr is not None and result.psnr > 40  # near-identical -> very high PSNR
    assert result.ssim is not None and result.ssim > 0.99


def test_different_maps_have_lower_similarity_than_identical():
    d1 = np.zeros((40, 40), dtype=np.float32)
    d1[5:10, 5:10] = 1.0
    d2 = np.zeros((40, 40), dtype=np.float32)
    d2[30:35, 30:35] = 1.0

    identical = evaluate_density_maps(d1, d1.copy())
    different = evaluate_density_maps(d1, d2)
    assert different.ssim < identical.ssim


def test_evaluate_with_gt_points_reports_gt_count():
    pred = np.zeros((60, 60), dtype=np.float32)
    pred[30, 30] = 1.0
    gt_points = np.array([[30, 30], [30, 32]], dtype=np.float32)
    result = evaluate_with_gt_points(pred, gt_points, image_size=(60, 60), sigma=3.0)
    assert abs(result.gt_count - 2.0) < 0.1


def test_count_only_evaluation_has_no_psnr_ssim():
    result = evaluate_count_only(pred_count=45.0, gt_count=50.0)
    assert result.mae == 5.0
    assert result.psnr is None
    assert result.ssim is None
