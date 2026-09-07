#!/usr/bin/env python3
"""Run crowd analysis on a single image (or a directory of images).

Examples:
    python scripts/run_image.py --input photo.jpg --output out/
    python scripts/run_image.py --input photos/ --output out/ --model DM-Count --weights SHA
    python scripts/run_image.py --input photo.jpg --output out/ --gt-points gt.json   # evaluation mode
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from crowdmon.config import load_config, pipeline_kwargs_from_config
from crowdmon.evaluation import evaluate_with_gt_points
from crowdmon.pipeline import CrowdAnalysisPipeline

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_points(frame_bgr, points_xy, color=(0, 255, 255), radius=2):
    out = frame_bgr.copy()
    for x, y in points_xy:
        cv2.circle(out, (int(x), int(y)), radius, color, -1)
    return out


def process_one(pipeline: CrowdAnalysisPipeline, img_path: Path, out_dir: Path, gt_points_path: Path | None, no_resize: bool):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  ! could not read {img_path}, skipping")
        return None

    result = pipeline.analyze(frame, resize=not no_resize)
    inf, loc, clusters, risk = result.inference, result.localization, result.clusters, result.risk

    stem = img_path.stem
    cv2.imwrite(str(out_dir / f"{stem}_heatmap.png"), result.heatmap_bgr)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), result.overlay_bgr)
    points_img = draw_points(frame, loc.points_xy)
    cv2.imwrite(str(out_dir / f"{stem}_points.png"), points_img)

    summary = {
        "image": str(img_path),
        "estimated_count": round(inf.count, 2),
        "model": inf.model_name,
        "weights": inf.model_weights,
        "device": inf.device,
        "latency_ms": round(inf.latency_ms, 1),
        "localization": {
            "method": loc.method,
            "n_points": int(len(loc.points_xy)),
        },
        "clusters": {
            "n_clusters": clusters.n_clusters,
            "n_noise": clusters.n_noise,
            "largest_cluster_size": clusters.largest_cluster_size,
            "average_cluster_size": round(clusters.average_cluster_size, 2),
        },
        "risk": {
            "status": risk.status,
            "score": risk.score,
            "explanation": risk.explanation,
            "components": risk.components,
        },
        "warnings": inf.warnings,
    }

    if gt_points_path is not None:
        with open(gt_points_path) as f:
            gt_points = np.array(json.load(f), dtype=np.float32)
        evaln = evaluate_with_gt_points(inf.density_map, gt_points, inf.original_size)
        summary["evaluation"] = {
            "mae": round(evaln.mae, 2),
            "rmse": round(evaln.rmse, 2),
            "psnr": None if evaln.psnr is None else round(evaln.psnr, 2),
            "ssim": None if evaln.ssim is None else round(evaln.ssim, 4),
            "gt_count": round(evaln.gt_count, 2),
            "note": evaln.note,
        }

    with open(out_dir / f"{stem}_result.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"  {img_path.name}: count={inf.count:.1f} clusters={clusters.n_clusters} "
        f"largest_cluster={clusters.largest_cluster_size} risk={risk.status} "
        f"({inf.latency_ms:.0f} ms on {inf.device})"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Image file or directory of images.")
    parser.add_argument("--output", default="output", help="Directory to write results to.")
    parser.add_argument("--config", default=None, help="Path to a YAML config (default: configs/default.yaml).")
    parser.add_argument("--model", default=None, choices=["DM-Count", "CSRNet"], help="Overrides config.")
    parser.add_argument("--weights", default=None, help="SHA, SHB, or QNRF (DM-Count only). Overrides config.")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"], help="Overrides config.")
    parser.add_argument("--no-resize", action="store_true", help="Disable downscaling; recommended for very dense crowds if you have the memory.")
    parser.add_argument("--gt-points", help="JSON file of [[x,y], ...] ground-truth head points, for evaluation mode.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    kwargs = pipeline_kwargs_from_config(load_config(args.config))
    if args.model:
        kwargs["model_name"] = args.model
    if args.weights:
        kwargs["model_weights"] = args.weights
    if args.device:
        kwargs["device"] = args.device
    pipeline = CrowdAnalysisPipeline(**kwargs)

    if in_path.is_dir():
        images = sorted(p for p in in_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            print(f"No images found in {in_path}")
            return
        print(f"Processing {len(images)} images from {in_path} ...")
        results = []
        for img_path in images:
            r = process_one(pipeline, img_path, out_dir, Path(args.gt_points) if args.gt_points else None, args.no_resize)
            if r:
                results.append(r)
        with open(out_dir / "batch_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {len(results)} results to {out_dir}")
    else:
        process_one(pipeline, in_path, out_dir, Path(args.gt_points) if args.gt_points else None, args.no_resize)
        print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()
