#!/usr/bin/env python3
"""Run crowd analysis on a video file, a directory of frames, or a webcam.

Examples:
    python scripts/run_video.py --input clip.mp4 --output out/annotated.mp4
    python scripts/run_video.py --input 0 --output out/webcam.mp4 --display   # webcam, live preview
    python scripts/run_video.py --input clip.mp4 --output out/annotated.mp4 --frame-skip 2  # faster on CPU

Prints running count / peak count / FPS / risk status to the console for
every analyzed frame, and (optionally) shows a live annotated preview
window with --display. A JSON count-history log is written alongside the
output video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from crowdmon.config import load_config, pipeline_kwargs_from_config
from crowdmon.pipeline import CrowdAnalysisPipeline
from crowdmon.video import process_source


def annotate(frame_bgr, result, state):
    out = result.overlay_bgr.copy()
    inf, clusters, risk = result.inference, result.clusters, result.risk

    lines = [
        f"Count: {inf.count:.0f}  Peak: {state.peak_count:.0f}",
        f"Clusters: {clusters.n_clusters}  Largest: {clusters.largest_cluster_size}",
        f"Risk: {risk.status} ({risk.score:.0f})",
        f"FPS: {state.fps_estimate:.1f}  Latency: {inf.latency_ms:.0f} ms  [{inf.device}]",
    ]
    y = 24
    for line in lines:
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Video file path, webcam index (e.g. 0), or image directory.")
    parser.add_argument("--output", default="output/annotated.mp4", help="Output annotated video path.")
    parser.add_argument("--config", default=None, help="Path to a YAML config (default: configs/default.yaml).")
    parser.add_argument("--model", default=None, choices=["DM-Count", "CSRNet"], help="Overrides config.")
    parser.add_argument("--weights", default=None, help="Overrides config.")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda", "mps"], help="Overrides config.")
    parser.add_argument("--frame-skip", type=int, default=0, help="Analyze every (N+1)th frame.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--display", action="store_true", help="Show a live preview window (requires a display).")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kwargs = pipeline_kwargs_from_config(load_config(args.config))
    if args.model:
        kwargs["model_name"] = args.model
    if args.weights:
        kwargs["model_weights"] = args.weights
    if args.device:
        kwargs["device"] = args.device
    pipeline = CrowdAnalysisPipeline(**kwargs)

    writer = None
    history_log = []
    prev_t = None
    import time

    for pframe in process_source(pipeline, args.input, frame_skip=args.frame_skip, max_frames=args.max_frames):
        now = time.time()
        if prev_t is not None:
            dt = now - prev_t
            if dt > 0:
                pframe.state.fps_estimate = 1.0 / dt
        prev_t = now

        annotated = annotate(pframe.frame_bgr, pframe.result, pframe.state)

        if writer is None:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, 20.0, (w, h))
        writer.write(annotated)

        r = pframe.result
        print(
            f"frame {pframe.state.frame_index}: count={r.inference.count:.1f} "
            f"peak={pframe.state.peak_count:.1f} risk={r.risk.status} "
            f"fps={pframe.state.fps_estimate:.1f}"
        )
        history_log.append(
            {
                "frame": pframe.state.frame_index,
                "count": round(r.inference.count, 2),
                "risk_status": r.risk.status,
                "risk_score": r.risk.score,
                "n_clusters": r.clusters.n_clusters,
                "latency_ms": round(r.inference.latency_ms, 1),
            }
        )

        if args.display:
            try:
                cv2.imshow("Crowd Monitoring", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error as exc:
                print(f"(--display unavailable in this environment: {exc}); continuing headless.")
                args.display = False

    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()

    log_path = out_path.with_suffix(".history.json")
    with open(log_path, "w") as f:
        json.dump(history_log, f, indent=2)

    peak = max((h["count"] for h in history_log), default=0)
    print(f"\nDone. {len(history_log)} frames analyzed. Peak count: {peak}")
    print(f"Annotated video: {out_path}")
    print(f"Count history:   {log_path}")


if __name__ == "__main__":
    main()
