"""Minimal Flask dashboard for the crowd monitoring system.

Routes:
  GET  /                render the upload UI
  POST /analyze/image    upload an image, run the pipeline, show results
  POST /analyze/video     upload a short video, run the pipeline, show a summary + annotated video
  GET  /status            JSON system status (model loaded, fps, latency, errors, ...)
  GET  /api/analyze       JSON API: multipart image upload -> JSON result (no HTML)

Functionality over visual polish, per the project brief: this is plain
server-rendered HTML + a little vanilla JS, no build step.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from crowd_monitoring.config import load_config, pipeline_kwargs_from_config  # noqa: E402
from crowd_monitoring.pipeline import CrowdAnalysisPipeline  # noqa: E402
from crowd_monitoring.system_status import GLOBAL_STATUS  # noqa: E402
from crowd_monitoring.video import process_source  # noqa: E402
from crowd_monitoring.video_encoding import ensure_browser_compatible_video  # noqa: E402

RESULTS_DIR = REPO_ROOT / "dashboard" / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = REPO_ROOT / "dashboard" / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_PIPELINE_KWARGS = pipeline_kwargs_from_config(load_config(os.environ.get("CROWD_MONITORING_CONFIG")))
if os.environ.get("CROWD_MONITORING_MODEL"):
    _PIPELINE_KWARGS["model_name"] = os.environ["CROWD_MONITORING_MODEL"]
if os.environ.get("CROWD_MONITORING_WEIGHTS"):
    _PIPELINE_KWARGS["model_weights"] = os.environ["CROWD_MONITORING_WEIGHTS"]
if os.environ.get("CROWD_MONITORING_DEVICE"):
    _PIPELINE_KWARGS["device"] = os.environ["CROWD_MONITORING_DEVICE"]

MODEL_NAME = _PIPELINE_KWARGS["model_name"]
MODEL_WEIGHTS = _PIPELINE_KWARGS["model_weights"]

app = Flask(__name__)
_pipeline: CrowdAnalysisPipeline | None = None


def get_pipeline() -> CrowdAnalysisPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = CrowdAnalysisPipeline(**_PIPELINE_KWARGS)
    return _pipeline


def _draw_points(frame_bgr, points_xy):
    out = frame_bgr.copy()
    for x, y in points_xy:
        cv2.circle(out, (int(x), int(y)), 2, (0, 255, 255), -1)
    return out


@app.route("/")
def index():
    return render_template("index.html", model=MODEL_NAME, weights=MODEL_WEIGHTS)


@app.route("/status")
def status():
    return jsonify(GLOBAL_STATUS.snapshot().to_dict())


@app.route("/analyze/image", methods=["POST"])
def analyze_image():
    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template("index.html", model=MODEL_NAME, weights=MODEL_WEIGHTS, error="No image selected.")

    run_id = uuid.uuid4().hex[:10]
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    upload_path = UPLOAD_DIR / f"{run_id}_{secure_filename(file.filename)}"
    file.save(upload_path)

    frame = cv2.imread(str(upload_path))
    if frame is None:
        return render_template("index.html", model=MODEL_NAME, weights=MODEL_WEIGHTS, error="Could not decode image.")

    pipeline = get_pipeline()
    result = pipeline.analyze(frame)
    inf, loc, clusters, risk = result.inference, result.localization, result.clusters, result.risk

    cv2.imwrite(str(run_dir / "original.jpg"), frame)
    cv2.imwrite(str(run_dir / "heatmap.jpg"), result.heatmap_bgr)
    cv2.imwrite(str(run_dir / "overlay.jpg"), result.overlay_bgr)
    cv2.imwrite(str(run_dir / "points.jpg"), _draw_points(frame, loc.points_xy))

    summary = {
        "count": round(inf.count, 1),
        "model": f"{inf.model_name} ({inf.model_weights})",
        "device": inf.device,
        "latency_ms": round(inf.latency_ms, 1),
        "n_points": len(loc.points_xy),
        "n_clusters": clusters.n_clusters,
        "largest_cluster": clusters.largest_cluster_size,
        "avg_cluster_size": round(clusters.average_cluster_size, 2),
        "risk_status": risk.status,
        "risk_score": risk.score,
        "risk_explanation": risk.explanation,
        "warnings": inf.warnings,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return render_template(
        "result_image.html",
        run_id=run_id,
        summary=summary,
        model=MODEL_NAME,
        weights=MODEL_WEIGHTS,
    )


@app.route("/analyze/video", methods=["POST"])
def analyze_video():
    file = request.files.get("video")
    if not file or file.filename == "":
        return render_template("index.html", model=MODEL_NAME, weights=MODEL_WEIGHTS, error="No video selected.")

    run_id = uuid.uuid4().hex[:10]
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / f"{run_id}_{secure_filename(file.filename)}"
    file.save(upload_path)

    frame_skip = int(request.form.get("frame_skip", 2))
    max_frames = int(request.form.get("max_frames", 150))

    pipeline = get_pipeline()
    out_path = run_dir / "annotated.mp4"
    writer = None
    history = []
    peak_count = 0.0

    for pframe in process_source(pipeline, str(upload_path), frame_skip=frame_skip, max_frames=max_frames):
        r = pframe.result
        peak_count = max(peak_count, r.inference.count)
        annotated = r.overlay_bgr.copy()
        text = f"Count:{r.inference.count:.0f} Peak:{peak_count:.0f} Risk:{r.risk.status}"
        cv2.putText(annotated, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        if writer is None:
            h, w = annotated.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
        writer.write(annotated)
        history.append({"frame": pframe.state.frame_index, "count": round(r.inference.count, 1), "risk": r.risk.status})

    if writer is not None:
        writer.release()
        try:
            ensure_browser_compatible_video(out_path)
        except Exception as exc:
            print(f"Warning: ffmpeg conversion failed for {out_path}: {exc}")

    summary = {
        "frames_analyzed": len(history),
        "peak_count": round(peak_count, 1),
        "final_count": history[-1]["count"] if history else 0,
        "final_risk": history[-1]["risk"] if history else "SAFE",
        "history": history,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return render_template("result_video.html", run_id=run_id, summary=summary, model=MODEL_NAME, weights=MODEL_WEIGHTS)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "no image uploaded"}), 400
    import numpy as np

    data = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "could not decode image"}), 400

    pipeline = get_pipeline()
    result = pipeline.analyze(frame)
    inf, loc, clusters, risk = result.inference, result.localization, result.clusters, result.risk

    return jsonify(
        {
            "count": inf.count,
            "model": inf.model_name,
            "weights": inf.model_weights,
            "device": inf.device,
            "latency_ms": inf.latency_ms,
            "n_points": len(loc.points_xy),
            "clusters": {
                "n_clusters": clusters.n_clusters,
                "largest_cluster_size": clusters.largest_cluster_size,
                "average_cluster_size": clusters.average_cluster_size,
            },
            "risk": {"status": risk.status, "score": risk.score, "explanation": risk.explanation},
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
