# Crowd Monitoring & Density Analysis System

An AI-based system that takes a photo, a video, or a webcam feed of a crowd
and produces: an estimated head count, a per-pixel density/heatmap
visualization, approximate crowd-concentration points, DBSCAN clustering of
those points, and a transparent rule-based SAFE/CAUTION/WARNING/CRITICAL
status -- all computed dynamically from real pretrained-model inference on
whatever you feed it. Nothing here is hardcoded, mocked, or pre-baked.

```
input image/video/webcam
        │
        ▼
 DM-Count / CSRNet  (density-regression CNN, pretrained)
        │
        ├──► integrated density map  ──► count
        ├──► colorized heatmap + overlay
        ├──► local density maxima  ──► approximate localization points
        │            │
        │            ▼
        │      DBSCAN clustering  ──► cluster count / largest cluster / concentration
        │
        ▼
 rule-based risk engine (count, density, concentration, growth trend)
        │
        ▼
 CLI output (images/JSON) or Flask dashboard
```

---

## 1. Model selection -- why DM-Count, not FIDTM

The two primary references named for this project were
[Crowd-Cluster-Monitoring-Heatmap](https://github.com/rhalbhavi/Crowd-Cluster-Monitoring-Heatmap)
(detector + DBSCAN + heatmap, the UX/analytics pattern this project follows)
and [FIDTM](https://github.com/dk-liang/FIDTM) (Focal Inverse Distance
Transform maps, a strong crowd-*localization* model). FIDTM was evaluated
first, seriously, and ultimately **not used**, for a concrete, checkable
reason rather than a stylistic preference:

- FIDTM's pretrained checkpoints (ShanghaiA/B, UCF-QNRF, JHU-Crowd++,
  NWPU-Crowd) are published **only on Baidu Disk and OneDrive**. Both were
  tested from this environment and are network-unreachable here (and Baidu
  Disk in particular is frequently unreachable/rate-limited even outside
  sandboxed environments, and needs an extraction password). FIDTM also
  needs an HRNet backbone for its best results, which adds another
  ImageNet-pretrained-weight dependency on top of that. "Prefer a strong
  pretrained model that can realistically be integrated and demonstrated
  tonight" ruled it out in practice, not in theory -- the weights simply
  could not be fetched.
- The four reference-tier repos were also inspected. `Crowd-Cluster-Monitoring-Heatmap`
  uses MobileNet-SSD person detection (good pattern for the clustering/heatmap
  UX, weak on occlusion since it's a generic detector). `Crowd-Management-System`
  fine-tunes YOLO for head detection (custom, unverified training data/license).
  `Crowd-Management-and-Stampede-protection` and `CrowdShield` are full
  detector-based (YOLOv8 person-detection) applications with FastAPI/Next.js
  stacks -- useful for the risk-engine design pattern (see `risk.py`), but
  the same detector-based approach undercounts in heavy occlusion, which is
  exactly the failure mode this project needs to avoid for the "extremely
  dense crowd" reference images.

What was actually used instead: **DM-Count** (VGG19 backbone,
density-regression), with **CSRNet** available as a second architecture.
Both are the same class of model FIDTM belongs to -- they predict a
per-pixel density field and integrate it to a count, so they don't need to
resolve individual, possibly heavily-overlapping people the way a detector
does, which is precisely what makes density-regression models handle severe
occlusion / extremely dense crowds well. The deciding, practical factor:
their pretrained checkpoints (ported from the original DM-Count/CSRNet
papers) are mirrored as **plain GitHub Release assets** by the
[LWCC project](https://github.com/tersekmatija/lwcc)
(`tersekmatija/lwcc_weights`), which this environment (and most restricted
networks) *can* actually reach -- verified with a real download and a real
loaded/forward-passed model, not just a README claim.

| | FIDTM | Generic detector (YOLO/MobileNet-SSD) | **DM-Count / CSRNet (chosen)** |
|---|---|---|---|
| Dense/occluded crowd accuracy | Excellent (designed for this) | Poor -- misses overlapping heads | Good -- regresses density, no per-instance resolution needed |
| Individual head points | Yes (native) | Yes (native) | No -- approximated via density-map peaks (documented, see §5) |
| Weight hosting | Baidu Disk / OneDrive (**unreachable here**) | GitHub (ultralytics releases) | **GitHub Releases (reachable, verified)** |
| CPU support | Yes | Yes | Yes (verified) |
| Apple MPS / CUDA | Not device-abstracted in upstream code | Yes (ultralytics) | Yes -- this repo adds device selection (upstream LWCC is CPU-only) |
| Backbone weight deps | Needs ImageNet-pretrained HRNet too | Self-contained | Self-contained (checkpoint already includes the VGG backbone) |

Given all of that, DM-Count (default: **QNRF** weights, which span
~50-12,000 people/image, the widest range and the best default for unknown
crowd density) is the model actually wired into the pipeline. CSRNet
(SHA/SHB) is included as a swappable alternative for comparison. If you
have unrestricted internet access on your own machine and want to plug
FIDTM in later for its native head-point output, the pipeline's model
interface (`crowdmon/inference.py`, `CrowdCounter`) is intentionally small
and swappable -- see §10.

---

## 2. What's real vs. approximated (read this before you trust a number)

- **Count and density map**: real model output, every time, from an actual
  forward pass through pretrained DM-Count/CSRNet weights. Verified during
  development on a genuine street-scene photo (not a reference image, not
  synthetic) -- see §9.
- **Localization points**: DM-Count/CSRNet were **not** trained to output
  individual coordinates (that's what a model like FIDTM or a detector is
  for). Rather than inventing points, this project extracts *local maxima*
  of the predicted density map -- the same idea FIDTM itself uses to turn a
  density-like map into discrete points. This tracks people reasonably well
  in sparse/medium crowds; in a very dense, heavily-occluded crowd the
  individual bumps merge and the *number of peaks undercounts* the true
  count. **The integrated density (the model's count) stays authoritative in
  that regime, not the number of peaks.** This limitation is documented
  in-code in `crowdmon/localization.py`, not glossed over.
- **Clusters**: real DBSCAN (`scikit-learn`) over whatever points were
  extracted above -- garbage in, garbage out, same as the point-extraction
  caveat.
- **Risk status**: a transparent, tunable heuristic (`crowdmon/risk.py`),
  explicitly **not** a validated stampede-prediction model. See §6.
- **PSNR/SSIM**: only ever computed in evaluation mode, against
  user-supplied ground truth. Never shown for a normal, no-ground-truth
  inference run, because there'd be nothing correct to compare against. See
  §7.

---

## 3. Installation

```bash
git clone <this repo>
cd crowd-monitoring-system
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .          # installs the `crowdmon` package (src/ layout)
```

Requires Python >= 3.9. `requirements.txt` pulls PyTorch from PyPI directly
(no CUDA-specific index needed -- the standard Linux/macOS wheel runs fine
CPU-only and picks up CUDA/MPS automatically if present at runtime).

### Pretrained weights

Weights are **not** committed to git (they're ~85 MB `.pth` files). They
download automatically the first time you run inference, or fetch them
up front:

```bash
python scripts/download_weights.py                      # default: DM-Count / QNRF
python scripts/download_weights.py --model DM-Count --weights SHA
python scripts/download_weights.py --all                # every model/weights combo
```

They land in `weights/` (see `weights/README.md`), sourced from
`https://github.com/tersekmatija/lwcc_weights/releases/download/v0.1/<name>.pth`
-- a plain HTTPS GET, no Google Drive/Baidu client needed.

| `--weights` | Trained on | Best for |
|---|---|---|
| `QNRF` (default) | UCF-QNRF | Unknown/mixed density -- widest range (~50-12,000 people/image) |
| `SHA` | ShanghaiTech A | Dense, mixed indoor/outdoor crowds |
| `SHB` | ShanghaiTech B | Sparser street-level crowds |

`--model CSRNet` supports `SHA`/`SHB` only (no QNRF checkpoint published).

---

## 4. Usage

### Image

```bash
python scripts/run_image.py --input examples/your_photo.jpg --output output/
python scripts/run_image.py --input examples/photos_dir/ --output output/     # batch, whole directory
python scripts/run_image.py --input photo.jpg --output output/ --model CSRNet --weights SHB
python scripts/run_image.py --input photo.jpg --output output/ --no-resize    # skip downscaling for very dense crowds
```

Writes, per image: `<name>_heatmap.png` (colorized density map),
`<name>_overlay.png` (heatmap blended on the original), `<name>_points.png`
(approximate localization points drawn on the original), and
`<name>_result.json` (count, clusters, risk, latency, device, warnings).

### Video / webcam / image directory

```bash
python scripts/run_video.py --input clip.mp4 --output output/annotated.mp4
python scripts/run_video.py --input 0 --output output/webcam.mp4 --display       # webcam index 0, live preview
python scripts/run_video.py --input clip.mp4 --output output/out.mp4 --frame-skip 3  # analyze every 4th frame (faster on CPU)
python scripts/run_video.py --input frames_dir/ --output output/out.mp4          # a directory of images treated as a sequence
```

Prints per-frame count / peak / clusters / risk / FPS to the console, writes
an annotated output video and a `<output>.history.json` count-history log.
`--display` opens a live `cv2.imshow` preview window (requires an actual
display; the script degrades to headless automatically if none is
available).

### Dashboard

```bash
python dashboard/app.py                      # http://localhost:5000
CROWDMON_WEIGHTS=SHA python dashboard/app.py  # override model weights
PORT=8080 python dashboard/app.py             # override port
```

Upload an image or a short video from the browser; results (original,
heatmap, overlay, localization points, cluster stats, risk status, latency)
render on the page. `/status` returns live backend state as JSON (model
loaded, device, FPS, last latency, error count, ONLINE/DEGRADED/OFFLINE) and
`/api/analyze` is a plain JSON API (multipart image upload in, JSON result
out) for programmatic use.

### Evaluation mode (PSNR/SSIM/MAE/RMSE)

```bash
python scripts/run_image.py --input photo.jpg --output output/ --gt-points gt_points.json
```

where `gt_points.json` is `[[x1,y1], [x2,y2], ...]` -- ground-truth head
coordinates for that exact image. See §7 for what this does and doesn't mean.

---

## 5. Head/point localization -- explicit limitation

Covered in §2 above; repeating the key point because it matters for how you
read the output: `*_points.png` and the `n_points`/cluster numbers are a
**derived approximation** (density-map local maxima), not a dedicated
head-detection network's output. Trust the model's integrated count for
"how many people," and treat the points as "roughly where the crowd is
concentrated" for clustering purposes -- especially in dense scenes, where
peak count runs well below the true count.

---

## 6. Crowd risk / status engine

`crowdmon/risk.py` computes a 0-100 score by blending four signals:

1. **Count** vs. absolute thresholds (`count_caution/warning/critical`).
2. **Relative density** -- estimated people per 100x100px patch of the frame.
   This is a unitless proxy for "how packed the frame looks," **not** a
   calibrated people-per-square-meter figure (that needs known camera
   height/angle, which a single arbitrary image doesn't provide).
3. **Cluster concentration** -- fraction of all localization points inside
   the single largest DBSCAN cluster; high means "crowd bunched in one
   spot," not spread out.
4. **Growth trend** (video only) -- relative change in count over a rolling
   window; `None`/neutral for a single image or the first few video frames,
   never penalized as if it were zero growth.

The blended score maps to `SAFE` (<25) / `CAUTION` (<50) / `WARNING` (<75) /
`CRITICAL` (>=75). Every threshold and blend weight lives in one place
(`RiskConfig`, overridable via `configs/default.yaml` under `risk:`) so it
can be calibrated per camera/venue.

**This is a rule-based heuristic for situational awareness, not a
scientifically validated stampede-prediction model.** Do not use it as the
sole basis for a real safety-critical decision.

---

## 7. Evaluation mode: what PSNR/SSIM actually mean here

PSNR and SSIM are only meaningful when comparing two things of the same
kind -- here, a predicted density map against a **ground-truth** density
map built from real head annotations for that exact image. Academic papers
(FIDTM included) report PSNR/SSIM only on benchmark datasets
(ShanghaiTech/UCF-QNRF/...) that ship such ground truth; there is none for
an arbitrary photo you drop into this demo.

So: **normal inference never computes or displays PSNR/SSIM** -- only count,
density map, heatmap, and localization/clusters, all of which are
well-defined without ground truth.

`crowdmon/evaluation.py` provides evaluation mode, used only when you supply
ground truth explicitly:

- `evaluate_with_gt_points(pred_density, gt_points_xy, image_size)` --
  builds a ground-truth density map from head-point annotations (a Gaussian
  placed at each point, the standard construction in the literature) and
  reports **MAE**, **RMSE**, **PSNR**, **SSIM**.
- `evaluate_density_maps(pred_density, gt_density)` -- if you already have a
  ground-truth density map (`.npy`, or from a benchmark dataset's `.h5`).
- `evaluate_count_only(pred_count, gt_count)` -- if you only know the true
  head count for an image (no points/map): reports MAE/RMSE only, and
  **explicitly returns `psnr=None, ssim=None`** rather than a number that
  would look meaningful but isn't.

Wired into the image CLI via `--gt-points path/to/points.json`.

---

## 8. System status

`crowdmon/system_status.py` tracks only what this process can actually
observe: whether the model is loaded, the input source connected, FPS,
last inference latency, error count, and time since the last successful
inference. It reports `ONLINE` / `DEGRADED` (recent error, or inference has
gone stale) / `OFFLINE` (model not loaded). **It never claims a cloud
connection** -- everything runs in this process; there is no external
service behind the status.

---

## 9. Testing -- what was actually run, not just claimed

All of the following were executed in this environment, not asserted from
memory:

- `pytest tests/` -- **49 tests, all passing**: module imports, device
  resolution/fallback, a **real** forward pass through the loaded DM-Count
  model on a synthetic image (not mocked), heatmap/overlay generation,
  density-map peak extraction, DBSCAN clustering, the risk engine's
  monotonicity and threshold behavior, evaluation-mode MAE/RMSE/PSNR/SSIM
  math, video/image-directory processing (including a real, generated
  `.mp4` read back through OpenCV), system-status state transitions, YAML
  config loading, and the Flask dashboard's routes via its test client
  (including a real multipart image upload through `/analyze/image` and
  `/api/analyze`).
- **Real end-to-end CLI run on a genuine (non-reference) street photo**:
  `scripts/run_image.py` against a frame pulled from an unrelated test
  video, DM-Count/QNRF estimated **22.7 people** for a scene with roughly
  15-25 visible pedestrians, and the generated heatmap visibly lit up on
  individual heads in the photo -- inspected directly, not just numerically
  sanity-checked.
- **Real end-to-end video run**: `scripts/run_video.py` against a genuine
  ~2-minute test clip (sampled), producing a real annotated `.mp4` and a
  real per-frame count-history JSON, with peak-count tracking working
  across frames.
- **Real dashboard run**: `python dashboard/app.py`, then live `curl`
  requests against `/status`, `/api/analyze`, and `/analyze/image` with an
  actual uploaded image -- confirmed the status flips `OFFLINE` -> `ONLINE`
  once a real inference completes, and the rendered result page contains
  the real computed count.
- **Weight download path exercised for real**: both `DM-Count_SHA.pth` and
  `DM-Count_QNRF.pth` (~86 MB each) were downloaded from the GitHub
  Releases URL by the actual `ensure_weights()` code path (not a stub), and
  the resulting checkpoints were loaded into the real model architecture
  and used for the runs above.

What wasn't (and, honestly, couldn't be) tested here: an actual webcam
(this is a headless remote container with no camera), and CUDA/MPS
execution (this container has neither a GPU nor Apple Silicon) -- the
device-selection code (`crowdmon/device.py`) has explicit tests for its
fallback logic, but the CUDA/MPS *code paths themselves* only get exercised
on hardware that has them. If you run this on a CUDA machine or an Apple
Silicon Mac, `--device auto` will pick them up automatically; please
sanity-check that one run there.

---

## 10. Hardware compatibility (CPU / MPS / CUDA)

`crowdmon/device.py` resolves `"auto"` to CUDA > Apple MPS > CPU, in that
order, and every script/dashboard exposes `--device {auto,cpu,cuda,mps}` (or
`device:` in the YAML config). Nothing in the pipeline hardcodes a backend.
If a chosen backend fails at model-load or first-inference time (e.g. an op
not yet supported on a given MPS/PyTorch version), `CrowdCounter`
automatically retries once on CPU and records a warning
(`InferenceResult.warnings`) rather than crashing.

This development/test environment is a CPU-only Linux container (no CUDA,
no MPS) -- CPU inference measured at roughly **1.5-2.5 seconds per frame**
at the default 1024px max side on this machine's CPU. Expect it to be
faster on a real GPU or Apple Silicon, and faster still with a smaller
`max_side` (`configs/default.yaml` -> `model.max_side`) or a larger
`--frame-skip` on video.

---

## 11. Project structure

```
crowd-monitoring-system/
├── src/crowdmon/           installable package
│   ├── models/              DM-Count / CSRNet architectures + weight registry & downloader
│   ├── device.py             CUDA/MPS/CPU resolution with fallback
│   ├── inference.py          CrowdCounter: preprocessing + forward pass -> count + density map
│   ├── heatmap.py            density map -> colorized heatmap / overlay
│   ├── localization.py       density-map peak extraction (approximate points)
│   ├── clustering.py         DBSCAN over points -> cluster stats
│   ├── risk.py                rule-based SAFE/CAUTION/WARNING/CRITICAL engine
│   ├── evaluation.py          MAE/RMSE/PSNR/SSIM against supplied ground truth
│   ├── video.py               video/webcam/image-dir loop, count history, peak tracking
│   ├── system_status.py       ONLINE/DEGRADED/OFFLINE backend status tracker
│   ├── config.py              YAML config loading
│   └── pipeline.py            wires all of the above together per frame
├── scripts/
│   ├── download_weights.py
│   ├── run_image.py
│   └── run_video.py
├── dashboard/                Flask app + templates (upload UI, JSON API, status)
├── configs/default.yaml      model / clustering / risk thresholds
├── weights/                   pretrained checkpoints (gitignored, downloaded on demand)
├── examples/                  drop your own test images/video here
└── tests/                     49 tests covering every module above
```

---

## 12. Troubleshooting

- **"Pretrained weights not found"** -- run
  `python scripts/download_weights.py` (or let the first inference call
  trigger the download; it needs outbound HTTPS to `github.com` /
  `objects.githubusercontent.com`).
- **Slow on CPU** -- lower `model.max_side` in `configs/default.yaml`, use
  `--frame-skip` on video, or use `SHB` weights (lighter distribution to
  fit, though the architecture/compute cost is identical across weight
  sets -- the real lever is `max_side`).
- **MPS/CUDA silently falls back to CPU** -- check
  `InferenceResult.warnings` / the dashboard's `/status` `last_error`; the
  pipeline retries on CPU rather than crashing, by design, but that means a
  hardware issue can look like "it's just slow" if you don't check.
- **`--display` does nothing / errors** -- you're running headless (e.g.
  over SSH or in a container with no X server); the script detects this and
  continues without a live preview, output video/JSON are unaffected.
- **Localization/cluster counts look low for a dense crowd** -- expected,
  see §5. Trust the count field, not the point count, for very dense scenes.
