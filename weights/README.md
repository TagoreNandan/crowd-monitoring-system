# weights/

Pretrained checkpoints are downloaded here automatically the first time you
run inference (or explicitly via `scripts/download_weights.py`). They are
**not** committed to git (see `.gitignore`) -- `.pth` files are ~85 MB each.

| File | Model | Trained on | Notes |
|---|---|---|---|
| `DM-Count_QNRF.pth` | DM-Count | UCF-QNRF | Default. Best all-around choice; QNRF spans ~50-12,000 people/image. |
| `DM-Count_SHA.pth` | DM-Count | ShanghaiTech A | Dense, mixed indoor/outdoor crowds. |
| `DM-Count_SHB.pth` | DM-Count | ShanghaiTech B | Sparser street-level crowds. |
| `CSRNet_SHA.pth` | CSRNet | ShanghaiTech A | Alternate backbone/cross-check. |
| `CSRNet_SHB.pth` | CSRNet | ShanghaiTech B | Alternate backbone/cross-check. |

Source: these are the original DM-Count / CSRNet checkpoints as mirrored by
the [LWCC project](https://github.com/tersekmatija/lwcc) on GitHub Releases
(`tersekmatija/lwcc_weights`), fetched with a plain HTTPS GET -- no Google
Drive / Baidu Disk / OneDrive client needed. See the top-level README for why
that mattered for this project.

## Person detector: YOLOv8n trained on CrowdHuman (real, visible bounding boxes)

| File | Purpose |
|---|---|
| `yolov8n_crowdhuman.pt` | YOLOv8n architecture, fine-tuned on CrowdHuman (~5.9 MB). **Default.** |
| `yolov8n.pt` | Ultralytics YOLOv8n, COCO-pretrained (~6.2 MB). Fallback only. |

This backs `crowd_monitoring.detection.PersonDetector`, used by the FastAPI backend
(image jobs, video jobs, and real-time feeds) to draw real, visible person
bounding boxes (`visible_count`/`detections`) as a signal *separate* from
DM-Count's own dense `estimated_count` -- DM-Count handles occlusion-robust
crowd-density estimation and never sees these files, while the detector finds
individual people that are large/resolvable enough for a modern detector to
find. It is not a substitute for DM-Count and intentionally won't produce a
box for every person in a heavily-occluded, very-dense crowd -- that's the
documented `visible != estimated` distinction, not a bug.

**Model history on this project** (each step replaced after direct
comparison on this project's own sample crowd photo/video, not guesswork):

1. MobileNet-SSD (2016 Caffe SSD300): 0-2 boxes on the sample photo, 0 on a
   4K sample video frame. Replaced -- couldn't do the job at all.
2. COCO-pretrained YOLOv8n/YOLOv8m (general 80-class detector, "person" is
   1 of 80): a single full-frame pass finds ~16-28 boxes. Trying to close the
   gap with SAHI-style 32-tile inference did find more (~163-193 boxes) but
   at 850ms/frame (YOLOv8n tiles) to **4.6 seconds/frame** (YOLOv8m tiles) --
   unusable for real-time, and the bigger model was both slower AND found
   *fewer* boxes tiled than the small one. Replaced -- treating the symptom
   (not enough boxes) by brute-forcing more inference passes of a
   crowd-naive model, rather than fixing the cause.
3. **YOLOv8n trained on CrowdHuman** (current): CrowdHuman is a dataset built
   specifically for dense, heavily-occluded crowds (~340k person instances
   across ~15k images). A single full-frame pass at imgsz=1600 finds
   **~135-226 boxes** on the same sample files in **~83-84ms** -- more boxes
   than the old 32-tile approach, at roughly 1/10th the latency, with no
   tiling needed.

Tiling (`generate_tiles`/`nms_boxes` in `src/crowd_monitoring/detection.py`) is still
available as an opt-in (`PersonDetector(use_tiling=True)`) for a caller with
a generous latency budget that wants to squeeze out a few more of the
tiniest/farthest boxes on a single still image -- it is not used by default
anywhere in this codebase.

Source: [yakhyo/yolov8-crowdhuman](https://github.com/yakhyo/yolov8-crowdhuman)
(`yolov8n_best.pt`, renamed here to `yolov8n_crowdhuman.pt`), a real GitHub
Release asset (plain HTTPS GET, not Google Drive/Baidu/an unofficial mirror),
single class `"person"`, YOLOv8n architecture trained on the public
CrowdHuman dataset via the standard Ultralytics training pipeline.

**Not** auto-downloaded (unlike DM-Count/`yolov8n.pt`, which have one
canonical publisher-hosted release asset each): fetch it explicitly with
```bash
curl -sL -o weights/yolov8n_crowdhuman.pt \
  https://github.com/yakhyo/yolov8-crowdhuman/releases/download/weights/yolov8n_best.pt
```
If it's absent, `PersonDetector` automatically falls back to the
COCO-pretrained `yolov8n.pt` (auto-downloaded from Ultralytics' own official
GitHub Releases, `ultralytics/assets`) -- fewer boxes in dense/occluded
crowds, but detection still works rather than becoming unavailable. If
*neither* file can be loaded (fully offline, first run), the backend reports
visible-person detection as unavailable for that job/feed (`visible_count:
null`, `detections: []`) -- estimated-count/density analysis from DM-Count is
unaffected either way. Override the weights location with
`CROWD_MONITORING_YOLO_WEIGHTS` if you want to point at your own copy.

**Runtime requirement**: needs the `ultralytics` package (`pip install
ultralytics`, already in `requirements.txt`). It declares a dependency on
plain `opencv-python`, which can shadow the `opencv-python-headless` build
this project otherwise uses -- see the note at the top of `requirements.txt`
if `cv2` starts behaving oddly after installing/upgrading it.

The `MobileNetSSD_deploy.prototxt` / `MobileNetSSD_deploy.caffemodel` files
that may still be sitting in this directory from the original detector are
no longer read by anything in this codebase and can be deleted.
