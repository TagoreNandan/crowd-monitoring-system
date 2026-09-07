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

## Person detector (optional, backend visible-count overlay only)

| File | Purpose |
|---|---|
| `MobileNetSSD_deploy.prototxt` | MobileNet-SSD network definition (small text file). |
| `MobileNetSSD_deploy.caffemodel` | MobileNet-SSD pretrained weights (~23 MB). |

These two files back `crowdmon.detection.PersonDetector`, used only by the
FastAPI backend's video job runner to draw real, visible person bounding
boxes (`visible_count`) as a *separate* signal alongside DM-Count's own
`estimated_count` -- DM-Count handles dense/aerial crowd-density estimation
(and never sees these files), while MobileNet-SSD only detects individual
people that are large/close enough for a classic single-shot detector to
resolve. It is not a substitute for DM-Count and won't produce meaningful
counts on dense aerial crowd shots -- that's expected, not a bug.

**Not** downloaded automatically, unlike the DM-Count/CSRNet checkpoints
above: the classic MobileNet-SSD Caffe export isn't published from one
canonical, versioned release, so this project doesn't guess a mirror URL to
fetch a binary weight file from.

Source: both files come from the original
[chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)
repo, pinned to commit
[`9740699`](https://github.com/chuanqi305/MobileNet-SSD/commit/97406996b1eee2d40eb0a00ae567cf41e23369f9),
fetched with a plain HTTPS GET (`raw.githubusercontent.com`) -- no Google
Drive / Baidu Disk / OneDrive client needed:

| File | URL (commit-pinned) | SHA1 |
|---|---|---|
| `MobileNetSSD_deploy.prototxt` | `.../97406996.../deploy.prototxt` | `50cf80235a8fcccc641bf9f8efc803edbf21c615` |
| `MobileNetSSD_deploy.caffemodel` | `.../97406996.../mobilenet_iter_73000.caffemodel` | `19e3ec38842f3e68b02c07a1c24424a1e9db57e9` |

This exact prototxt/caffemodel pairing and both SHA1s are independently
verified against OpenCV's own official test-data manifest
(`opencv_extra/testdata/dnn/download_models.py`, entry `MobileNet-SSD
(caffemodel)`/`MobileNet-SSD (prototxt)`), and reproduce OpenCV's own
`Reproducibility_MobileNet_SSD` unit-test ground truth
(`opencv_extra/testdata/dnn/mobilenet_ssd_caffe_out.npy` on
`opencv_extra/testdata/dnn/street.png`) to floating-point precision. An
earlier, differently-sized `MobileNetSSD_deploy.prototxt` variant that was
briefly checked in here (also from the chuanqi305 repo, but a different,
119-layer network definition) loaded without error against this caffemodel
but silently produced zero valid detections -- same repo, same file names,
but an incompatible pairing. Don't swap in a prototxt from elsewhere without
re-verifying against the SHA1s above.

If replacing these files, get both from the same pinned commit (or point
`CROWDMON_SSD_PROTOTXT` / `CROWDMON_SSD_CAFFEMODEL` at your own verified
copies) rather than mixing sources. If they're absent, the backend simply
reports visible-person detection as unavailable for that job
(`visible_count: null`) -- estimated-count analysis is unaffected either
way.

**Runtime requirement**: `readNetFromCaffe` needs `opencv-python-headless`
4.x -- 5.0.0 (the version a loose `>=4.8` pin resolves to) dropped the
Caffe/Darknet/Torch importers outright, so `PersonDetector` can't even load
the model on it. `opencv-python-headless==4.9.0.80` through `4.14.0.94` were
all spot-checked against the `Reproducibility_MobileNet_SSD`-style ground
truth above and reproduce it correctly. `requirements.txt` pins the specific
version this project validated against
(`opencv-python-headless==4.11.0.86`); if you need to move that pin, re-run
the ground-truth check above first rather than assuming any 4.x build works.
