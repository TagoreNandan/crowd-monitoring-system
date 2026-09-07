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
FastAPI backend's video job runner to draw real person bounding boxes
(`visible_count`) on top of DM-Count's separate `estimated_count` -- **not**
downloaded automatically, unlike the DM-Count/CSRNet checkpoints above.
Unlike those, the classic MobileNet-SSD Caffe export isn't published from one
canonical, versioned release, so this project doesn't guess a mirror URL to
fetch a binary weight file from. Get both files from the original
[chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)
project (or any mirror you trust) and place them here, or point
`CROWDMON_SSD_PROTOTXT` / `CROWDMON_SSD_CAFFEMODEL` at your own copies. If
they're absent, the backend simply reports visible-person detection as
unavailable for that job (`visible_count: null`) -- estimated-count analysis
is unaffected either way.
