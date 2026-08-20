# PENGWIN 2026 — Task 2 (PENGWIN-Interact) submission

Interactive **instance** segmentation of pelvic/femur fracture fragments: a CT volume + a JSON of
per-fragment clicks → an integer label map with PENGWIN ranges (Sacrum 1–50, Left Hip 51–100,
Right Hip 101–150, Femur 151–200).

This repository is a research fork of **nnU-Net v2**; the submission code lives in
[`pengwin_inference/`](pengwin_inference/) (Grand Challenge container) and
[`nnunetv2/training/`](nnunetv2/training/) (trainers + click data loading). This file covers only
how to **build and run the final Task 2 container**. For the training recipe and data conversion
see [`pengwin_inference/README.md`](pengwin_inference/README.md).

## Method

Per fragment, the model does **click-conditioned binary segmentation** — the fragment's own click
is the positive prompt, all other fragments' clicks are negatives — so it runs **one forward pass
per fragment**. The network is warm-started from the released **nnInteractive v1.0** checkpoint
(8-channel input: CT + 7 interaction channels, only the point channels used) and fine-tuned to be
**refinement-aware**: at inference each fragment is predicted, its own prediction is fed back into
nnInteractive's `initSeg` channel, and it is re-predicted (iterative refinement). Per-fragment
foreground probabilities are merged into the final instance map by **arg-max competition**, then
restored to the original image geometry. An adaptive cap bounds total forward passes per case so
many-fragment cases stay within the runtime budget. No TTA, no external data beyond nnInteractive.

## Build

```bash
./build.sh                      # -> pengwin-task2:latest
./build.sh pengwin-task2:v1     # or a custom tag
```

Requires Docker with the NVIDIA runtime for GPU inference. The image is self-contained (installs
this fork editable + `connected-components-3d` and `edt`); it needs **no network at runtime**
(Grand Challenge runs it with `--network none`).

## Model weights

Weights are **not** in this repo. [**Download**](https://hub.dkfz.de/s/FTMkykaCL2rPzZ5)  and extract them so a `fragment/` folder sits at the
root of the model directory:

```
<MODEL_DIR>/
└── fragment/
    ├── dataset.json
    ├── plans.json
    ├── dataset_fingerprint.json
    └── fold_0/
        └── checkpoint_best.pth      # 8-channel nnInteractive-warm-started refine model
```

- [**Download link:**](https://hub.dkfz.de/s/FTMkykaCL2rPzZ5) 
- On Grand Challenge the model tarball is uploaded separately and auto-extracted to
  `/opt/ml/model`, i.e. the checkpoint ends up at `/opt/ml/model/fragment/fold_0/checkpoint_best.pth`.
- Make the files world-readable (`chmod -R a+rX <MODEL_DIR>`): the container runs as uid 1000.

## Run on a case (same interface as Grand Challenge)

Grand Challenge mounts `/input` (read-only), `/output` (write), and `/opt/ml/model` (read-only),
and runs the image with `--network none`. To reproduce that locally:

```bash
CASE=001
WORK=$(mktemp -d)
mkdir -p "$WORK/input/images/peripelvic-fracture-ct" "$WORK/output"
cp image.mha "$WORK/input/images/peripelvic-fracture-ct/$CASE.mha"
cp clicks.json "$WORK/input/peripelvic-fragment-clicks.json"
chmod -R 777 "$WORK"

docker run --rm --network none --gpus all --memory=16g \
  -v "$WORK/input":/input:ro \
  -v "$WORK/output":/output \
  -v "<MODEL_DIR>":/opt/ml/model:ro \
  pengwin-task2:latest

# result: $WORK/output/images/peripelvic-fracture-ct-segmentation/$CASE.mha  (uint, labels 0–200)
```

A convenience wrapper that also checks the output labels and reports wall time vs. the 10-min
budget is provided:

```bash
IMAGE=pengwin-task2:latest MODEL=<MODEL_DIR> DATA=<Extracted_root> \
  bash pengwin_inference/test_local.sh <CASE_ID> center_of_mass
```

### I/O contract

| path | role |
|------|------|
| `/input/images/peripelvic-fracture-ct/<id>.mha` | input CT |
| `/input/peripelvic-fragment-clicks.json` | per-fragment clicks (anatomy encoded in point names) |
| `/opt/ml/model/fragment/…` | model weights (extracted from the uploaded tarball) |
| `/output/images/peripelvic-fracture-ct-segmentation/<id>.mha` | output instance label map (0–200) |

## Configuration (baked into the image; override with `-e`)

| env var | submission value | meaning |
|---|---|---|
| `PENGWIN_CLICK_LAYOUT` | `nninteractive` | 8-channel nnInteractive input layout |
| `PENGWIN_CHECKPOINT` | `checkpoint_best.pth` | checkpoint loaded from `fragment/fold_0/` |
| `PENGWIN_REFINE` | `2` | iterative refinement passes (initSeg feedback) |
| `PENGWIN_REFINE_MAX_PASSES` | `60` | cap on total forward passes/case (auto-lowers refine on many-fragment cases) |
| `PENGWIN_ROI_MULT` | `1.5` | predict a 1.5×patch window around each click (speed) |
| `PENGWIN_ASSEMBLY` | `argmax` | overlap resolution by foreground probability |
| `PENGWIN_TTA` | `0` | test-time mirroring off |
| `PENGWIN_USE_ANATOMY` / `PENGWIN_ROUTING` | `0` / `off` | fragment-only, no anatomy model / routing |

## Reproduce training (optional)

See [`pengwin_inference/README.md`](pengwin_inference/README.md) for data conversion and the full
training recipe. The submitted model is trained with:

```bash
# Final submitted model: batch size 4, trained on ALL cases (fold "all"). The resulting
# checkpoint is shipped under fragment/fold_0/ so the container (PENGWIN_FRAG_FOLD=0) loads it.
nnUNetv2_train 458 3d_fullres_ps192_bs4 all -tr trialsTrainerPengwinFragNNIRefine \
    -p nnUNetResEncUNetLPlans_noResampling \
    -pretrained_weights <nnInteractive_v1.0/fold_0/checkpoint_final.pth>
```

(`trialsTrainerPengwinFragNNIRefine` in
[`nnunetv2/training/nnUNetTrainer/trialsTrainer.py`](nnunetv2/training/nnUNetTrainer/trialsTrainer.py);
click simulation in
[`nnunetv2/training/dataloading/`](nnunetv2/training/dataloading/).)
