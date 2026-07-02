# PENGWIN 2026 Task 2 (PENGWIN-Interact) — training & inference

Interactive **instance** segmentation of pelvic/femur fracture fragments. A CT + a JSON of
per-fragment clicks → an integer label map (Sacrum 1–50, Left Hip 51–100, Right Hip 101–150,
Femur 151–200). Built on this nnU-Net fork's click-conditioned trainer.

## Pipeline

The fragment model is **binary and click-conditioned**: the positive click marks the target
fragment, all other fragments' clicks are negatives → one forward pass per fragment. An
optional anatomy model (5-class) is used only to mask predictions, copy single-fragment
bones, and hole-fill. Per-fragment binary masks are merged into the PENGWIN label ranges.

## 1. Convert the data

```bash
python nnunetv2/dataset_conversion/Dataset456_458_PENGWIN.py \
    --src /path/to/Pengwin/Extracted \
    --frag-otf          # Dataset458: CT + contiguous instance seg + clicksTr  (ablations B & C)
    # --frag-precomp     # Dataset457: baseline-style 3-channel binary           (ablation A)
    # --anatomy          # Dataset456: 5-class anatomy + 4 click heatmaps        (optional)
```
CT files are symlinked by default (`--copy-ct` to copy). `--strategy` fixes one click strategy
(default: random per case). PENGWIN JSON points are `(z,y,x)`; the anatomy is read from each
point's `name`.

## 2. Train a fragment model — pick one ablation

```bash
# Ablation B — on-the-fly clicks, ResEnc points net (the trials ClickGen approach)
nnUNetv2_plan_and_preprocess -d 458 -pl nnUNetPlannerResEncL -c 3d_fullres
nnUNetv2_train 458 3d_fullres 0 -tr trialsTrainerPengwinFrag -p nnUNetResEncUNetLPlans
#   ...trialsTrainerPengwinFragTversky for the Focal-Tversky loss variant

# Ablation A — precomputed fg/bg channels, stock trainer
nnUNetv2_plan_and_preprocess -d 457 -c 3d_fullres
nnUNetv2_train 457 3d_fullres 0

# Ablation C — warm-start from the released nnInteractive v1.0 checkpoint (recommended)
nnUNetv2_move_plans_between_datasets -s 225 -t 458 \
    -sp nnUNetResEncUNetLPlans_noResampling -tp nnUNetResEncUNetLPlans_noResampling
nnUNetv2_preprocess -d 458 -plans_name nnUNetResEncUNetLPlans_noResampling -c 3d_fullres_ps192
nnUNetv2_train 458 3d_fullres_ps192 0 -tr trialsTrainerPengwinFragNNI \
    -p nnUNetResEncUNetLPlans_noResampling \
    -pretrained_weights ~/.nninteractive/models/nnInteractive_v1.0/fold_0/checkpoint_final.pth
```
Ablation C keeps nnInteractive's full 8-channel input (only the positive/negative *point*
channels are populated) so the pretrained prompt-reading stem and binary head transfer
exactly (956/956 params load). Requires the vendored `no_resampling_hack` resampler (added to
`nnunetv2/preprocessing/resampling/default_resampling.py`).

## 3. Inference (local)

```bash
PENGWIN_CLICK_LAYOUT=nninteractive \        # or "pair" for ablations A/B
PENGWIN_FRAG_MODEL=/path/to/nnUNet_results/Dataset458.../trialsTrainerPengwinFragNNI__... \
python -m pengwin_inference.inference \
    --ct case/image.mha --clicks case/peripelvic-fragment-clicks.json --output pred.mha
```

Env config: `PENGWIN_FRAG_MODEL`, `PENGWIN_FRAG_FOLD`, `PENGWIN_CLICK_LAYOUT`
(`pair`|`nninteractive`), `PENGWIN_USE_ANATOMY` (+`PENGWIN_ANAT_MODEL`), `PENGWIN_ROUTING`
(`on`|`off`, the pelvic/femur spacing rule — an ablation lever; baseline does not route),
`PENGWIN_TTA`.

## 4. Grand Challenge container

```bash
docker build -f pengwin_inference/Dockerfile -t pengwin-task2 .
```
Reads `/input/images/peripelvic-fracture-ct/*.mha` + `/input/peripelvic-fragment-clicks.json`,
loads the model from `/opt/ml/model/fragment` (and `/opt/ml/model/anatomy` if
`PENGWIN_USE_ANATOMY=1`), writes `/output/images/peripelvic-fracture-ct-segmentation/*.mha`.
Runs with `--network none`; scratch in `/tmp`.

## Layout

| file | role |
|------|------|
| `nnunetv2/training/dataloading/pengwin_clicks.py` | click parsing, 4-strategy simulation, anatomy↔range mapping, Gaussian rendering |
| `nnunetv2/training/dataloading/data_loader_clicks.py::nnUNetDataLoaderPengwinFrag` | on-the-fly per-fragment click dataloader (`pair`/`nninteractive` layouts) |
| `nnunetv2/training/nnUNetTrainer/trialsTrainer.py::trialsTrainerPengwinFrag*` | fragment trainers (B, Tversky, C) |
| `nnunetv2/dataset_conversion/Dataset456_458_PENGWIN.py` | PENGWIN → nnU-Net datasets |
| `pengwin_inference/{routing,predict,postprocess,inference}.py` | container: route → predict-per-fragment → assemble |
