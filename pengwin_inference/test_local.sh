#!/bin/bash
# Local Grand-Challenge-equivalent test of the PENGWIN Task 2 container.
# Usage:  bash pengwin_inference/test_local.sh [CASE_ID] [STRATEGY]
#   e.g.  bash pengwin_inference/test_local.sh 193 center_of_mass
# Env overrides: IMAGE, DATA (Extracted root), MODEL (staged model dir), PYBIN, GPUS
set -e

CASE=${1:-193}
STRATEGY=${2:-center_of_mass}
IMAGE=${IMAGE:-pengwin-task2}
DATA=${DATA:-/home/j683r/local-work-temporary/ingest/Pengwin/Extracted}
MODEL=${MODEL:-/home/j683r/pengwin_submission/model}
PYBIN=${PYBIN:-/home/j683r/miniconda3/envs/trials25/bin/python}
GPUS=${GPUS:---gpus all}          # set GPUS="" to test the No-GPU (CPU) path

WORK=$(mktemp -d /tmp/pengwin_test.XXXXXX)
mkdir -p "$WORK/input/images/peripelvic-fracture-ct" "$WORK/output"
cp "$DATA/$CASE/image.mha" "$WORK/input/images/peripelvic-fracture-ct/$CASE.mha"
cp "$DATA/PENGWIN26_task2_train_clicks/$STRATEGY/$CASE/peripelvic-fragment-clicks.json" \
   "$WORK/input/peripelvic-fragment-clicks.json"
chmod -R 777 "$WORK"              # container runs as uid 1000; let it write /output
chmod -R a+rX "$MODEL" 2>/dev/null || true   # container uid 1000 must be able to read the model

echo "== running $IMAGE on case $CASE ($STRATEGY) =="
START=$(date +%s)
docker run --rm --network none $GPUS --memory=16g --memory-swap=16g \
  -v "$WORK/input":/input:ro \
  -v "$WORK/output":/output \
  -v "$MODEL":/opt/ml/model:ro \
  "$IMAGE"
ELAPSED=$(( $(date +%s) - START ))
echo "== elapsed: ${ELAPSED}s  (Grand Challenge limit: 600s/case) =="

OUT="$WORK/output/images/peripelvic-fracture-ct-segmentation"
echo "== output =="; ls -la "$OUT"
"$PYBIN" - "$OUT/$CASE.mha" "$DATA/$CASE/label.mha" <<'PY'
import sys, numpy as np, SimpleITK as sitk
pred = sitk.ReadImage(sys.argv[1]); a = sitk.GetArrayFromImage(pred)
gt = sitk.GetArrayFromImage(sitk.ReadImage(sys.argv[2]))
print("pred dtype:", a.dtype, "shape:", a.shape, "== GT shape:", a.shape == gt.shape)
print("pred labels:", np.unique(a).tolist())
print("GT   labels:", np.unique(gt).tolist())
oor = [int(l) for l in np.unique(a) if l > 0 and not (1 <= l <= 200)]
print("out-of-range labels:", oor or "none")
PY
echo "== workdir: $WORK  (rm -rf when done) =="
