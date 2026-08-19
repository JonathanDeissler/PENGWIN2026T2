#!/usr/bin/env bash
# Build the PENGWIN 2026 Task 2 (PENGWIN-Interact) submission container.
#
#   ./build.sh [IMAGE_TAG]        # default tag: pengwin-task2:latest
#
# The build context is the repository root: the Dockerfile COPYs this nnU-Net fork
# (`nnunetv2/`), the inference package (`pengwin_inference/`), and `pyproject.toml`.
# Model weights are NOT baked into the image -- they are mounted at /opt/ml/model at
# runtime (see README.md "Model weights" and "Run on a case").
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${1:-pengwin-task2:latest}"

echo "Building ${IMAGE} ..."
docker build -f pengwin_inference/Dockerfile -t "${IMAGE}" .
echo
echo "Built ${IMAGE}"
echo "Next: download the model weights, extract to a folder with a fragment/ subdir,"
echo "      and run  ./pengwin_inference/test_local.sh  (see README.md)."
