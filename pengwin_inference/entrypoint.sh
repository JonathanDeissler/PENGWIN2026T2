#!/bin/bash
# Grand Challenge mounts /tmp fresh at runtime, so create the scratch dirs here (not at build).
set -e
mkdir -p "${nnUNet_raw:-/tmp/nnUNet_raw}" \
         "${nnUNet_preprocessed:-/tmp/nnUNet_preprocessed}" \
         "${nnUNet_results:-/tmp/nnUNet_results}" \
         "${MPLCONFIGDIR:-/tmp/mpl}"
exec python -m pengwin_inference.inference "$@"
