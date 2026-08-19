# PENGWIN 2026 Task 2 (PENGWIN-Interact) inference container.
# Grand Challenge runs this with `--network none`; the model tarball is extracted to
# /opt/ml/model and the CT + clicks are mounted read-only under /input.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    nnUNet_raw=/tmp/nnUNet_raw \
    nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    nnUNet_results=/tmp/nnUNet_results \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /opt/app

# nnU-Net (this fork) + inference deps
COPY pyproject.toml /opt/app/pyproject.toml
COPY nnunetv2 /opt/app/nnunetv2
COPY pengwin_inference /opt/app/pengwin_inference
# cc3d is provided by connected-components-3d (already an nnU-Net dep); edt is imported by
# nnunetv2/.../dataloading/utils.py at module load, so it must be present at runtime.
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir connected-components-3d edt

# Non-root user (Grand Challenge requirement). /tmp scratch dirs are created at runtime by
# the entrypoint, because GC mounts /tmp fresh (a build-time mkdir would be shadowed).
RUN useradd -m -u 1000 user && chmod +x /opt/app/pengwin_inference/entrypoint.sh \
    && chown -R user:user /opt/app
USER user

# Grand Challenge phase limits: 10 min/case, <=16 GB DRAM, NVIDIA T4 (16 GB VRAM) or CPU.
# Configured for the nnInteractive-warm-started fragment model (ablation C), full-volume
# inference (ROI off) since that gave the best offline metrics and fit the memory budget.
#   PENGWIN_CLICK_LAYOUT=nninteractive  -> ablation C (set to "pair" for A/B models)
#   PENGWIN_ROI_MULT=0                  -> full-volume sliding window (ROI off)
#   PENGWIN_CHECKPOINT=checkpoint_best.pth
#   anatomy OFF, TTA OFF, CPU threads capped for the RAM budget
# NOTE: full-volume + ablation C runs at native spacing (large volumes). VERIFY it stays under
# 10 min/case ON A T4 (slower than a dev GPU). If it times out, set PENGWIN_ROI_MULT=1.5 and
# rebuild (a small metric hit for a big speedup).
ENV PENGWIN_FRAG_MODEL=/opt/ml/model/fragment \
    PENGWIN_CLICK_LAYOUT=nninteractive \
    PENGWIN_REFINE=2 \
    PENGWIN_REFINE_MAX_PASSES=60 \
    PENGWIN_ROI_MULT=1.5 \
    PENGWIN_CHECKPOINT=checkpoint_best.pth \
    PENGWIN_ASSEMBLY=argmax \
    PENGWIN_USE_ANATOMY=0 \
    PENGWIN_ROUTING=off \
    PENGWIN_TTA=0 \
    OMP_NUM_THREADS=4 \
    nnUNet_def_n_proc=4
# SUBMISSION = nnInteractive warm-start (ablation C) + 1 refinement pass. The model tarball's
# fragment/ MUST be the 8-channel NNI checkpoint (trialsTrainerPengwinFragNNI*), NOT the 9-ch
# strategy model. refine=1 doubles forward passes/fragment -- verified under 10 min/case on T4.
# TTA (mirroring) improves metrics but is ~8x the inference cost on 3D. ROI MUST stay on
# (PENGWIN_ROI_MULT=1.5) and you MUST verify the worst-case (most-fragment) case finishes
# < 10 min on a T4 -- a timeout is scored as a failed case. If it's over budget: drop TTA
# back to 0, or lower ROI to 1.0, or reduce fragments' per-pass cost.

ENTRYPOINT ["/opt/app/pengwin_inference/entrypoint.sh"]
