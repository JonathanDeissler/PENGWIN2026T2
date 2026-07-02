# PENGWIN 2026 Task 2 (PENGWIN-Interact) inference container.
# Grand Challenge runs this with `--network none`; the model tarball is extracted to
# /opt/ml/model and the CT + clicks are mounted read-only under /input.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    nnUNet_raw=/tmp/nnUNet_raw \
    nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    nnUNet_results=/tmp/nnUNet_results

WORKDIR /opt/app

# nnU-Net (this fork) + inference deps
COPY pyproject.toml /opt/app/pyproject.toml
COPY nnunetv2 /opt/app/nnunetv2
COPY pengwin_inference /opt/app/pengwin_inference
RUN pip install --no-cache-dir -e . && pip install --no-cache-dir cc3d edt

# Non-root user (Grand Challenge requirement)
RUN useradd -m -u 1000 user && mkdir -p /tmp/nnUNet_raw /tmp/nnUNet_preprocessed /tmp/nnUNet_results \
    && chown -R user:user /opt/app /tmp/nnUNet_raw /tmp/nnUNet_preprocessed /tmp/nnUNet_results
USER user

# Defaults: nnInteractive warm-started fragment model, no anatomy model, no routing.
ENV PENGWIN_FRAG_MODEL=/opt/ml/model/fragment \
    PENGWIN_CLICK_LAYOUT=nninteractive \
    PENGWIN_USE_ANATOMY=0 \
    PENGWIN_ROUTING=off \
    PENGWIN_TTA=0

ENTRYPOINT ["python", "-m", "pengwin_inference.inference"]
