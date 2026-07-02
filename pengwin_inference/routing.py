"""Pelvic vs. femur case routing (challenge-provided rule). Used as an ablation lever:
when enabled, restricts which anatomies are emitted; the baseline does not route."""
from __future__ import annotations

import SimpleITK as sitk


def classify_pelvic_femur(spacing_x, spacing_y, spacing_z, physical_x_mm, physical_z_mm) -> str:
    """Verbatim challenge rule (decision tree over spacing + physical field-of-view)."""
    if physical_x_mm <= 285.35:
        if spacing_x <= 0.71:
            return "pelvic"
        elif spacing_z <= 0.90:
            return "femur"
        else:
            return "pelvic" if spacing_y <= 0.91 else "femur"
    else:
        if spacing_z <= 0.68:
            return "pelvic" if physical_z_mm <= 193.55 else "femur"
        else:
            return "pelvic" if physical_z_mm <= 390.78 else "femur"


def classify_image(img: sitk.Image) -> str:
    """Apply :func:`classify_pelvic_femur` to a SimpleITK image (spacing is (x, y, z))."""
    sx, sy, sz = img.GetSpacing()
    dx, dy, dz = img.GetSize()
    return classify_pelvic_femur(sx, sy, sz, sx * dx, sz * dz)


# Anatomy ids that may appear in each case type (sacrum=1, left_hip=2, right_hip=3, femur=4).
ROUTING_ALLOWED_ANATOMIES = {
    "pelvic": {1, 2, 3},
    "femur": {4},
}
