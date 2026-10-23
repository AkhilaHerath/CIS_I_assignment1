"""
Pose-error metrics used to report calibration accuracy.
"""

from typing import Dict, Sequence, Tuple

import numpy as np

from data_types.nominal_types import Frame


def frame_error(F_a: Frame, F_b: Frame) -> Tuple[float, float]:
    """
    D TODO: implement.

    The mismatch between two frames is defined as
    dF = F_a^-1 F_b ~= Frame(I + sk(alpha), epsilon); report its size.

    input:
        F_a - reference Frame (e.g. ground truth)
        F_b - Frame to compare (e.g. your prediction)
    output:
        (angle_rad, dist_mm) - ||alpha|| (the rotation angle of dF.R) and
            ||epsilon|| (the norm of dF.p)
    """
    dF = F_a.inv() * F_b

    p = np.array(dF.p.vec).reshape(3)
    dist_mm = np.linalg.norm(p)

    R = dF.R.matrix
    c = (np.trace(R) - 1) / 2       #arccos(theta) = ((trace of rot)-1)/2
    c = np.clip(c, -1.0, 1.0)       # keeps cos in the range of -1,1 
    angle_rad = np.arccos(c)

    return float(angle_rad), float(dist_mm)


def summarize_errors(errors: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """
    D TODO: implement.

    input:
        errors - list of (angle_rad, dist_mm) pairs from frame_error
    output:
        dict with at least the keys
            n, rms_angle, max_angle, rms_pos, max_pos
    """
    angles = []
    dists = []
    for angle, dist in errors:
        angles.append(angle)
        dists.append(dist)
    angles = np.array(angles)
    dists = np.array(dists)

    summary = {}
    summary["n"] = len(errors)
    summary["rms_angles"] = float(np.sqrt(np.mean(angles ** 2)))
    summary["max_angle"] = float(np.max(angles))
    summary["rms_pos"] = float(np.sqrt(np.mean(dists ** 2)))
    summary["max_pos"] = float(np.max(dists))
    return summary