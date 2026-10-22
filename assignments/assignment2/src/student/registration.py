"""
Rigid registration helpers: point-set registration, frame averaging and
marker-body refinement.
"""

from typing import List

import numpy as np

from data_types.nominal_types import vct3, Rot, Frame

from assignments.assignment1.src.tracker import Tracker
from ..simulation import PlacedBody, sample_marker_bodies, most_recent_observed_markers, update_marker_body



def register_points(a: List[vct3], b: List[vct3]) -> Frame:
    """
    TODO: implement (your own -- do not call the simulator's solver).

    Rigid point-set registration: find the least-squares Frame F such that
    F * a[i] ~= b[i] for every corresponding pair. Think about what happens
    with fewer than 3 points, with collinear points, and with a reflection.

    input:
        a - points in the source frame (e.g. a marker body's local layout)
        b - the same points observed in the target frame, same order
    output: Frame mapping a -> b
    """
    raise NotImplementedError("TODO: implement register_points")


def average_frames(frames: List[Frame]) -> Frame:
    """
    TODO: implement.

    Average of several noisy estimates of the same Frame. Rotations cannot
    simply be averaged element-wise -- the result must be a proper
    rotation matrix.

    input:
        frames - List[Frame], all estimates of the same transform
    output: the averaged Frame
    """
    raise NotImplementedError("TODO: implement average_frames")


def refine_marker_body(trk: Tracker, placed: PlacedBody, n_readings: int = 30) -> List[vct3]:
    """
    TODO: implement.

    "It is possible to improve your estimate of a_m,k through a calibration
    process involving multiple observations of the marker body using the
    tracking camera and then averaging them."

    Take n_readings SampleMarkerBodies readings, use
    most_recent_observed_markers(...) plus your register_points to bring
    each reading's b_m,k back into the body's own frame, average, and push
    the result into the body with update_marker_body(...). Note that each
    reading costs delta_t_samp of simulated time, and that the tracker
    jiggles between readings.

    input:
        trk - Tracker the body is tracked by
        placed - PlacedBody whose body's nominal marker positions should be refined
        n_readings - number of readings to average
    output: the new List[vct3] of marker positions (also stored on the body)
    """

    body = placed.body
    layout = []
    for a in body.nominal_marker_positions:
        layout.append(vct3(a))

    readings = []
    for i in range(n_readings):
        sample_marker_bodies(trk)       # single snapshot from tracker
        b = most_recent_observed_markers(body)      # marker position in tracker frame
        if b is None:           # skip if body was not visible
            continue
        F = register_points(layout, b)      # gets the transform from body frame to tracker frame
        F_inv = F.inv()

        # move each marker position back to the body frame
        this_reading = []
        for bk in b:
            a = F_inv * bk
            this_reading.append(np.array(a.vec).reshape(3))
        readings.append(this_reading)

    if len(readings) == 0:
        raise RuntimeError("the body was never visible")

    # average the readings and update the body
    averages = np.mean(readings, axis=0)
    new_layout = []
    for row in averages:
        new_layout.append(vct3(row[0], row[1], row[2]))

    update_marker_body(body, new_layout)
    return new_layout