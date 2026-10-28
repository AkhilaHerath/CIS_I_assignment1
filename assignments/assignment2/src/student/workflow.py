"""
Data-collection workflow: which robot poses to visit, how to read them, and
how to register the robot base to the tracker.
"""
from .. import simulation
from . import registration

from dataclasses import dataclass, field
from typing import List

import numpy as np

from data_types.nominal_types import vct3, Frame

from ..robot import Robot
from assignments.assignment1.src.tracker import Tracker
from ..simulation import PlacedBody


@dataclass
class CalibrationData:
    """One calibration data set.

    thetas[j] is the commanded (encoder) joint vector and frames[j] the
    calibration body's observed pose relative to the tracker, F_ee,j (this
    handout's tracker has no jiggle, so the tracker frame is a fixed
    reference). t_collect is the simulated time the collection took.
    """
    thetas: List[np.ndarray] = field(default_factory=list)
    frames: List[Frame] = field(default_factory=list)
    t_collect: float = 0.0

    def __len__(self):
        return len(self.thetas)


def measure_home_offsets(rob: Robot, n_homings: int = 5) -> np.ndarray:
    """
    TODO: implement.

    HomeRobot() drives the robot to theta* = 0 and returns the encoder
    values there, i.e. -delta_theta_sys(home) - delta_theta_rnd. Repeat it
    and average to estimate the systematic joint offsets at home (after the
    first homing the move is tiny, so each repeat costs about t_settle).

    input:
        rob - the robot (rob.home_robot())
        n_homings - how many homings to average
    output: estimated delta_theta_sys at home, np.ndarray of 6
    """

    if n_homings < 1:
        raise ValueError("n_homings must be at least 1.")

    readings = []

    for _ in range(n_homings):
        readings.append(rob.home_robot().copy())

    return -np.mean(readings, axis=0)

    raise NotImplementedError("TODO: implement measure_home_offsets")


# this should contain a desired accuracy which will affect the number of calibration poses
def plan_calibration_poses(rob: Robot, trk: Tracker, mbee_local_positions: List[vct3], desired_accuracy: float, **kwargs) -> List[np.ndarray]:
    """
    TODO: implement -- this is a design task.

    Choose the joint vectors the robot will visit during calibration to reach the desired accuracy. Every
    parameter of your kinematic model must be excited by the data (axis
    tilts need travel along each axis; the wrist axes and F_we need wrist
    rotations; the joint distortion needs several stops along each joint),
    every pose must keep all calibration-body markers inside the tracker's
    viewing volume (Tracker.is_visible, using the nominal rob.F_rob and
    rob.fwd_kins -- they are only tens of mm off), and the order of the
    poses determines the motion time. The wrist moves at 0.1 rad/s: a pi/2
    turn costs ~16 s.

    The handout's method 1: decide how closely spaced the samples along an
    axis must be from the error bounds (delta_sys^trk, omega_sys^trk,
    delta_sys^rob, omega_sys^rob), then sweep each axis in turn from the
    home position. Joints 1-3 are decoupled from 4-6.

    input:
        rob - the Robot (nominal kinematics + limits)
        trk - the Tracker (for visibility checks)
        mbee_local_positions - the calibration body's marker layout a*_k
        desired_accuracy - target RMS orientation error in rad (the limiting
            metric); use it to size the plan -- how many distinct poses and
            how often each is read (a pose listed twice in a row is
            re-commanded and re-read, costing only t_settle + delta_t_samp)
        kwargs - whatever your design needs (grid sizes, margins, ...)
    output: ordered List of joint vectors (np.ndarray of 6)
    """

    if not np.isfinite(desired_accuracy) or desired_accuracy <= 0:
        raise ValueError("desired_accuracy must be positive and finite.")

    # Approximate random orientation-noise budget.
    # Replace 75 mm with a layout-based effective spread if available.
    sigma_marker = 0.1       # mm
    effective_spread = 75.0  # mm
    angular_noise_budget = 0.5 * desired_accuracy

    n_readings = max(
        1,
        int(np.ceil(
            (sigma_marker /
             (effective_spread * angular_noise_budget)) ** 2
        ))
    )

    # Initial sampling density, scaled with requested accuracy.
    # These spacings are design starting points, not an accuracy guarantee.
    density = max(1.0, 1e-3 / desired_accuracy)
    n_linear = int(np.ceil(6 * density)) + 1
    n_wrist = int(np.ceil(6 * density)) + 1

    home = np.zeros(6)
    center = np.array([300., 0., 300., 0., 0., 0.])
    poses = []

    def visible(theta):
        F_ee_world = rob.F_rob * rob.fwd_kins(theta)
        markers_world = [
            F_ee_world * a for a in mbee_local_positions
        ]
        return trk.is_visible(markers_world)

    def add_pose(theta):
        theta = np.asarray(theta, dtype=float).reshape(6)

        if not visible(theta):
            raise ValueError(
                f"Planned pose is not nominally visible: {theta}"
            )

        # Consecutive duplicates cause move + sample on each repetition.
        poses.extend(theta.copy() for _ in range(n_readings))

    # Linear-axis sweeps with the other joints at home.
    linear_ranges = [
        (0.0, 600.0),
        (-300.0, 300.0),
        (0.0, 600.0),
    ]

    for joint, (low, high) in enumerate(linear_ranges):
        add_pose(home)

        for value in np.linspace(low, high, n_linear):
            theta = home.copy()
            theta[joint] = value
            add_pose(theta)

        add_pose(home)

    # Wrist sweeps at the center of the translational work volume.
    # Sweep monotonically to limit slow wrist travel.
    wrist_ranges = [
        (-0.75 * np.pi, 0.75 * np.pi),
        (-0.75 * np.pi, 0.75 * np.pi),
        (-np.pi, np.pi),
    ]

    for joint, (low, high) in enumerate(wrist_ranges, start=3):
        add_pose(center)

        for value in np.linspace(low, high, n_wrist):
            theta = center.copy()
            theta[joint] = value
            add_pose(theta)

        add_pose(center)
        add_pose(home)

    # Additional wrist configurations to help constrain coupled geometry.
    for angles in [
        (0.8, 0.8, 0.8),
        (0.8, -0.8, -0.8),
        (-0.8, -0.8, 0.8),
        (-0.8, 0.8, -0.8),
    ]:
        theta = center.copy()
        theta[3:] = angles
        add_pose(theta)

    add_pose(home)
    return poses
    raise NotImplementedError("TODO: implement plan_calibration_poses")


def collect_calibration_data(rob: Robot, trk: Tracker, mbee_placed: PlacedBody,
                             thetas: List[np.ndarray], repeats: int = 1) -> CalibrationData:
    """
    TODO: implement.

    Visit each joint vector with rob.move_joints, read the calibration body
    with sample_marker_bodies and store its frame relative to the tracker.
    Decide what to do when the body is not visible (None), and whether/how
    to use repeats (re-sampling costs 0.03 s; re-commanding the same pose
    costs t_settle and re-rolls the joint repeatability noise).

    input:
        rob, trk - robot and tracker
        mbee_placed - the robot calibration body (attached)
        thetas - joint vectors to visit, in order
        repeats - readings per pose
    output: CalibrationData
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")

    data = CalibrationData()
    t_start = simulation.clock.elapsed

    for theta in thetas:
        theta = np.asarray(theta, dtype=float).reshape(6)

        # Move once per pose.
        rob.move_joints(theta)

        # Take repeated tracker readings without moving again.
        for _ in range(repeats):
            simulation.sample_marker_bodies(trk)

            F_obs = simulation.most_recent_observed_frame(
                mbee_placed.body
            )

            if F_obs is None:
                raise ValueError(
                    f"Calibration body was not observed at pose {theta}."
                )

            data.thetas.append(theta.copy())
            data.frames.append(F_obs)

    data.t_collect = simulation.clock.elapsed - t_start

    return data
    raise NotImplementedError("TODO: implement collect_calibration_data")


def estimate_base_frame(data: CalibrationData, kins: Robot) -> Frame:
    """
    TODO: implement.

    With the calibration body's frame being exactly F_ee, every sample
    satisfies  F_ee,j = F_X * kins.fwd_kins(theta_j)  for one constant
    unknown F_X (the robot base expressed in the tracker's frame). Estimate
    F_X from all samples (your average_frames may help).

    input:
        data - CalibrationData
        kins - the kinematics to use (the nominal Robot at first; later your
            ImprovedKinematics)
    output: F_X
    """

    if len(data) == 0:
        raise ValueError("At least one observation is required.")

    if len(data.thetas) != len(data.frames):
        raise ValueError("Each joint vector must have a matching frame.")

    base_estimates = []

    for theta, F_obs in zip(data.thetas, data.frames):
        F_nom = kins.fwd_kins(theta)
        F_X_j = F_obs * F_nom.inv()
        base_estimates.append(F_X_j)

    return registration.average_frames(base_estimates)
    raise NotImplementedError("TODO: implement estimate_base_frame")