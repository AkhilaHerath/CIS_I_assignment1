"""
The calibrated kinematic model: ImprovedFwdKins / ImprovedInvKins.
"""

from typing import Dict, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from data_types.nominal_types import vct3, Rot, Frame
from uncertainty_networks.se3 import log_so3

from ..robot import Robot
from .workflow import CalibrationData


class ImprovedKinematics(Robot):
    """
    Polynomial calibration with measured home offsets and a fixed base-roll gauge.
    Pose corrections use [rotation vector; translation], not SE(3) twist coordinates.
    Parameter 6 is fixed to zero during fitting. Zero parameters reproduce nominal
    kinematics only when joint_offsets is zero.
    Polynomial distortion is an approximation; validate on unseen poses. No
    simulator truth or simulated motion is used during fitting.

    Original parameter-layout guidance (Exp below means the chart described above):

    A Robot whose fwd_kins / inv_kins account for the calibrated error
    model. You choose the parameterization; the handout's error model is
        theta*_k = theta_k + delta_theta_k,sys(theta_k)   (all joints; ||.|| <= delta_sys, slope <= omega_sys)
        d*_k = R(phi_k) d_k                                (k = 2..6, ||phi_k|| <= phi_max; d*_1 = d_1)
        F*_we = F_we dF_ee                                 (||alpha_ee|| <= 0.01, ||epsilon_ee|| <= 0.5)
    plus the constant base frame F_X relating the robot base to the
    tracker (see workflow.estimate_base_frame). HomeRobot() gives you
    delta_theta_sys at home directly (workflow.measure_home_offsets); pass
    it as joint_offsets. Every parameter you include must be identifiable
    from your data -- look at the singular values of the Jacobian.

    A parameter vector that covers this model, for example:
        params[0:6]     xi_X          twist of the base frame about F_X_init, F_X = F_X_init * Exp(xi_X)
        params[6:16]    (t1, t2) x 5  tilt of d_2, d_3, n_4, n_5, n_6, two components each in the
                                      plane perpendicular to the nominal axis
        params[16:22]   xi_we         twist of the tool frame, F_we = F_WE * Exp(xi_we)
        params[22:]     c_k,j         per-joint distortion polynomial coefficients
                                      delta_theta_k,sys = offset_k + sum_j c_k,j (theta_k / scale_k)^j
    with every entry zero meaning "the nominal robot at F_X_init".

    The nominal Robot's attributes (AXES, F_CW, F_WE, limits, timing,
    wrist_solutions) are inherited and may be reused.
    """

    def __init__(self, robot: Robot, F_X_init: Frame, joint_offsets=None, n_poly: int = 2):
        super().__init__(robot.name + "_improved", robot.F_rob, robot.theta_min, robot.theta_max,
                         robot.t_settle, robot.theta_dot_max, robot.theta_ddot_max)
        self.F_X_init = F_X_init  # starting estimate of the base frame (from the nominal fit)
        self.joint_offsets = np.zeros(6) if joint_offsets is None else np.asarray(joint_offsets, dtype=np.float64).reshape(6)
        self.n_poly = int(n_poly)      # e.g. the degree of a per-joint distortion polynomial
        if self.n_poly < 1:
            raise ValueError("n_poly must be at least 1")
        self.joint_offsets = self.joint_offsets.copy()
        self.scales = np.maximum(np.maximum(abs(self.theta_min), abs(self.theta_max)), 1.0)
        self.nominal_axes = np.array([a.vec.ravel() for a in robot.AXES], dtype=float)
        self.tangents = []
        for axis in self.nominal_axes[1:]:
            axis = axis / np.linalg.norm(axis)
            e = np.eye(3)[np.argmin(abs(axis))]
            u = e - axis * (axis @ e)
            u /= np.linalg.norm(u)
            self.tangents.append(np.column_stack((u, np.cross(axis, u))))
        self.params = self.initial_params()
        # Gauge: fix rotation about d1 by fixing d2's out-of-xy tilt.
        # For the supplied Cartesian axes this is parameter 6.
        self.active = np.delete(np.arange(len(self.params)), 6)
        self.rotation_scale = 1e-3  # rad: target-based weighting, not a noise estimate
        self.position_scale = 0.2   # mm

    # ------------------------------------------------------------------ parameterization

    def initial_params(self) -> np.ndarray:
        """
        TODO: implement.

        output: the parameter vector that makes this model identical to the
                nominal kinematics with base frame F_X_init -- with the
                layout in the class docstring that is simply
                np.zeros(6 + 10 + 6 + 6 * self.n_poly)
        """
        return np.zeros(22 + 6 * self.n_poly)

    def base_frame(self, params: Optional[np.ndarray] = None) -> Frame:
        """
        TODO: implement.

        output: F_X (robot base relative to the tracker) for params (default:
                self.params) -- e.g. F_X_init * Exp(params[0:6]), see
                uFrame.perturb / uncertainty_networks.se3.exp_se3
        """
        p = self.params if params is None else np.asarray(params, dtype=float)
        return self.F_X_init * self._correction(p[:6])

    # ------------------------------------------------------------------ kinematics

    def fwd_kins(self, theta, params: Optional[np.ndarray] = None) -> Frame:
        """
        ImprovedFwdKins(theta).

        TODO: implement.

        input: theta - 6 encoder joint values; params - defaults to self.params
        output: F_ee relative to the robot base frame, with the calibrated
                error model (joint offsets + distortion, tilted axes, tool
                frame offset) applied
        """
        p = self.params if params is None else np.asarray(params, dtype=float)
        theta = np.asarray(theta, dtype=float).reshape(6)
        powers = (theta / self.scales)[:, None] ** np.arange(1, self.n_poly + 1)
        corrected = theta + self.joint_offsets + np.sum(
            p[22:].reshape(6, self.n_poly) * powers, axis=1)
        axes, _ = self._axes(p)
        translation = corrected[:3] @ axes[:3]
        R = np.eye(3)
        for k in range(3, 6):
            R = R @ Rotation.from_rotvec(axes[k] * corrected[k]).as_matrix()
        return (Frame(Rot(), vct3(*translation)) * self.F_CW
                * Frame(self._rotation(R), vct3()) * self.F_WE
                * self._correction(p[16:22]))

    def predict_in_tracker(self, theta, params: Optional[np.ndarray] = None) -> Frame:
        """The end-effector pose in the tracker frame: base_frame * fwd_kins."""
        return self.base_frame(params) * self.fwd_kins(theta, params)

    def inv_kins(self, F: Frame, seed=None, tol: float = 1e-10, max_iter: int = 50) -> np.ndarray:
        """
        ImprovedInvKins(F) -- provided.

        There is no closed form once the axes are tilted, so this iterates
        on your fwd_kins: both nominal wrist branches (Robot.inv_kins'
        closed form) are refined with damped Gauss-Newton / Levenberg-
        Marquardt steps on the pose error [alpha; epsilon] of
        fwd_kins(theta)^-1 F, and the in-limit solution nearest `seed`
        (default: the current command) is returned. Works as soon as your
        fwd_kins does.

        input: F - desired F_ee relative to the robot base frame
        output: theta with fwd_kins(theta) ~= F; raises ValueError if no
                branch satisfies the joint limits
        """
        seed = self.theta_cmd if seed is None else np.asarray(seed, dtype=np.float64).reshape(6)
        p_w = F.p - F.R * self.F_WE.p - self.F_CW.p   # nominal wrist center -> seeds for the linear joints
        step_max = np.array([50.0] * 3 + [0.3] * 3)
        h = np.array([1e-3] * 3 + [1e-6] * 3)

        def pose_delta(theta):
            dF = self.fwd_kins(theta).inv() * F
            return np.concatenate([log_so3(dF.R.matrix), dF.p.vec.ravel()])

        candidates = []
        for wrist in self.wrist_solutions(F.R):
            theta = np.concatenate([[p_w.x, p_w.y, p_w.z], wrist])
            lam, delta = 1e-4, pose_delta(theta)
            for _ in range(max_iter):
                cost = float(delta @ delta)
                if np.sqrt(cost) < tol:
                    break
                J = np.zeros((6, 6))
                for k in range(6):
                    th = theta.copy()
                    th[k] += h[k]
                    J[:, k] = (pose_delta(th) - delta) / h[k]
                while True:  # Levenberg-Marquardt: shrink the step until the cost decreases
                    JtJ = J.T @ J
                    step = -np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-12), J.T @ delta)
                    trial = theta + np.clip(step, -step_max, step_max)
                    delta_trial = pose_delta(trial)
                    if float(delta_trial @ delta_trial) < cost or lam > 1e6:
                        theta, delta, lam = trial, delta_trial, max(lam / 10.0, 1e-12)
                        break
                    lam *= 10.0
            theta[3] = (theta[3] + np.pi) % (2 * np.pi) - np.pi
            theta[5] = (theta[5] + np.pi) % (2 * np.pi) - np.pi
            if self.within_limits(theta):
                candidates.append(theta)
        if not candidates:
            raise ValueError(f"inv_kins: no joint solution within limits for {F!r}")
        return min(candidates, key=lambda th: float(np.linalg.norm(th - seed)))

    # ------------------------------------------------------------------ fitting

    def residuals(self, params: np.ndarray, data: CalibrationData) -> np.ndarray:
        """
        TODO: implement.

        input: params - candidate parameter vector; data - CalibrationData
        output: flat residual vector comparing predict_in_tracker(theta_j,
                params) with data.frames[j] for every sample (choose how to
                weigh rotation vs translation)
        """
        if len(data.thetas) != len(data.frames) or not len(data.thetas):
            raise ValueError("Expected nonempty, paired joint vectors and frames")
        result = []
        for theta, observed in zip(data.thetas, data.frames):
            delta = observed.inv() * self.predict_in_tracker(theta, params)
            result.extend(log_so3(delta.R.matrix) / self.rotation_scale)
            result.extend(delta.p.vec.ravel() / self.position_scale)
        return np.asarray(result)

    def fit(self, data: CalibrationData, **kwargs):
        """
        TODO: implement.

        Non-linear least squares over the parameters (scipy.optimize is
        allowed). Must set self.params. Return whatever diagnostics you want
        to report (final cost, iterations, ...).
        """
        start = self.params.copy()
        start[6] = 0.0

        def unpack(x):
            p = start.copy()
            p[self.active] = x
            return p

        kwargs.setdefault("x_scale", "jac")
        kwargs.setdefault("max_nfev", 500)
        result = least_squares(
            lambda x: self.residuals(unpack(x), data),
            start[self.active], **kwargs)
        if not np.all(np.isfinite(result.x)):
            raise ValueError("Fit returned nonfinite parameters")
        self.params = unpack(result.x)
        self.fit_result = result
        return dict(success=bool(result.success), message=result.message,
                    cost=float(result.cost), nfev=result.nfev,
                    optimality=float(result.optimality),
                    n_params=len(self.active))

    def jacobian_rank_report(self, data: CalibrationData, params: Optional[np.ndarray] = None) -> dict:
        """
        TODO: implement.

        Numerically differentiate residuals w.r.t. params and report the
        singular values (and which parameter directions are poorly
        determined). Use it to justify your parameterization and your
        choice of calibration poses in the report.
        """
        p = (self.params if params is None else np.asarray(params, dtype=float)).copy()
        # Scale columns by characteristic physical parameter changes.
        scales = np.ones(len(p))
        scales[:3] = scales[6:19] = 0.01
        scales[3:6] = scales[19:22] = 1.0
        scales[22:] = np.repeat(
            np.array([1., 1., 1., .01, .01, .01]), self.n_poly)
        J = np.empty((len(self.residuals(p, data)), len(self.active)))
        h = 1e-4
        for col, k in enumerate(self.active):
            plus, minus = p.copy(), p.copy()
            plus[k] += h * scales[k]
            minus[k] -= h * scales[k]
            J[:, col] = (self.residuals(plus, data) -
                         self.residuals(minus, data)) / (2 * h)
        _, singular, Vt = np.linalg.svd(J, full_matrices=True)
        threshold = singular[0] * 1e-6 if singular.size else 0.
        rank = int(np.sum(singular > threshold))
        return dict(rank=rank, n_params=len(self.active),
                    singular_values=singular, threshold=threshold,
                    active_indices=self.active.copy(),
                    parameter_scales=scales[self.active],
                    weak_directions_scaled=Vt[rank:],
                    gauge="parameter 6 fixed: d2 remains in base xy plane")

    # ------------------------------------------------------------------ reporting

    def parameters_report(self) -> Dict[str, np.ndarray]:
        """
        TODO: implement.

        The handout (method 1 (e)) asks your program to report the updated
        kinematic parameters in physical terms, and your debugging runs to
        compare them with ground truth (rob.ground_truth_parameters()).
        Report at least: the axis tilt vectors / unit axes, the angles
        between the axes, the per-joint offsets and scale factors, and the
        tool-frame offset. Note which of these are only defined up to the
        base-frame gauge (and so are better compared as angles between axes).
        """
        axes, tilts = self._axes(self.params)
        angles = np.degrees(np.arccos(np.clip(axes @ axes.T, -1., 1.)))
        coefficients = self.params[22:].reshape(6, self.n_poly)
        return dict(unit_axes=axes, axis_tilt_vectors=tilts,
                    axis_angles_deg=angles,
                    joint_offset=self.joint_offsets.copy(),
                    joint_scale=1. + coefficients[:, 0] / self.scales,
                    distortion_coefficients=coefficients.copy(),
                    polynomial_scales=self.scales.copy(),
                    tool_frame_offset=self._correction(self.params[16:22]),
                    base_frame=self.base_frame(),
                    gauge="d1 fixed; d2 constrained to base xy plane")

    @staticmethod
    def _rotation(matrix):
        # Convert through the known Rot(axis-name, angle) interface.
        # Intrinsic XYZ gives Rx(a) Ry(b) Rz(c).
        a, b, c = Rotation.from_matrix(matrix).as_euler("XYZ")
        return Rot("x", a) * Rot("y", b) * Rot("z", c)

    @classmethod
    def _correction(cls, xi):
        # Rotation-vector + independent translation chart, NOT an SE(3) twist.
        return Frame(cls._rotation(Rotation.from_rotvec(xi[:3]).as_matrix()),
                     vct3(*xi[3:6]))

    def _axes(self, params):
        axes = self.nominal_axes.copy()
        tilts = np.zeros((6, 3))
        for k in range(1, 6):
            tilts[k] = self.tangents[k - 1] @ params[6 + 2*(k-1):8 + 2*(k-1)]
            axes[k] = Rotation.from_rotvec(tilts[k]).apply(axes[k])
        return axes, tilts
