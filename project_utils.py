"""
quaternion_utils.py
-------------------
Foundational quaternion math for a Quaternion EKF (QEKF / error-state MEKF).

CONVENTION (read this — mixing conventions is the #1 source of bugs):
  * Hamilton quaternions, SCALAR-FIRST: q = [w, x, y, z]
  * A quaternion represents a rotation from body frame -> world frame.
  * Rotation composition: q_total = q_new (x) q_old   (left-multiply by increment)

NOTE ON DATASET CONVENTION:
  ROS (and many .bag / dataset files) store quaternions SCALAR-LAST: [x, y, z, w].
  When you load ground truth, reorder to [w, x, y, z] before using these functions.
  Helper `from_xyzw` / `to_xyzw` are provided for exactly that.

Run this file directly (python quaternion_utils.py) to execute the self-tests.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


# =========================================================
# Helper Functions (Quaternion Operations etc.)
# =========================================================

def normalize(q):
    """Return the unit quaternion. Guards against divide-by-zero AND
    against non-finite input (inf/NaN), which np.linalg.norm does NOT
    reliably catch: norm(q) == inf is not < 1e-12, so the old check let
    q/inf silently through -- which numpy evaluates to an all-zero
    vector, not an error. That zero vector then propagates for one or
    more calls (e.g. through several UKF sigma points) before crashing
    deep inside scipy's Rotation.from_quat with a "zero norm quaternions"
    message that points nowhere near the real cause. Raising here instead
    means the traceback lands on the actual q that broke, at the point
    it broke.
    """
    q = np.asarray(q, dtype=float)
    if not np.all(np.isfinite(q)):
        raise ValueError(f"normalize(): quaternion has non-finite components: {q}")
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Cannot normalize a zero quaternion.")
    return q / n

def skew(vec):
    return np.array([
        [0,-vec[2],vec[1]],
        [vec[2],0,-vec[0]],
        [-vec[1],vec[0],0]
    ])

def convert_q_order(q_xyzw):
    q_xyzw = np.array(q_xyzw)
    return np.array([q_xyzw[1],q_xyzw[2],q_xyzw[3],q_xyzw[0]])

def conjugate(q):
    """Quaternion conjugate. For a unit quaternion this is also the inverse."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def inverse(q):
    """Quaternion inverse = conjugate / norm^2."""
    q = np.asarray(q, dtype=float)
    return conjugate(q) / np.dot(q, q)


def multiply(q1, q2):
    """
    Hamilton product q1 (x) q2, scalar-first.
    Order matters: this rotates first by q2, then by q1.
    """
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def delta_quaternion(delta_theta):
    return np.concatenate([[1.0], 0.5 * delta_theta])


def ensure_quaternion_continuity(q, q_prev):
    if np.dot(q, q_prev) < 0:
        return -q
    return q


# ----------------------------------------------------------------------
# Conversions
# ----------------------------------------------------------------------
def to_rotation_matrix(q):
    """Unit quaternion -> 3x3 rotation matrix (body -> world)."""
    w, x, y, z = normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def from_rotation_matrix(R):
    """3x3 rotation matrix -> unit quaternion (scalar-first), Shepperd's method."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return normalize(np.array([w, x, y, z]))


def to_euler(q):
    """
    Unit quaternion -> (roll, pitch, yaw) in radians, ZYX convention.
    Use this to compute per-axis RMSE in your metrics module.
    """
    q_xyzw = [q[1], q[2], q[3], q[0]]
    r = R.from_quat(q_xyzw)
    # NOTE: this used to pass degrees=True, contradicting the docstring
    # above. Every caller (main.py, metrics.py) already wraps this in
    # np.degrees(...) itself, expecting radians here -- so the bug made
    # every downstream roll/pitch/yaw value get converted degrees->degrees
    # a second time (multiplied by ~57.3), which axis_error_deg then wraps
    # into [-180,180] via wrap_deg(), turning it into effectively aliased
    # noise rather than a scaled-but-recoverable error. Returning true
    # radians here (matching the docstring and the already-correct
    # self-test below) fixes every caller with zero changes on their end.
    return r.as_euler('xyz', degrees=False)


# ----------------------------------------------------------------------
# Error-state mappings (the heart of the MEKF)
# ----------------------------------------------------------------------
def from_axis_angle(axis_angle):
    """
    Rotation vector (axis * angle, in radians) -> unit quaternion.
    This is the exponential map so(3) -> SO(3). Used to convert the
    integrated angular increment (omega * dt) into a quaternion.
    """
    theta = np.linalg.norm(axis_angle)
    if theta < 1e-8:
        # small-angle approximation, then normalize
        return normalize(np.array([1.0, *(0.5 * np.asarray(axis_angle))]))
    axis = np.asarray(axis_angle) / theta
    half = theta / 2.0
    return np.array([np.cos(half), *(np.sin(half) * axis)])


def to_axis_angle(q):
    """
    Unit quaternion -> rotation vector (log map SO(3) -> so(3)).
    Returns a 3-vector. Used to pull a small attitude error into the
    tangent space for the error-state correction.
    """
    q = normalize(q)
    w = np.clip(q[0], -1.0, 1.0)
    v = q[1:]
    nv = np.linalg.norm(v)
    if nv < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(nv, w)
    return (angle / nv) * v


# ----------------------------------------------------------------------
# Metric helper
# ----------------------------------------------------------------------
def angular_distance(q_est, q_true):
    """
    Geodesic angular error between two orientations, in radians.
    This is the 'qAE' metric: the single best scalar measure of
    orientation accuracy. Handles the q == -q double-cover correctly.
    """
    q_est = normalize(q_est)
    q_true = normalize(q_true)
    dot = np.abs(np.dot(q_est, q_true))      # |.| handles double cover
    dot = np.clip(dot, -1.0, 1.0)
    return 2.0 * np.arccos(dot)


# ----------------------------------------------------------------------
# Dataset convention helpers (ROS / .bag use [x, y, z, w])
# ----------------------------------------------------------------------
def from_xyzw(q_xyzw):
    """Convert scalar-LAST [x,y,z,w] (ROS) -> scalar-FIRST [w,x,y,z]."""
    x, y, z, w = q_xyzw
    return np.array([w, x, y, z])


def to_xyzw(q_wxyz):
    """Convert scalar-FIRST [w,x,y,z] -> scalar-LAST [x,y,z,w] (ROS)."""
    w, x, y, z = q_wxyz
    return np.array([x, y, z, w])


# ----------------------------------------------------------------------
# Self-tests — run `python quaternion_utils.py` to verify the math.
# ----------------------------------------------------------------------
def _run_self_tests():
    rng = np.random.default_rng(0)
    tol = 1e-9

    # identity
    qI = np.array([1.0, 0, 0, 0])
    assert np.allclose(multiply(qI, qI), qI), "identity multiply failed"

    # conjugate of a rotation undoes it
    q = normalize(rng.standard_normal(4))
    assert np.allclose(multiply(q, conjugate(q)), qI, atol=tol), "conjugate failed"

    # rotation matrix round-trip
    R = to_rotation_matrix(q)
    assert np.allclose(R @ R.T, np.eye(3), atol=tol), "R not orthonormal"
    q_back = from_rotation_matrix(R)
    # account for double cover (q and -q are the same rotation)
    assert angular_distance(q, q_back) < 1e-6, "matrix round-trip failed"

    # axis-angle exp/log round-trip
    rv = rng.standard_normal(3) * 0.3
    assert np.allclose(to_axis_angle(from_axis_angle(rv)), rv, atol=1e-6), \
        "exp/log round-trip failed"

    # angular distance of identical orientations is zero
    assert angular_distance(q, q) < tol, "self angular distance nonzero"

    # angular distance is symmetric and matches a known 90-degree rotation
    q90 = from_axis_angle([0, 0, np.pi / 2])   # 90 deg about z
    assert abs(angular_distance(qI, q90) - np.pi / 2) < 1e-6, "90deg test failed"

    # euler round-trip on a moderate orientation
    eul = to_euler(q90)
    assert abs(eul[2] - np.pi / 2) < 1e-6, "euler yaw failed"

    # ROS convention helpers
    assert np.allclose(from_xyzw(to_xyzw(q)), q), "xyzw round-trip failed"

    print("All quaternion self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()