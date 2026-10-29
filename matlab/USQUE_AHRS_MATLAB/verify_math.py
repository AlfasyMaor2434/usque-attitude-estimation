"""
verify_math.py -- checks of usque.py (and therefore of USQUE_AHRS.m, which verify_port.py shows
is numerically identical) against the two reference papers, using scipy's Rotation as an
independent implementation.

  Sola, "Quaternion kinematics for the error-state Kalman filter" (Hamilton convention)
  Crassidis & Markley, "Unscented Filtering for Spacecraft Attitude Estimation" (USQUE)
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import main_usque as mu   # noqa: E402
import usque              # noqa: E402

rng = np.random.default_rng(1)
U = usque.USQUE
sp = lambda q: Rot.from_quat([q[1], q[2], q[3], q[0]])        # Hamilton wxyz -> scipy (xyzw, active, body->world)
rq = lambda: (lambda v: v / np.linalg.norm(v))(rng.normal(size=4))
ok = lambda name, err, tol: print(f"  {'PASS' if err < tol else 'FAIL'}  {name:78s} max error {err:.2e}")

print("Sola -- quaternion algebra and kinematics (Hamilton convention)")
i, j, k = np.array([0, 1., 0, 0]), np.array([0, 0, 1., 0]), np.array([0, 0, 0, 1.])
ok("ij = k, jk = i, ki = j   (Sola eq. 2b: Hamilton, right-handed)",
   max(np.abs(U._quat_mult(i, j) - k).max(), np.abs(U._quat_mult(j, k) - i).max(), np.abs(U._quat_mult(k, i) - j).max()), 1e-15)
e = 0
for _ in range(200):
    p, q = rq(), rq()
    e = max(e, np.abs(sp(U._quat_mult(p, q)).as_matrix() - sp(p).as_matrix() @ sp(q).as_matrix()).max())
ok("R(p (x) q) = R(p) R(q)   (Sola: composition of rotations)", e, 1e-12)

f = U(dict(mu.FILTER_PARAMS, use_mag_bias_state=False), initial_accel=np.array([0, 0, -9.81]), initial_mag=np.array([1., 0, 0.5]))
e = 0
for _ in range(200):
    q = rq(); ref = rng.normal(size=3); ref /= np.linalg.norm(ref)
    ha, hm = U._h_from_q(q, ref)
    R = sp(q).as_matrix()                                       # body -> nav
    e = max(e, np.abs(hm - R.T @ ref).max(), np.abs(ha - R.T @ np.array([0, 0, -1.0])).max())
ok("measurement model: h_mag = R(q)^T ref,  h_acc = R(q)^T (0,0,-1)   (Sola: x_G = q (x) x_L (x) q*)", e, 1e-12)

# kinematics: qdot = 1/2 q (x) omega_body  ->  q_{n+1} = q_n (x) q{omega dt}   (Sola eq. 211 and its zeroth-order integration)
p = dict(mu.FILTER_PARAMS, use_mag_bias_state=False, P0_att=np.ones(3) * 1e-16, P0_bias=1e-20, gyro_noise_var=1e-20, bias_rw_var=1e-24)
f = U(p, initial_accel=np.array([0, 0, -9.81]), initial_mag=np.array([1., 0, 0.5]))
f.q_nom = rq()                                               # any start attitude
Rref = sp(f.q_nom); e = 0
for n in range(2000):
    w = np.array([0.7 * np.sin(0.01 * n), -1.1, 0.4 + 0.3 * np.cos(0.02 * n)]); dt = 0.004
    f.predict(w, dt)
    Rref = Rref * Rot.from_rotvec(w * dt)                        # right-multiply: body-frame rate
    e = max(e, np.degrees((Rref.inv() * sp(f.q_nom)).magnitude()))
ok("predict(): q <- q (x) Exp(omega dt), 2000 steps (error in deg)", e, 1e-6)

print("\nCrassidis & Markley -- generalized Rodrigues parameters (a = 1, f = 4) and the error definition")
e = e2 = e3 = 0
for _ in range(500):
    th = rng.normal(size=3); th *= rng.uniform(0, 3.9) / np.linalg.norm(th)    # |dp| < 4, i.e. rotations below 180 deg
    dq = f._error_vec_to_quat(th)
    e = max(e, abs(np.linalg.norm(dq) - 1))                                         # eq. 21 gives a unit quaternion
    e2 = max(e2, np.abs(f._quat_to_error_vec(dq) - th).max())                      # eq. 20 and 21 are inverses
    a_, f_ = 1.0, 4.0; n2 = th @ th
    q4 = (-a_ * n2 + f_ * np.sqrt(f_ ** 2 + (1 - a_ ** 2) * n2)) / (f_ ** 2 + n2)   # eq. 21a written out again
    e3 = max(e3, abs(dq[0] - q4), np.abs(dq[1:] - (a_ + q4) / f_ * th).max())       # eq. 21b
ok("eq. 21a/21b: error vector -> unit quaternion", max(e, e3), 1e-13)
ok("eq. 20 inverts eq. 21 (round trip, rotations below 180 deg)", e2, 1e-10)
e = 0
for _ in range(200):
    th = rng.normal(size=3) * 1e-3
    e = max(e, np.abs(sp(f._error_vec_to_quat(th)).as_rotvec() - th).max() / 1e-3)
ok("f = 2(a+1): |dp| = rotation angle for small errors (relative error at 1e-3 rad)", e, 1e-6)

# error placement: sigma quaternion = dq (x) q_hat in the paper (attitude matrix A = nav->body, eq. 32b)
#                  = q_hat (x) dq here (Hamilton, body->nav).  A(q) = R(q)^T, so both are the same rotation.
A = lambda q: sp(q).as_matrix().T
e = 0
for _ in range(200):
    q = rq(); dq = f._error_vec_to_quat(rng.normal(size=3) * 0.3)
    e = max(e, np.abs(A(dq) @ A(q) - A(U._quat_mult(q, dq))).max())
ok("eq. 32b: A(dq) A(q)  ==  attitude of q (x) dq  (same body-frame error, other notation)", e, 1e-12)

print("\nCrassidis & Markley -- sigma points, weights, process noise")
L, lam = f.L, f.lambda_
print(f"  INFO  n = {L} states (paper: 6; the 7th is the magnetic heading-bias state), lambda = {lam:g} (paper: 1)")
ok("mean weights: W0 = lambda/(n+lambda), Wi = 1/(2(n+lambda))   (eq. 7, 9)",
   max(abs(f.Wm[0] - lam / (L + lam)), np.abs(f.Wm[1:] - 0.5 / (L + lam)).max(), abs(f.Wm.sum() - 1)), 1e-15)
print(f"  NOTE  covariance weight of the centre point: here {f.Wc[0]:.3f}, paper {lam / (L + lam):.3f} "
      f"(scaled-UT term 1 - alpha^2 + beta = 2 added; the centre point's deviation from the mean is ~0, so the effect is second order)")
P = np.diag(rng.uniform(1e-6, 1e-3, L)); S = f._get_sigma_points(P)
rec = sum((1.0 / (2 * (L + lam))) * np.outer(S[i], S[i]) for i in range(1, 2 * L + 1))
ok("sigma points: +-columns of sqrt((n+lambda) P) reproduce P   (eq. 5)", np.abs(rec - P).max() / P.max(), 1e-10)
sv2, su2, dt = 9e-5, 1e-10, 0.004
G = np.block([[(sv2 * dt + su2 * dt ** 3 / 3) * np.eye(3), -(su2 * dt ** 2 / 2) * np.eye(3)], [-(su2 * dt ** 2 / 2) * np.eye(3), su2 * dt * np.eye(3)]])
p = dict(mu.FILTER_PARAMS, use_mag_bias_state=False, gyro_noise_var=sv2, bias_rw_var=su2, P0_att=np.ones(3) * 1e-6, P0_bias=1e-6)     # above the 1e-9 eigenvalue floor of _enforce_pd
f2 = U(p, initial_accel=np.array([0, 0, -9.81]), initial_mag=np.array([1., 0, 0.5]))
P0 = f2.P.copy(); f2.predict(np.zeros(3), dt)
grow = (f2.P - P0)[:6, :6]
Phi = np.block([[np.eye(3), -dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
expect = Phi @ P0[:6, :6] @ Phi.T + G - P0[:6, :6]
ok("process noise added per step = G Q G^T of eq. 41 (incl. the attitude-bias cross term)", np.abs(grow - expect).max() / G[0, 0], 1e-6)
print("  NOTE  the paper splits this noise half before / half after the propagation (eq. 5a, 8, 42, 'trapezoid');"
      "\n        here the whole of eq. 41 is added after it. The two agree to first order in dt.")
