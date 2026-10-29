"""
metrics.py
----------
Evaluation metrics for the Filters with Interactive Plots.
Contains plotting functions for BOTH Monte Carlo and Batch modes,
including MAE, Max Absolute Error, Separated NIS, Bias Convergence, and Covariance Tracking.
"""

import numpy as np
import csv
from scipy.stats import chi2
import project_utils as q

def mocap_boresight(q_est, quat_mocap, omega=None, dyn_thresh=0.05,
                    min_dyn_frac=0.15, label=""):
    """Task 4.4 -- solve for the constant boresight rotation between the
    filter's body frame and the mocap rigid-body frame, then report the
    residual attitude error after removing it. Accelerometer-independent
    (uses only the filter quaternion and the mocap pose).

    Model (body-side, the physical one for a marker-to-IMU mount offset):
        R_mocap(t) = R_filter(t) @ R_hat
    solved as the rotation mean of  R_filter(t).T @ R_mocap(t)  over
    DYNAMIC samples (|omega| > dyn_thresh rad/s -- a static-only fit is
    degenerate in yaw and just re-measures the t0 seed). Also reports the
    nav-side fit  R_mocap = R_navhat @ R_filter  for contrast.

    q_est, quat_mocap : (N,4) wxyz, ALIGNED (same timestamps).
    omega             : (N,3) body rate rad/s; if None, derived from q_est.
    Returns a dict; prints a one-line summary.
    """
    from scipy.spatial.transform import Rotation as _Rot
    qe = np.asarray(q_est, float); qm = np.asarray(quat_mocap, float)
    n = min(len(qe), len(qm))
    qe, qm = qe[:n], qm[:n]
    Re = _Rot.from_quat(qe[:, [1, 2, 3, 0]])
    Rm = _Rot.from_quat(qm[:, [1, 2, 3, 0]])

    if omega is not None and len(omega) >= n:
        w = np.linalg.norm(np.asarray(omega, float)[:n], axis=1)
    else:
        # body rate from consecutive filter quats
        rel = (Re[:-1].inv() * Re[1:]).as_rotvec()
        w = np.concatenate([[0.0], np.linalg.norm(rel, axis=1)]) * 1.0
        w = w / max(np.median(np.diff(np.arange(n))), 1e-9)  # ~per-sample; scale-free gate below
        dyn_thresh = np.percentile(w, 60)  # fall back to a relative gate

    dyn = w > dyn_thresh
    if dyn.mean() < min_dyn_frac:
        dyn = w > np.percentile(w, 100 * (1 - max(min_dyn_frac, 0.15)))
    Re_d, Rm_d = Re[dyn], Rm[dyn]

    def _rot_mean(Ra, Rb):
        M = np.zeros((3, 3))
        for A, B in zip(Ra.as_matrix(), Rb.as_matrix()):
            M += A.T @ B
        U, _, Vt = np.linalg.svd(M)
        D = np.diag([1.0, 1.0, np.linalg.det(U @ Vt)])
        return _Rot.from_matrix(U @ D @ Vt)

    R_hat = _rot_mean(Re_d, Rm_d)                       # body-side: Rm = Re @ R_hat
    res_body = (Re_d * R_hat).inv() * Rm_d
    rv_b = np.degrees(res_body.as_rotvec())            # (Nd,3) roll/pitch/yaw residual
    # nav-side contrast: Rm = R_navhat @ Re
    M = np.zeros((3, 3))
    for A, B in zip(Re_d.as_matrix(), Rm_d.as_matrix()):
        M += B @ A.T
    U, _, Vt = np.linalg.svd(M)
    R_navhat = _Rot.from_matrix(U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt)
    res_nav = R_navhat.inv() * (Rm_d * Re_d.inv())
    rv_n = np.degrees(res_nav.as_rotvec())

    bs_e = R_hat.as_euler('xyz', degrees=True)
    nav_e = R_navhat.as_euler('xyz', degrees=True)
    out = dict(
        label=label, n=n, n_dyn=int(dyn.sum()),
        boresight_rpy_deg=[float(x) for x in bs_e],
        resid_rms_deg=float(np.sqrt(np.mean(np.sum(rv_b ** 2, axis=1)))),
        resid_rms_rpy_deg=[float(np.sqrt(np.mean(rv_b[:, k] ** 2))) for k in range(3)],
        navside_rpy_deg=[float(x) for x in nav_e],
        navside_resid_rms_deg=float(np.sqrt(np.mean(np.sum(rv_n ** 2, axis=1)))),
    )
    print(f"  mocap boresight{(' ['+label+']') if label else ''}: "
          f"R_hat(rpy) = [{bs_e[0]:+.2f}, {bs_e[1]:+.2f}, {bs_e[2]:+.2f}] deg  "
          f"(n_dyn={out['n_dyn']}/{n})")
    print(f"    post-boresight residual RMS: {out['resid_rms_deg']:.3f} deg  "
          f"(roll/pitch/yaw {out['resid_rms_rpy_deg'][0]:.3f}/"
          f"{out['resid_rms_rpy_deg'][1]:.3f}/{out['resid_rms_rpy_deg'][2]:.3f})  "
          f"| nav-side fit residual {out['navside_resid_rms_deg']:.3f} deg")
    return out


# ------------------------------------------------------------------ helpers
def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))

def mae(x):
    return float(np.mean(np.abs(np.asarray(x))))

def max_err(x):
    return float(np.max(np.abs(np.asarray(x))))

def wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0

def _canonical(dq):
    return dq if dq[0] >= 0 else -dq

def _truncate_to_shortest(list_of_arrays):
    min_len = min(len(arr) for arr in list_of_arrays)
    return np.array([arr[:min_len] for arr in list_of_arrays])

# ------------------------------------------------------------------ accuracy
def qae_deg(q_est, q_true):
    return np.degrees(
        np.array([q.angular_distance(e, g) for e, g in zip(q_est, q_true)])
    )

def axis_error_deg(q_est, q_true):
    e_est  = np.degrees(np.array([q.to_euler(e) for e in q_est]))
    e_true = np.degrees(np.array([q.to_euler(g) for g in q_true]))
    return wrap_deg(e_est - e_true)

# ------------------------------------------------------------------ consistency helpers
def _right_matrix(quat):
    qw, qx, qy, qz = quat
    return np.array([
        [qw, -qx, -qy, -qz],
        [qx,  qw,  qz, -qy],
        [qy, -qz,  qw,  qx],
        [qz,  qy, -qx,  qw]
    ])

def _error_and_cov(q_est, q_true, P_in):
    q_err = q.multiply(q.inverse(q_est), q_true)
    if q_err[0] < 0:
        q_err = -q_err

    norm_vec = np.linalg.norm(q_err[1:4])
    if norm_vec > 1e-8:
        angle = 2.0 * np.arccos(np.clip(q_err[0], -1.0, 1.0))
        theta = (q_err[1:4] / norm_vec) * angle
    else:
        theta = np.zeros(3)

    dim = P_in.shape[0]

    if dim == 3:
        P_theta = P_in
    elif dim == 6:
        P_theta = P_in[0:3, 0:3]
    elif dim == 15:
        # Error-state layout used by Qf_es_ekf_ukf.py is
        # [delta_theta(0:3), delta_vel(3:6), delta_pos(6:9),
        #  delta_b_accel(9:12), delta_b_gyro(12:15)] -- attitude is the
        # FIRST block, not the third. The previous P_in[6:9, 6:9] was
        # silently reading the POSITION covariance into the attitude
        # NEES, which would report a plausible-looking but meaningless
        # number (nothing raises an error for a 3x3 slice of the wrong
        # block).
        P_theta = P_in[0:3, 0:3]
    else:
        P_qq = P_in[0:4, 0:4]
        qw, qx, qy, qz = q_est
        Jp = np.array([
            [-qx, qw, qz, -qy],
            [-qy, -qz, qw, qx],
            [-qz, qy, -qx, qw]
        ]) * 2.0
        P_theta = Jp @ P_qq @ Jp.T

    return theta, P_theta

def nees_series(q_est, q_true, P_hist_in, valid_mask=None):
    import numpy as np
    N = len(q_est)
    nees_vals = np.zeros(N)

    if valid_mask is None:
        valid_mask = np.ones(N, dtype=bool)

    for i in range(N):
        if not valid_mask[i]:
            continue
        theta, P_theta = _error_and_cov(q_est[i], q_true[i], P_hist_in[i])
        try:
            P_inv = np.linalg.inv(P_theta)
        except np.linalg.LinAlgError:
            P_inv = np.linalg.pinv(P_theta)

        nees_vals[i] = theta.T @ P_inv @ theta

    return nees_vals, valid_mask, None

def nis_series(innov, S_hist, valid):
    idx = np.where(valid)[0]
    nis = np.array(
        [innov[i] @ np.linalg.solve(S_hist[i], innov[i]) for i in idx]
    )
    return nis, idx

def calc_single_nis(innov, S):
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    return float(innov.T @ S_inv @ innov)

# ------------------------------------------------------------------ REPORTS
def print_evaluation_report(qae_runs, axis_err_runs, nees_runs, nis_runs, N, dof_att=3,
                            dof_meas=6, accel_ref=None, nis_split=None):
    """accel_ref: optional dict with batch-mean roll/pitch metrics scored vs
    the quasi-static accel-tilt reference (and yaw vs the RTK heading),
    printed as a second column next to the vs-Wahba-GT numbers. Keys:
    roll_rmse pitch_rmse yaw_rmse roll_mae pitch_mae yaw_mae roll_max
    pitch_max yaw_max roll_nees pitch_nees yaw_nees n_windows.

    nis_split: optional list of (label, dof, per_run_arrays) tuples. Each
    channel's marginal NIS is pooled across flights and printed with its
    own ideal value (= dof) and chi-square 95% band, decomposing the
    pooled dof=6 GPS ANIS into position / velocity / yaw / mag / gravity
    contributions."""
    mean_qae = np.mean([np.mean(r) for r in qae_runs])
    mean_nees = np.mean([np.mean(r) for r in nees_runs])

    valid_nis_runs = [r for r in nis_runs if len(r) > 0]
    if valid_nis_runs:
        mean_nis = np.mean([np.mean(r) for r in valid_nis_runs])
    else:
        mean_nis = 0.0

    mean_roll_rmse = np.mean([rmse(err[:, 0]) for err in axis_err_runs])
    mean_pitch_rmse = np.mean([rmse(err[:, 1]) for err in axis_err_runs])
    mean_yaw_rmse = np.mean([rmse(err[:, 2]) for err in axis_err_runs])

    mean_roll_mae = np.mean([mae(err[:, 0]) for err in axis_err_runs])
    mean_pitch_mae = np.mean([mae(err[:, 1]) for err in axis_err_runs])
    mean_yaw_mae = np.mean([mae(err[:, 2]) for err in axis_err_runs])

    max_roll = np.max([max_err(err[:, 0]) for err in axis_err_runs])
    max_pitch = np.max([max_err(err[:, 1]) for err in axis_err_runs])
    max_yaw = np.max([max_err(err[:, 2]) for err in axis_err_runs])

    lo_a, hi_a = chi2.ppf(0.025, N * dof_att) / N,  chi2.ppf(0.975, N * dof_att) / N
    lo_m, hi_m = chi2.ppf(0.025, N * dof_meas) / N, chi2.ppf(0.975, N * dof_meas) / N

    a = accel_ref or {}

    def col2(key, unit="deg", note=""):
        """right-hand 'vs accel-tilt' cell for a metric key, or blank."""
        if not a or key not in a or a[key] is None:
            return ""
        return f"     |   {a[key]:7.3f} {unit}{note}"

    nw = a.get("n_windows")
    yw_note = "     |   (= Wahba yaw; gravity carries no heading)"
    print(f"\n============= EVALUATION REPORT (N={N}) =============")
    if a:
        print(f"{'':<18}   vs Wahba-GT (full flight)     vs accel-tilt "
              f"(roll/pitch only, {nw} still-window(s); yaw & its qAE part use Wahba yaw)")
    print("--- ACCURACY (RMSE - Root Mean Square Error) ---")
    print(f"Mean qAE         : {mean_qae:7.3f} deg" + col2("qae"))
    print(f"Roll  Mean RMSE  : {mean_roll_rmse:7.3f} deg" + col2("roll_rmse"))
    print(f"Pitch Mean RMSE  : {mean_pitch_rmse:7.3f} deg" + col2("pitch_rmse"))
    print(f"Yaw   Mean RMSE  : {mean_yaw_rmse:7.3f} deg" + (yw_note if a else ""))

    print("\n--- ACCURACY (MAE - Mean Absolute Error) ---")
    print(f"Roll  Mean MAE   : {mean_roll_mae:7.3f} deg" + col2("roll_mae"))
    print(f"Pitch Mean MAE   : {mean_pitch_mae:7.3f} deg" + col2("pitch_mae"))
    print(f"Yaw   Mean MAE   : {mean_yaw_mae:7.3f} deg" + (yw_note if a else ""))

    print("\n--- SAFETY (Max Absolute Error) ---")
    print(f"Roll  Max Error  : {max_roll:7.3f} deg" + col2("roll_max"))
    print(f"Pitch Max Error  : {max_pitch:7.3f} deg" + col2("pitch_max"))
    print(f"Yaw   Max Error  : {max_yaw:7.3f} deg" + (yw_note if a else ""))

    print("\n--- CONSISTENCY (ANEES / ANIS) ---")
    print(f"ANEES mean       : {mean_nees:9.2f}   (ideal ~ {dof_att})" +
          (f"     |   Roll {a.get('roll_nees', float('nan')):.1f} / "
           f"Pitch {a.get('pitch_nees', float('nan')):.1f}  (per-axis, dof=1, vs accel-tilt)" if a else ""))
    print(f"  95% bounds     : [{lo_a:.2f}, {hi_a:.2f}]")
    if valid_nis_runs:
        print(f"ANIS  mean       : {mean_nis:7.3f}   (ideal ~ {dof_meas})   "
              f"(innovation-based, no accel-tilt variant; = pooled GPS pos+vel)")
        print(f"  95% bounds     : [{lo_m:.2f}, {hi_m:.2f}]")

    if nis_split:
        print("\n--- ANIS BY UPDATE TYPE (per-channel marginal NIS) ---")
        print(f"{'channel':<14} {'mean':>8} {'median':>8}  {'ideal':>5}   "
              f"{'95% band':>16}   flights  samples")
        for label, dof, per_run in nis_split:
            runs = [np.asarray(r, dtype=float) for r in (per_run or [])
                    if r is not None and len(r) > 0]
            if not runs:
                continue
            n_flights = len(runs)
            n_samp = int(sum(len(r) for r in runs))
            ch_mean = float(np.mean([r.mean() for r in runs]))
            ch_med = float(np.mean([np.median(r) for r in runs]))
            lo = chi2.ppf(0.025, n_samp * dof) / n_samp
            hi = chi2.ppf(0.975, n_samp * dof) / n_samp
            flag = "" if lo <= ch_mean <= hi else "  <-- outside band"
            print(f"{label:<14} {ch_mean:8.3f} {ch_med:8.3f}  {dof:5d}   "
                  f"[{lo:6.2f}, {hi:6.2f}]   {n_flights:5d}   {n_samp:7d}{flag}")
        print("  (mean/median are averaged over flights; band is the pooled-sample "
              "chi-square interval)")
    print("=====================================================")

# ------------------------------------------------------------------ PLOTS
def plot_bias_convergence(t, bg_hist, ba_hist, title_suffix=""):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    bg_hist_deg = np.degrees(bg_hist)
    ax[0].plot(t, bg_hist_deg[:, 0], label="Gyro Bias X", color='r')
    ax[0].plot(t, bg_hist_deg[:, 1], label="Gyro Bias Y", color='g')
    ax[0].plot(t, bg_hist_deg[:, 2], label="Gyro Bias Z", color='b')
    ax[0].set_title("Gyroscope Bias Convergence")
    ax[0].set_ylabel("Bias (deg/s)")
    ax[0].grid(True)
    ax[0].legend(loc="upper right")

    ax[1].plot(t, ba_hist[:, 0], label="Accel Bias X", color='r')
    ax[1].plot(t, ba_hist[:, 1], label="Accel Bias Y", color='g')
    ax[1].plot(t, ba_hist[:, 2], label="Accel Bias Z", color='b')
    ax[1].set_title("Accelerometer Bias Convergence")
    ax[1].set_ylabel("Bias (m/s^2)")
    ax[1].set_xlabel("Time (s)")
    ax[1].grid(True)
    ax[1].legend(loc="upper right")
    if title_suffix:
        fig.suptitle(title_suffix)
    fig.tight_layout()
    return fig, ax

def plot_separated_nis(t_gps, nis_gps, dof_gps, t_mag, nis_mag, dof_mag, N=1):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    if len(nis_gps) > 0:
        lo_gps, hi_gps = chi2.ppf(0.025, N * dof_gps) / N, chi2.ppf(0.975, N * dof_gps) / N
        ax[0].plot(t_gps, nis_gps, color="royalblue", linewidth=0.8, label="GPS NIS")
        ax[0].axhline(dof_gps, color="black", linewidth=1.2, label=f"Ideal = {dof_gps}")
        ax[0].axhline(hi_gps, color="crimson", linestyle="--", linewidth=1, label="95% Bound")
        ax[0].set_ylim(0, min(hi_gps * 4, np.percentile(nis_gps, 98) * 1.5))
    ax[0].set_title(f"GPS NIS Consistency (DOF = {dof_gps})")
    ax[0].set_ylabel("NIS")
    ax[0].grid(True)
    ax[0].legend(loc="upper right")

    if len(nis_mag) > 0:
        lo_mag, hi_mag = chi2.ppf(0.025, N * dof_mag) / N, chi2.ppf(0.975, N * dof_mag) / N
        ax[1].plot(t_mag, nis_mag, color="darkorange", linewidth=0.8, label="Mag NIS")
        ax[1].axhline(dof_mag, color="black", linewidth=1.2, label=f"Ideal = {dof_mag}")
        ax[1].axhline(hi_mag, color="crimson", linestyle="--", linewidth=1, label="95% Bound")
        ax[1].set_ylim(0, min(hi_mag * 4, np.percentile(nis_mag, 98) * 1.5))
    ax[1].set_title(f"Magnetometer NIS Consistency (DOF = {dof_mag})")
    ax[1].set_ylabel("NIS")
    ax[1].set_xlabel("Time (s)")
    ax[1].grid(True)
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    return fig, ax

def plot_covariance_diagonals(t, P_hist, title_suffix=""):
    """
    מצייר את ערכי האלכסון של מטריצת השונות P לאורך זמן (בסולם לוגריתמי).
    מאפשר לוודא שהמטריצה לא קורסת לאפס או מתפוצצת.
    """
    import matplotlib.pyplot as plt

    dim_P = P_hist.shape[1]

    if dim_P == 15: # matches the filter's actual error-state layout:
        # [delta_theta(0:3), delta_vel(3:6), delta_pos(6:9),
        #  delta_b_accel(9:12), delta_b_gyro(12:15)]. The previous version
        # of this function assumed [pos, vel, att, bg, ba] -- three of the
        # five panels were plotting/labeling the wrong covariance blocks.
        fig, ax = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

        # Attitude Variances
        ax[0].plot(t, P_hist[:, 0, 0], label='Var(Att Roll)', color='r')
        ax[0].plot(t, P_hist[:, 1, 1], label='Var(Att Pitch)', color='g')
        ax[0].plot(t, P_hist[:, 2, 2], label='Var(Att Yaw)', color='b')
        ax[0].set_title('Attitude Covariance (P_att)')
        ax[0].set_yscale('log')
        ax[0].grid(True, which="both", ls="--", alpha=0.5)
        ax[0].legend(loc="upper right")

        # Velocity Variances
        ax[1].plot(t, P_hist[:, 3, 3], label='Var(Vel X)', color='r')
        ax[1].plot(t, P_hist[:, 4, 4], label='Var(Vel Y)', color='g')
        ax[1].plot(t, P_hist[:, 5, 5], label='Var(Vel Z)', color='b')
        ax[1].set_title('Velocity Covariance (P_vel)')
        ax[1].set_yscale('log')
        ax[1].grid(True, which="both", ls="--", alpha=0.5)
        ax[1].legend(loc="upper right")

        # Position Variances
        ax[2].plot(t, P_hist[:, 6, 6], label='Var(Pos X)', color='r')
        ax[2].plot(t, P_hist[:, 7, 7], label='Var(Pos Y)', color='g')
        ax[2].plot(t, P_hist[:, 8, 8], label='Var(Pos Z)', color='b')
        ax[2].set_title('Position Covariance (P_pos)')
        ax[2].set_yscale('log')
        ax[2].grid(True, which="both", ls="--", alpha=0.5)
        ax[2].legend(loc="upper right")

        # Accel Bias Variances
        ax[3].plot(t, P_hist[:, 9, 9], label='Var(ba X)', color='r')
        ax[3].plot(t, P_hist[:, 10, 10], label='Var(ba Y)', color='g')
        ax[3].plot(t, P_hist[:, 11, 11], label='Var(ba Z)', color='b')
        ax[3].set_title('Accel Bias Covariance (P_ba)')
        ax[3].set_yscale('log')
        ax[3].grid(True, which="both", ls="--", alpha=0.5)
        ax[3].legend(loc="upper right")

        # Gyro Bias Variances
        ax[4].plot(t, P_hist[:, 12, 12], label='Var(bg X)', color='r')
        ax[4].plot(t, P_hist[:, 13, 13], label='Var(bg Y)', color='g')
        ax[4].plot(t, P_hist[:, 14, 14], label='Var(bg Z)', color='b')
        ax[4].set_title('Gyro Bias Covariance (P_bg)')
        ax[4].set_yscale('log')
        ax[4].set_xlabel('Time (s)')
        ax[4].grid(True, which="both", ls="--", alpha=0.5)
        ax[4].legend(loc="upper right")
        if title_suffix:
            fig.suptitle(title_suffix)
        fig.tight_layout()
        return fig, ax
    else:
        fig, ax = plt.subplots(figsize=(11, 4))
        for k in range(dim_P):
            ax.plot(t, P_hist[:, k, k], label=f'State {k}')
        ax.set_title('Covariance Diagonals Evolution' + (f' -- {title_suffix}' if title_suffix else ''))
        ax.set_yscale('log')
        ax.set_xlabel('Time (s)')
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
        fig.tight_layout()
        return fig, ax


def plot_euler_comparison(t_cmp, eul_est, eul_gt, accel_pts=None):
    """accel_pts: optional (t, roll_deg, pitch_deg) of the quasi-static
    accel-tilt reference -- plotted as markers on the Roll/Pitch panels so
    the filter can be compared against BOTH truths (Wahba GT line +
    accel-tilt points). No yaw (gravity carries no heading)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    names = ["Roll", "Pitch", "Yaw"]

    eul_gt_unwrapped = np.unwrap(np.radians(eul_gt), axis=0)
    eul_est_unwrapped = np.unwrap(np.radians(eul_est), axis=0)

    eul_gt_deg = np.degrees(eul_gt_unwrapped)
    eul_est_deg = np.degrees(eul_est_unwrapped)

    for k in range(3):
        ax[k].plot(t_cmp, eul_gt_deg[:, k], label="Ground Truth (Wahba)", color='dodgerblue', linewidth=1.5)
        ax[k].plot(t_cmp, eul_est_deg[:, k], label="Filter Estimate", color='darkorange', linewidth=1.5, linestyle='--')
        ax[k].set_ylabel(f"{names[k]} (deg)")
        ax[k].grid(True)

    if accel_pts is not None:
        t_a, roll_a, pitch_a = accel_pts
        if len(t_a):
            ax[0].scatter(t_a, roll_a, color='green', s=70, zorder=5, marker='o',
                          edgecolors='k', label="Truth (accel-tilt)")
            ax[1].scatter(t_a, pitch_a, color='green', s=70, zorder=5, marker='o',
                          edgecolors='k', label="Truth (accel-tilt)")
    for k in range(3):
        ax[k].legend(loc="upper right")
    ax[0].set_title("Euler Angles Comparison (Single Run - Unwrapped)")
    ax[2].set_xlabel("time (s)")
    fig.tight_layout()
    return fig, ax

def plot_batch_cdf(axis_err_runs):
    import matplotlib.pyplot as plt

    all_roll = np.concatenate([np.abs(err[:, 0]) for err in axis_err_runs])
    all_pitch = np.concatenate([np.abs(err[:, 1]) for err in axis_err_runs])
    all_yaw = np.concatenate([np.abs(err[:, 2]) for err in axis_err_runs])

    fig, ax = plt.subplots(figsize=(10, 6))

    for data, name, color in zip([all_roll, all_pitch, all_yaw],
                                 ['Roll', 'Pitch', 'Yaw'],
                                 ['red', 'green', 'blue']):
        sorted_data = np.sort(data)
        p = 1. * np.arange(len(data)) / (len(data) - 1)
        ax.plot(sorted_data, p, label=name, color=color, linewidth=2)

    ax.axhline(0.9, color='gray', linestyle='--', linewidth=1, label='90% Threshold')
    ax.set_xlim(0, np.percentile(np.concatenate([all_roll, all_pitch, all_yaw]), 98))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Absolute Error (degrees)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Cumulative Distribution Function (CDF) of Estimation Errors")
    ax.grid(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, ax

def plot_batch_rmse_bars(axis_err_runs, dataset_names=None):
    import matplotlib.pyplot as plt

    N = len(axis_err_runs)
    roll_rmses = [rmse(err[:, 0]) for err in axis_err_runs]
    pitch_rmses = [rmse(err[:, 1]) for err in axis_err_runs]
    yaw_rmses = [rmse(err[:, 2]) for err in axis_err_runs]

    x = np.arange(N)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, roll_rmses, width, label='Roll RMSE', color='red')
    ax.bar(x, pitch_rmses, width, label='Pitch RMSE', color='green')
    ax.bar(x + width, yaw_rmses, width, label='Yaw RMSE', color='blue')

    ax.set_ylabel('RMSE (degrees)')
    ax.set_title('Estimation RMSE per Flight Dataset')

    if dataset_names and len(dataset_names) == N:
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([f"Flight {i+1}" for i in range(N)])

    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    return fig, ax

def plot_batch_cdf_accel(ar_runs, axis_err_runs=None):
    """CDF of the filter's roll/pitch error at the quasi-static accel-tilt
    windows (Wahba-GT-free), pooled over all flights, plus the filter's
    yaw error vs the Wahba GT (RTK-derived, trustworthy -- identical data
    to plot_batch_cdf's yaw curve / Fig. 4). Companion to plot_batch_cdf.
      ar_runs        per-flight dicts from attitude_reference_report()
                     (may hold None / window-less dicts)
      axis_err_runs  list of (T,3) filter-vs-Wahba-GT error arrays (deg);
                     supplies the yaw curve."""
    import matplotlib.pyplot as plt

    def _pool(key):
        vals = [np.abs(np.asarray(d[key], float))
                for d in ar_runs if d and d.get(key) is not None and len(d[key])]
        return np.concatenate(vals) if vals else np.array([])

    f_roll, f_pitch = _pool("d_roll_vs_accel"), _pool("d_pitch_vs_accel")
    f_yaw = (np.concatenate([np.abs(e[:, 2]) for e in axis_err_runs])
             if axis_err_runs else np.array([]))
    if len(f_roll) == 0:
        print("[plot_batch_cdf_accel] no quasi-static windows -- skipped")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    def _cdf(data, label, color):
        if len(data) < 2:
            return
        s = np.sort(data)
        p = 1. * np.arange(len(s)) / (len(s) - 1)
        ax.plot(s, p, label=label, color=color, linewidth=2)

    _cdf(f_roll, f"Roll  (vs accel-tilt, n={len(f_roll)} win)", 'red')
    _cdf(f_pitch, f"Pitch (vs accel-tilt, n={len(f_pitch)} win)", 'green')
    _cdf(f_yaw, f"Yaw   (vs Wahba GT, n={len(f_yaw)} samp)", 'blue')

    ax.axhline(0.9, color='gray', linestyle='--', linewidth=1, label='90% Threshold')
    _all = np.concatenate([a for a in (f_roll, f_pitch, f_yaw) if len(a)])
    ax.set_xlim(0, max(float(np.percentile(_all, 98)), 1.0))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Absolute Error (degrees)  --  roll/pitch at quasi-static windows, yaw all samples")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF of Estimation Errors vs Trustworthy References "
                 "(accel-tilt roll/pitch, RTK-yaw)")
    ax.grid(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig, ax

def plot_batch_rmse_accel(ar_runs, axis_err_runs=None, dataset_names=None):
    """Per-flight attitude RMSE, front bars = the trustworthy references:
    roll/pitch vs the quasi-static accel-tilt reference (Wahba-GT-free),
    yaw vs the Wahba GT (RTK-derived). Behind (translucent) = the SAME
    axis' filter-vs-Wahba-GT RMSE over the full trimmed window -- i.e.
    exactly the bars in plot_batch_rmse_bars / Fig. 3 -- so each flight
    shows the accel-tilt truth against the noisy-GT truth side by side.
    "Nw" = still-window count that flight (2-4 typical -> thin sample).
      axis_err_runs  list of (T,3) filter-vs-Wahba-GT error arrays (deg)."""
    import matplotlib.pyplot as plt

    runs = [(i, d) for i, d in enumerate(ar_runs)
            if d and "filter_roll_rmse_vs_accel" in d]
    if not runs:
        print("[plot_batch_rmse_accel] no quasi-static windows -- skipped")
        return None

    idxs = [i for i, _ in runs]
    f_roll = [d["filter_roll_rmse_vs_accel"] for _, d in runs]
    f_pitch = [d["filter_pitch_rmse_vs_accel"] for _, d in runs]
    nwin = [d.get("n_static_windows", 0) for _, d in runs]

    have_gt = axis_err_runs is not None and len(axis_err_runs) > max(idxs)
    if have_gt:
        gt_roll = [rmse(axis_err_runs[i][:, 0]) for i in idxs]
        gt_pitch = [rmse(axis_err_runs[i][:, 1]) for i in idxs]
        gt_yaw = [rmse(axis_err_runs[i][:, 2]) for i in idxs]
    else:
        gt_roll = gt_pitch = gt_yaw = [np.nan] * len(idxs)

    x = np.arange(len(runs))
    w, off = 0.25, 0.28      # bar width, roll/pitch/yaw cluster offset

    fig, ax = plt.subplots(figsize=(13, 5.5))
    # behind: filter vs Wahba GT over the full window (= Fig. 3)
    ax.bar(x - off, gt_roll, w, color='red', alpha=0.30,
           label='Roll  vs Wahba GT (full window, = Fig. 3)')
    ax.bar(x, gt_pitch, w, color='green', alpha=0.30,
           label='Pitch vs Wahba GT (full window, = Fig. 3)')
    ax.bar(x + off, gt_yaw, w, color='blue', alpha=0.30,
           label='Yaw   vs Wahba GT (full window, = Fig. 3)')
    # front: the trustworthy references (same width -> translucent cap = GT excess)
    ax.bar(x - off, f_roll, w, color='red', label='Roll  vs accel-tilt')
    ax.bar(x, f_pitch, w, color='green', label='Pitch vs accel-tilt')
    ax.bar(x + off, gt_yaw, w, color='blue', label='Yaw   vs Wahba GT (RTK-derived)')

    for xi, n in zip(x, nwin):
        ax.annotate(f"{n}w", (xi, 0.0), textcoords="offset points", xytext=(0, 2),
                    ha='center', va='bottom', fontsize=7, color='dimgray')

    ax.set_ylabel('RMSE (degrees)')
    ax.set_title('Per-Flight Attitude RMSE: trustworthy reference (front) vs Wahba GT (behind)')
    ax.set_xticks(x)
    if dataset_names and len(dataset_names) >= max(idxs) + 1:
        ax.set_xticklabels([dataset_names[i] for i in idxs], rotation=45, ha='right')
    else:
        ax.set_xticklabels([f"Flight {i + 1}" for i in idxs])
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout()
    return fig, ax

def plot_monte_carlo_axis_errors(t_cmp, axis_err_runs, accel_pts=None):
    """accel_pts: optional dict with 't', 'd_roll', 'd_pitch' (per
    quasi-static window) -- overlaid as markers on the Roll/Pitch panels
    so the Wahba-GT error curve and the accel-tilt error can be read
    together. No yaw overlay (gravity carries no heading)."""
    import matplotlib.pyplot as plt
    err_trunc = _truncate_to_shortest(axis_err_runs)
    min_len = len(err_trunc[0])
    t_cmp = t_cmp[:min_len]

    mean_err = np.mean(err_trunc, axis=0)
    std_err = np.std(err_trunc, axis=0)
    N = len(axis_err_runs)

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    names = ["Roll Error", "Pitch Error", "Yaw Error"]
    for k in range(3):
        ax[k].plot(t_cmp, mean_err[:, k], label="Mean Error vs Wahba GT", color='darkorange', linewidth=1.5, zorder=2)
        ax[k].fill_between(t_cmp, mean_err[:, k] - 3*std_err[:, k], mean_err[:, k] + 3*std_err[:, k],
                           color='orange', alpha=0.3, label="±3σ Spread", zorder=1)
        ax[k].axhline(0, color='black', linestyle='--', linewidth=1, zorder=3)
        ax[k].set_ylabel(f"{names[k]} (deg)")
        ax[k].grid(True)

    if accel_pts and len(accel_pts.get("t", [])):
        ta = np.asarray(accel_pts["t"])
        # d_roll/d_pitch are (accel-tilt - filter); negate to match the
        # (filter - GT) sign convention of the orange curve.
        ax[0].scatter(ta, -np.asarray(accel_pts["d_roll"]), color='green', s=55,
                      marker='o', edgecolors='k', zorder=5, label="Error vs accel-tilt")
        ax[1].scatter(ta, -np.asarray(accel_pts["d_pitch"]), color='green', s=55,
                      marker='o', edgecolors='k', zorder=5, label="Error vs accel-tilt")
    for k in range(3):
        ax[k].legend(loc="upper right")

    ax[0].set_title(f"Averaged Axis Errors over Time ({N} Runs)")
    ax[2].set_xlabel("time (s)")
    fig.tight_layout()
    return fig, ax

def plot_monte_carlo_qae(t_cmp, qae_runs, accel_pts=None):
    """accel_pts: optional dict with 't', 'qae' (per quasi-static window,
    roll/pitch from accel-tilt + yaw from Wahba) -- overlaid as markers."""
    import matplotlib.pyplot as plt
    qae_trunc = _truncate_to_shortest(qae_runs)
    min_len = len(qae_trunc[0])
    t_cmp = t_cmp[:min_len]

    mean_qae = np.mean(qae_trunc, axis=0)
    std_qae = np.std(qae_trunc, axis=0)
    N = len(qae_runs)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(t_cmp, mean_qae, color="crimson", label="Mean qAE vs Wahba GT")
    ax.fill_between(t_cmp, np.clip(mean_qae - 3*std_qae, 0, None), mean_qae + 3*std_qae, color='red', alpha=0.2, label="±3σ Spread")
    if accel_pts and len(accel_pts.get("t", [])):
        ax.scatter(np.asarray(accel_pts["t"]), np.asarray(accel_pts["qae"]), color='green',
                   s=55, marker='o', edgecolors='k', zorder=5,
                   label="qAE vs accel-tilt (roll/pitch accel, yaw Wahba)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angular error (deg)")
    ax.set_title(f"Averaged Total Orientation Error (qAE) over {N} runs")
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax

def plot_monte_carlo_consistency(t_nees, nees_runs, t_nis, nis_runs, N, dof_att=3, dof_meas=6,
                                 accel_pts=None):
    """accel_pts: optional dict with 't', 'roll_nees', 'pitch_nees' (per
    quasi-static window, each dof=1 vs accel-tilt). Their sum (dof=2) is
    overlaid as markers -- the consistency check NOT contaminated by the
    Wahba-GT roll/pitch noise that inflates the teal ANEES line."""
    import matplotlib.pyplot as plt
    if not nis_runs or len(nis_runs) == 0 or len(nis_runs[0]) == 0:
        pass

    nees_trunc = _truncate_to_shortest(nees_runs)
    t_nees = t_nees[:len(nees_trunc[0])]
    anees = np.mean(nees_trunc, axis=0)
    lo_a, hi_a = chi2.ppf(0.025, N * dof_att) / N,  chi2.ppf(0.975, N * dof_att) / N

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t_nees, anees, color="teal", linewidth=0.7, label="ANEES vs Wahba GT (dof=3)")
    ax.axhline(dof_att, color="black", linewidth=1.2, label=f"ideal = {dof_att}")
    ax.axhline(lo_a, color="crimson", linestyle="--", linewidth=0.9)
    ax.axhline(hi_a, color="crimson", linestyle="--", linewidth=0.9, label=f"95% bounds (N={N})")

    y_top = min(hi_a * 4, np.percentile(anees, 98) * 1.2)
    if accel_pts and len(accel_pts.get("t", [])):
        rp = np.asarray(accel_pts["roll_nees"]) + np.asarray(accel_pts["pitch_nees"])
        ax.scatter(np.asarray(accel_pts["t"]), rp, color='green', s=55, marker='o',
                   edgecolors='k', zorder=5, label="NEES vs accel-tilt (roll+pitch, dof=2)")
        ax.axhline(2.0, color='green', linestyle=':', linewidth=1.0, label="ideal = 2 (accel-tilt)")
        y_top = max(y_top, float(np.percentile(rp, 95)) * 1.3 if len(rp) else y_top)

    ax.set_ylabel("NEES")
    ax.set_xlabel("time (s)")
    ax.set_ylim(0, y_top)
    ax.set_title(f"Averaged NEES over {N} runs -- Wahba-GT line vs accel-tilt markers")
    ax.grid(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig, ax

def plot_trajectory_3d(pos_gt, pos_est, ate=None, title_suffix=""):
    """Ground-truth vs filter trajectory as a 3D line plot, INSANE paper
    Fig. 29 style. Inputs are NED (N, E, Down); drawn as N / E / Up.

    NO alignment is applied: this filter is GPS-aided, so its estimate
    already lives in the absolute NED frame. (Fig. 29 aligns its VIO
    result with an Umeyama fit first, because VIO has 4 unobservable DoF
    -- global yaw + 3D translation -- and is otherwise not comparable to
    an absolute GT. That step is neither needed nor appropriate here, and
    showing the raw trajectory is the stronger claim.)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

    gt = np.asarray(pos_gt, float)
    es = np.asarray(pos_est, float)
    Ngt, Egt, Ugt = gt[:, 0], gt[:, 1], -gt[:, 2]
    Nes, Ees, Ues = es[:, 0], es[:, 1], -es[:, 2]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Ngt, Egt, Ugt, color='dodgerblue', linewidth=1.8, label='Ground Truth (RTK)')
    ax.plot(Nes, Ees, Ues, color='darkorange', linewidth=1.8, label='Filter Estimate')
    ax.scatter(Ngt[0], Egt[0], Ugt[0], color='green', s=55, marker='o', label='start')
    ax.scatter(Ngt[-1], Egt[-1], Ugt[-1], color='red', s=55, marker='s', label='end')

    allN = np.concatenate([Ngt, Nes])
    allE = np.concatenate([Egt, Ees])
    allU = np.concatenate([Ugt, Ues])
    r = max(np.ptp(allN), np.ptp(allE), np.ptp(allU)) / 2.0 or 1.0
    cN, cE, cU = allN.mean(), allE.mean(), allU.mean()
    ax.set_xlim(cN - r, cN + r)
    ax.set_ylim(cE - r, cE + r)
    ax.set_zlim(cU - r, cU + r)

    ax.set_xlabel('N [m]')
    ax.set_ylabel('E [m]')
    ax.set_zlabel('Up [m]')
    ttl = 'Trajectory: Ground Truth vs Filter Estimate  (absolute NED, no alignment)'
    if ate is not None:
        ttl += f'\nATE = {ate:.3f} m'
    if title_suffix:
        ttl += f'   [{title_suffix}]'
    ax.set_title(ttl)
    ax.legend(loc='upper left')
    fig.tight_layout()
    return fig, ax

def plot_3d_attitude_trajectory(pos_gt, quat_gt, quat_est, step=150):
    import plotly.graph_objects as go
    import os
    import webbrowser

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=pos_gt[:, 0], y=pos_gt[:, 1], z=pos_gt[:, 2],
        mode='lines', name='Flight Path (RTK)',
        line=dict(color='gray', width=3, dash='solid'), opacity=0.3
    ))

    max_range = np.array([pos_gt[:, 0].max()-pos_gt[:, 0].min(),
                          pos_gt[:, 1].max()-pos_gt[:, 1].min(),
                          pos_gt[:, 2].max()-pos_gt[:, 2].min()]).max()
    axis_len = max_range * 0.06
    shaft_len = axis_len * 0.7
    cone_len = axis_len * 0.3

    axes_data = {
        'True X': {'color': 'red',     'shafts': [[], [], []], 'cones': [[], [], [], [], [], []]},
        'True Y': {'color': 'green',   'shafts': [[], [], []], 'cones': [[], [], [], [], [], []]},
        'True Z': {'color': 'blue',    'shafts': [[], [], []], 'cones': [[], [], [], [], [], []]},
        'Est X':  {'color': 'orange',  'shafts': [[], [], []], 'cones': [[], [], [], [], [], []]},
        'Est Y':  {'color': 'magenta', 'shafts': [[], [], []], 'cones': [[], [], [], [], [], []]},
        'Est Z':  {'color': 'cyan',    'shafts': [[], [], []], 'cones': [[], [], [], [], [], []]},
    }

    for i in range(0, len(pos_gt), step):
        p = pos_gt[i]
        R_gt = q.to_rotation_matrix(quat_gt[i])
        R_est = q.to_rotation_matrix(quat_est[i])

        for col in range(3):
            dir_gt = R_gt[:, col]
            dir_est = R_est[:, col]

            gt_name = f'True {chr(88+col)}'
            axes_data[gt_name]['shafts'][0].extend([p[0], p[0] + dir_gt[0]*shaft_len, None])
            axes_data[gt_name]['shafts'][1].extend([p[1], p[1] + dir_gt[1]*shaft_len, None])
            axes_data[gt_name]['shafts'][2].extend([p[2], p[2] + dir_gt[2]*shaft_len, None])
            axes_data[gt_name]['cones'][0].append(p[0] + dir_gt[0]*shaft_len)
            axes_data[gt_name]['cones'][1].append(p[1] + dir_gt[1]*shaft_len)
            axes_data[gt_name]['cones'][2].append(p[2] + dir_gt[2]*shaft_len)
            axes_data[gt_name]['cones'][3].append(dir_gt[0]*cone_len)
            axes_data[gt_name]['cones'][4].append(dir_gt[1]*cone_len)
            axes_data[gt_name]['cones'][5].append(dir_gt[2]*cone_len)

            est_name = f'Est {chr(88+col)}'
            axes_data[est_name]['shafts'][0].extend([p[0], p[0] + dir_est[0]*shaft_len, None])
            axes_data[est_name]['shafts'][1].extend([p[1], p[1] + dir_est[1]*shaft_len, None])
            axes_data[est_name]['shafts'][2].extend([p[2], p[2] + dir_est[2]*shaft_len, None])
            axes_data[est_name]['cones'][0].append(p[0] + dir_est[0]*shaft_len)
            axes_data[est_name]['cones'][1].append(p[1] + dir_est[1]*shaft_len)
            axes_data[est_name]['cones'][2].append(p[2] + dir_est[2]*shaft_len)
            axes_data[est_name]['cones'][3].append(dir_est[0]*cone_len)
            axes_data[est_name]['cones'][4].append(dir_est[1]*cone_len)
            axes_data[est_name]['cones'][5].append(dir_est[2]*cone_len)

    for name, data in axes_data.items():
        color = data['color']
        is_true = 'True' in name
        fig.add_trace(go.Scatter3d(
            x=data['shafts'][0], y=data['shafts'][1], z=data['shafts'][2],
            mode='lines', name=name,
            line=dict(color=color, width=4 if is_true else 7), legendgroup=name
        ))
        fig.add_trace(go.Cone(
            x=data['cones'][0], y=data['cones'][1], z=data['cones'][2],
            u=data['cones'][3], v=data['cones'][4], w=data['cones'][5],
            colorscale=[[0, color], [1, color]], showscale=False,
            sizemode='absolute', sizeref=cone_len * 1.5, anchor='tail',
            name=name, legendgroup=name, showlegend=False
        ))

    fig.update_layout(
        title="Interactive 3D Flight Trajectory with Attitude Arrows",
        scene=dict(
            xaxis_title='North (m)', yaxis_title='East (m)', zaxis_title='Down (m)',
            aspectmode='data', zaxis=dict(autorange="reversed")
        ),
        legend=dict(x=0, y=1), margin=dict(l=0, r=0, b=0, t=40)
    )

    html_file = "3d_trajectory_arrows.html"
    fig.write_html(html_file)
    print(f"\n>>> 3D interactive plot saved to {html_file}. Opening in browser...")
    webbrowser.open('file://' + os.path.realpath(html_file))

def export_batch_results_to_csv(qae_runs, axis_err_runs, nees_runs, nis_runs, filter_name="UKF", filename=None):
    if filename is None:
        filename = f"תוצאות {filter_name}.csv"

    header1 = [
        "0", "RMSE - Root Mean Square Error", "", "", "",
        "MAE -Mean Absolute Error", "", "",
        "Safety - Max Absolute Error", "", "", "", ""
    ]

    header2 = [
        "run", "Roll Mean", "Pitch Mean", "Yaw Mean", "Mean qAE",
        "Roll Mean MAE", "Pitch Mean MAE", "Yaw Mean MAE",
        "Roll Max Error", "Pitch Max Error", "Yaw Max Error",
        "NIS (Ideal 6)", "NEES (ideal 3)"
    ]

    with open(filename, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header1)
        writer.writerow(header2)

        for i in range(len(qae_runs)):
            axis_err = axis_err_runs[i]

            roll_rmse = round(rmse(axis_err[:, 0]), 3)
            pitch_rmse = round(rmse(axis_err[:, 1]), 3)
            yaw_rmse = round(rmse(axis_err[:, 2]), 3)

            mean_qae = round(np.mean(qae_runs[i]), 3)

            roll_mae = round(mae(axis_err[:, 0]), 3)
            pitch_mae = round(mae(axis_err[:, 1]), 3)
            yaw_mae = round(mae(axis_err[:, 2]), 3)

            roll_max = round(max_err(axis_err[:, 0]), 3)
            pitch_max = round(max_err(axis_err[:, 1]), 3)
            yaw_max = round(max_err(axis_err[:, 2]), 3)

            nis_mean = round(np.mean(nis_runs[i]), 1) if (nis_runs and i < len(nis_runs) and len(nis_runs[i]) > 0) else ""
            nees_mean = round(np.mean(nees_runs[i]), 1) if (nees_runs and i < len(nees_runs) and len(nees_runs[i]) > 0) else ""

            row = [
                i + 1,
                roll_rmse, pitch_rmse, yaw_rmse, mean_qae,
                roll_mae, pitch_mae, yaw_mae,
                roll_max, pitch_max, yaw_max,
                nis_mean, nees_mean
            ]
            writer.writerow(row)

    print(f"\n>>> Batch results successfully exported to '{filename}'")

def export_accel_ref_csv(ar_runs, dataset_names=None, filter_name="HYBRID", filename=None):
    """Per-flight error metrics scored vs the quasi-static accel-tilt
    reference (roll/pitch) -- the Wahba-GT-free companion to
    export_batch_results_to_csv. `ar_runs` are the per-flight dicts from
    attitude_reference_report() (may contain None / window-less dicts)."""
    if filename is None:
        filename = f"תוצאות {filter_name} accel-tilt.csv"

    cols = ["flight", "n_windows",
            "Roll RMSE", "Pitch RMSE", "Roll MAE", "Pitch MAE",
            "Roll MaxAbs", "Pitch MaxAbs", "Roll bias(signed)", "Pitch bias(signed)",
            "Roll NEES (dof1)", "Pitch NEES (dof1)", "qAE (accel r/p, Wahba yaw)",
            "gyro-consist Roll", "gyro-consist Pitch",
            "WahbaGT Roll vs accel", "WahbaGT Pitch vs accel"]

    def g(d, k):
        v = d.get(k)
        return round(v, 3) if isinstance(v, (int, float)) else ""

    with open(filename, mode='w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, d in enumerate(ar_runs):
            name = dataset_names[i] if dataset_names and i < len(dataset_names) else f"run {i+1}"
            if not d or "filter_roll_rmse_vs_accel" not in d:
                w.writerow([name, d.get("n_static_windows", 0) if d else 0] + [""] * (len(cols) - 2))
                continue
            w.writerow([
                name, d.get("n_static_windows", 0),
                g(d, "filter_roll_rmse_vs_accel"), g(d, "filter_pitch_rmse_vs_accel"),
                g(d, "filter_roll_mae_vs_accel"), g(d, "filter_pitch_mae_vs_accel"),
                g(d, "filter_roll_max_vs_accel"), g(d, "filter_pitch_max_vs_accel"),
                g(d, "filter_roll_bias_vs_accel"), g(d, "filter_pitch_bias_vs_accel"),
                g(d, "filter_roll_nees_vs_accel"), g(d, "filter_pitch_nees_vs_accel"),
                g(d, "qae_accelref"),
                g(d, "filter_roll_gyro_consistency_rmse"), g(d, "filter_pitch_gyro_consistency_rmse"),
                g(d, "gt_roll_rmse_vs_accel"), g(d, "gt_pitch_rmse_vs_accel"),
            ])
    print(f">>> Accel-tilt results exported to '{filename}'")