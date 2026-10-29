"""
metrics_usque.py
----------------
USQUE-specific evaluation layer on top of metrics.py.

metrics.py is shared with the other filters (EKF / UKF / 15-state ES-EKF), so
it is left untouched. This module re-exports everything from it and overrides
ONLY the pieces that are wrong or misleading for the 6-state error-state USQUE:

  1. print_evaluation_report():
       * time-averaged ANEES / ANIS get a proper confidence band.
         metrics.py builds the band with  chi2.ppf(q, N*dof)/N  where N is the
         number of Monte-Carlo RUNS (=1 in the usual single-flight case). But
         the scalar it prints is ALSO averaged over ~10^4 time steps, so it was
         being compared against the single-sample interval (e.g. [0.22, 9.35]
         for dof=3) -- an ANEES of 8 then "passes" a test it should fail.
         Here the band uses an effective sample count derived from the NEES
         autocorrelation time (Sokal), which is the honest yardstick for a
         time-averaged consistency statistic.
       * ANIS is labelled as post-adaptive-R: usque.update() returns P_zz built
         with the INFLATED R, so NIS is deflated exactly when the adaptive
         modules fire. Treat ANIS as optimistic, not as ground truth.
       * the accel-bias / GPS / 15-state columns are dropped (not applicable).

  2. axis_error_deg(): guards the ZYX-Euler gimbal-lock region (|pitch|~90 deg),
     where roll/yaw are ill-defined and wrap_deg() turns the ambiguity into
     huge spurious error. qAE (geodesic) stays the primary accuracy metric.

  3. nees_per_axis(): roll/pitch/yaw NEES (ideal 1 each) for diagnosing WHICH
     axis is inconsistent -- usually pitch, from sustained linear acceleration
     tilting the gravity reference.

  4. print_evaluation_report(accel_ref=...): the accel-tilt reference block
     (roll/pitch scored at quasi-static windows, no Wahba GT involved) and the
     gyro-integration cross-check. NOTE for USQUE: the accelerometer is fused
     continuously, so "vs accel-tilt" measures AGREEMENT with a signal the
     filter already uses; only the gyro cross-check is independent of it.

  5. plot_bias_convergence() / plot_covariance_diagonals(): 6-state versions
     (gyro bias only, no accel-bias state; attitude + bias sigma panels).

  6. build_accel_ref(), yaw_vs_rtk(), mean_histories_6(): the helper glue
     main.py keeps privately, adapted to the 6-state layout.

  7. plot_attitude_trajectory(): USQUE has NO position estimate, so the
     "trajectory" figure is the ground-truth flight path with the GT and the
     estimated body-axis arrows drawn along it (matplotlib, no plotly).

NEES itself is already correct in metrics.py for USQUE and is inherited as-is:
_error_and_cov() forms  q_err = q_est^-1 (x) q_true  (a right / body-frame
error) and slices P[0:3,0:3] (the GRP attitude block) -- both match USQUE's
right-multiplicative error convention and main_usque.py's left-side world
alignment (which does not rotate the body-frame covariance).

Drop-in: change `import metrics as mt` -> `import metrics_usque as mt` in
main_usque.py. Every name main uses still resolves.
"""

import numpy as np
from scipy.stats import chi2

from metrics import *            # noqa: F401,F403  (rmse, plots, CSV, qae_deg, nees_series, ...)
from metrics import _error_and_cov   # private helper (rotation-vector error + attitude block)
import project_utils as q

# --------------------------------------------------------------------------
# Effective sample size for a time-averaged consistency statistic
# --------------------------------------------------------------------------
def integrated_autocorr_time(x, sokal_c=5.0):
    """Sokal-windowed integrated autocorrelation time tau of series x.
    A run of K samples carries ~K/tau independent pieces of information."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 16:
        return 1.0
    x = x - x.mean()
    v = np.dot(x, x) / n
    if v <= 0:
        return 1.0
    # linear (zero-padded) autocovariance via FFT, biased 1/n normalisation
    m = 1
    while m < 2 * n:
        m <<= 1
    f = np.fft.rfft(x, m)
    acf = np.fft.irfft(f * np.conj(f), m)[:n] / (n * v)   # acf[0] == 1
    tau = 1.0
    for k in range(1, n):
        tau += 2.0 * acf[k]
        if k >= sokal_c * tau:
            break
    return float(max(tau, 1.0))


def effective_sample_size(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1.0
    return float(max(1.0, x.size / integrated_autocorr_time(x)))


def anees_interval(k_eff, dof, alpha=0.05):
    """Two-sided (1-alpha) interval for the MEAN of k_eff independent
    NEES samples, each with `dof` degrees of freedom. Ideal value is `dof`."""
    k_eff = max(k_eff, 1.0)
    lo = chi2.ppf(alpha / 2.0, k_eff * dof) / k_eff
    hi = chi2.ppf(1.0 - alpha / 2.0, k_eff * dof) / k_eff
    return lo, hi


def _pool(runs):
    """flatten a list of 1-D arrays (skip empties)."""
    runs = [np.asarray(r, float) for r in runs if r is not None and len(r) > 0]
    return np.concatenate(runs) if runs else np.array([])


# --------------------------------------------------------------------------
# Accuracy: gimbal-lock-safe axis error
# --------------------------------------------------------------------------
def axis_error_deg(q_est, q_true, pitch_guard_deg=85.0):
    """roll/pitch/yaw error in deg, ZYX Euler, with the gimbal-lock band
    blanked to NaN so it does not pollute RMSE/MAE. qAE is unaffected."""
    e_est = np.degrees(np.array([q.to_euler(e) for e in q_est]))
    e_true = np.degrees(np.array([q.to_euler(g) for g in q_true]))
    err = wrap_deg(e_est - e_true)                      # noqa: F405 (from metrics)
    lock = (np.abs(e_true[:, 1]) > pitch_guard_deg) | (np.abs(e_est[:, 1]) > pitch_guard_deg)
    if np.any(lock):
        err = err.astype(float).copy()
        err[lock, 0] = np.nan     # roll  ill-defined
        err[lock, 2] = np.nan     # yaw   ill-defined
    return err


# --------------------------------------------------------------------------
# Per-axis NEES (ideal 1.0 per axis)
# --------------------------------------------------------------------------
def nees_per_axis(q_est, q_true, P_hist, valid_mask=None):
    N = len(q_est)
    out = np.full((N, 3), np.nan)
    for i in range(N):
        if valid_mask is not None and not valid_mask[i]:
            continue
        qe = q.normalize(q_est[i])
        qt = q.normalize(q_true[i])
        dq = q.multiply(q.inverse(qe), qt)
        if dq[0] < 0:
            dq = -dq
        nv = np.linalg.norm(dq[1:4])
        ang = 2.0 * np.arctan2(nv, np.clip(dq[0], -1.0, 1.0))
        theta = (dq[1:4] / nv) * ang if nv > 1e-9 else np.zeros(3)
        P = np.asarray(P_hist[i], float)
        P_att = P[0:3, 0:3] if P.shape[0] >= 3 else P
        d = np.clip(np.diag(P_att), 1e-15, None)
        out[i] = theta ** 2 / d
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def print_evaluation_report(qae_runs, axis_err_runs, nees_runs, nis_runs, N,
                            dof_att=3, dof_meas=6, accel_ref=None,
                            nees_axis_runs=None, label="", nees_ref=None):
    """USQUE evaluation report. `accel_ref` (from build_accel_ref) adds the
    accel-tilt block; `label` tags the header (e.g. the init mode).
    `nees_ref`: optional dict(mean=..., sigma_ref_deg=(r,p,y), axis=(r,p,y)) -- the
    ANEES recomputed with the reference's own uncertainty in the denominator."""

    def _nanmean_over_runs(fn, runs):
        vals = [fn(r) for r in runs if r is not None and len(r) > 0]
        return float(np.nanmean(vals)) if vals else float("nan")

    mean_qae = _nanmean_over_runs(np.nanmean, qae_runs)
    mean_nees = _nanmean_over_runs(np.nanmean, nees_runs)
    valid_nis = [r for r in nis_runs if r is not None and len(r) > 0]
    mean_nis = float(np.nanmean([np.nanmean(r) for r in valid_nis])) if valid_nis else 0.0

    r_rmse = _nanmean_over_runs(lambda e: _rms(e[:, 0]), axis_err_runs)
    p_rmse = _nanmean_over_runs(lambda e: _rms(e[:, 1]), axis_err_runs)
    y_rmse = _nanmean_over_runs(lambda e: _rms(e[:, 2]), axis_err_runs)
    r_mae = _nanmean_over_runs(lambda e: np.nanmean(np.abs(e[:, 0])), axis_err_runs)
    p_mae = _nanmean_over_runs(lambda e: np.nanmean(np.abs(e[:, 1])), axis_err_runs)
    y_mae = _nanmean_over_runs(lambda e: np.nanmean(np.abs(e[:, 2])), axis_err_runs)
    r_max = _nanmean_over_runs(lambda e: np.nanmax(np.abs(e[:, 0])), axis_err_runs)
    p_max = _nanmean_over_runs(lambda e: np.nanmax(np.abs(e[:, 1])), axis_err_runs)
    y_max = _nanmean_over_runs(lambda e: np.nanmax(np.abs(e[:, 2])), axis_err_runs)

    # ---- consistency bands -------------------------------------------------
    nees_all = _pool(nees_runs)
    k_eff_nees = sum(effective_sample_size(r) for r in nees_runs if r is not None and len(r) > 0)
    ss_lo_a, ss_hi_a = anees_interval(1.0, dof_att)                 # single-sample
    ta_lo_a, ta_hi_a = anees_interval(max(k_eff_nees, 1.0), dof_att)  # time-averaged

    if valid_nis:
        nis_all = _pool(valid_nis)
        k_eff_nis = sum(effective_sample_size(r) for r in valid_nis)
        ss_lo_m, ss_hi_m = anees_interval(1.0, dof_meas)
        ta_lo_m, ta_hi_m = anees_interval(max(k_eff_nis, 1.0), dof_meas)

    def _verdict(val, lo, hi):
        return "OK" if lo <= val <= hi else ("HIGH (over-confident)" if val > hi else "LOW (conservative)")

    print(f"\n============= USQUE EVALUATION REPORT (runs={N})"
          f"{' [' + label + ']' if label else ''} =============")
    print("--- ACCURACY ---")
    print(f"Mean qAE (geodesic, primary) : {mean_qae:7.3f} deg")
    print(f"Roll  RMSE / MAE / Max       : {r_rmse:7.3f} / {r_mae:7.3f} / {r_max:7.3f} deg")
    print(f"Pitch RMSE / MAE / Max       : {p_rmse:7.3f} / {p_mae:7.3f} / {p_max:7.3f} deg")
    print(f"Yaw   RMSE / MAE / Max       : {y_rmse:7.3f} / {y_mae:7.3f} / {y_max:7.3f} deg")
    ae = [np.concatenate([np.abs(e[:, k]) for e in axis_err_runs if e is not None and len(e)]) for k in range(3)]
    ae = [a[np.isfinite(a)] for a in ae]
    print(f"P90 |error| (pooled)         : {np.percentile(ae[0], 90):7.2f} / {np.percentile(ae[1], 90):7.2f} / "
          f"{np.percentile(ae[2], 90):7.2f} deg   (roll / pitch / yaw)")
    print(f"CDF(5 deg) (pooled, goal 0.9): {(ae[0] < 5).mean():7.2f} / {(ae[1] < 5).mean():7.2f} / "
          f"{(ae[2] < 5).mean():7.2f}       (fraction of samples with |error| < 5 deg)")

    a = accel_ref or {}
    if a:
        def _f(k):
            v = a.get(k)
            return f"{v:7.3f}" if v is not None else "    n/a"
        print(f"\n--- vs ACCEL-TILT reference ({a.get('n_windows', 0)} still window(s); roll/pitch only) ---")
        print(f"qAE (accel r/p + Wahba yaw)  : {_f('qae')} deg")
        print(f"Roll  RMSE / MAE / Max       : {_f('roll_rmse')} / {_f('roll_mae')} / {_f('roll_max')} deg")
        print(f"Pitch RMSE / MAE / Max       : {_f('pitch_rmse')} / {_f('pitch_mae')} / {_f('pitch_max')} deg")
        fl = a.get('floor')
        if fl is not None and a.get('roll_rmse') is not None:
            unres = a['roll_rmse'] <= fl and a['pitch_rmse'] <= fl
            print(f"reference floor              : ~{fl:.2f} deg   "
                  + ("<-- values at/below the floor are UNRESOLVED (no resolving power), not 'small'"
                     if unres else ""))
        if a.get('n_windows', 0) < 8:
            print(f"  WARNING: only {a.get('n_windows', 0)} still window(s) over {a.get('n_flights', 0)} flight(s): "
                  f"these are observations, not an RMSE.")
        print(f"per-axis NEES (dof=1)        : roll {_f('roll_nees')} / pitch {_f('pitch_nees')}")
        if a.get('gt_roll') is not None:
            print(f"Wahba-GT's own error vs tilt : roll {a['gt_roll']:.2f} / pitch {a['gt_pitch']:.2f} deg "
                  f"(why 'vs GT' roll/pitch >> 'vs accel-tilt')")
        if a.get('gyro_roll') is not None:
            print(f"gyro-integration cross-check : roll {a['gyro_roll']:.2f} / pitch {a['gyro_pitch']:.2f} deg "
                  f"(independent of accel; drift-bounded)")
        print("  NOTE: USQUE fuses the accelerometer every step, so this column is agreement with a")
        print("        signal the filter already uses -- an optimistic score, not independent accuracy.")

    print("\n--- CONSISTENCY (ANEES, attitude, dof={}) ---".format(dof_att))
    print(f"ANEES vs GT (P only)         : {mean_nees:9.3f}   (ideal {dof_att})")
    print(f"  per-sample 95% band        : [{ss_lo_a:6.2f}, {ss_hi_a:6.2f}]   (for the NEES-vs-time curve)")
    print(f"  time-avg 95% band          : [{ta_lo_a:6.3f}, {ta_hi_a:6.3f}]   "
          f"(N_eff={k_eff_nees:.0f} of {nees_all.size} samples)")
    print(f"  verdict                    : {_verdict(mean_nees, ta_lo_a, ta_hi_a)}")
    if nees_ref is not None:
        sr = nees_ref['sigma_ref_deg']
        print(f"ANEES, GT-noise-aware        : {nees_ref['mean']:9.3f}   (ideal {dof_att}; denominator P + P_ref, "
              f"P_ref sigma r/p/y = {sr[0]:.2f}/{sr[1]:.2f}/{sr[2]:.2f} deg)")
        if nees_ref.get('axis') is not None:
            ax_ = nees_ref['axis']
            print(f"  per-axis (ideal 1)         : roll {ax_[0]:.2f} / pitch {ax_[1]:.2f} / yaw {ax_[2]:.2f}")
        print("  NOTE: P_ref is estimated from the Wahba-vs-accel-tilt residuals / RTK sigma; filter and GT")
        print("        errors are positively correlated (shared magnetometer) so this is optimistic.")

    if nees_axis_runs is not None:
        ax = _pool_2d(nees_axis_runs)
        if ax.size:
            m = np.nanmean(ax, axis=0)
            print(f"  per-axis NEES (ideal 1)    : roll {m[0]:.2f} / pitch {m[1]:.2f} / yaw {m[2]:.2f}")

    if valid_nis:
        print("\n--- CONSISTENCY (ANIS, measurement, dof={}) ---".format(dof_meas))
        print(f"ANIS  (base R, un-inflated)  : {mean_nis:9.3f}   (ideal {dof_meas})")
        print(f"  time-avg 95% band          : [{ta_lo_m:6.3f}, {ta_hi_m:6.3f}]   (N_eff={k_eff_nis:.0f})")
        print(f"  verdict                    : {_verdict(mean_nis, ta_lo_m, ta_hi_m)}")
        print("  NOTE: S uses the un-inflated R (update() returns S_base for reporting and uses the")
        print("        inflated S only for the gain). ANIS << ideal means R is too large or the")
        print("        adaptive branches distrust the channel most of the time -- see the firing table.")

    print("\n--- CAVEATS ---")
    print("* qAE / axis errors are scored against the site GT WITHOUT any world-frame alignment")
    print("  (main_usque.ALIGN_TO_GT=False); if it was switched on, a constant heading offset is")
    print("  absorbed and does not appear above. Mars/outdoor/transition GT roll/pitch is Wahba-")
    print("  noisy (~8-11 deg): trust yaw, the accel-tilt block and the gyro cross-check instead.")
    print("* NEES uses the body-frame GRP block P[0:3,0:3]; matches the filter's right-")
    print("  multiplicative error and is unaffected by a left-side world alignment.")
    print("============================================================")


def _rms(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def _pool_2d(runs):
    runs = [np.asarray(r, float) for r in runs if r is not None and len(r) > 0]
    return np.vstack(runs) if runs else np.empty((0, 3))


# --------------------------------------------------------------------------
# Glue that main.py keeps privately, adapted to the 6-state USQUE layout
# --------------------------------------------------------------------------
def yaw_vs_rtk(t_imu, q_est, P_hist, t_hdg, yaw_hdg):
    """Yaw error and 1-DOF NEES against the dual-RTK heading (~1 deg, bias
    free -- unlike the Wahba GT roll/pitch). P[2,2] is the body-frame yaw
    variance of the GRP error state. Returns a dict or None."""
    if t_hdg is None or yaw_hdg is None or len(t_hdg) < 5:
        return None
    idx = np.clip(np.searchsorted(t_imu, np.asarray(t_hdg)), 0, len(t_imu) - 1)
    yaw_est = np.array([q.to_euler(q_est[i])[2] for i in idx])
    e = (yaw_est - np.asarray(yaw_hdg) + np.pi) % (2.0 * np.pi) - np.pi
    var = np.maximum(P_hist[idx][:, 2, 2], 1e-12)
    nees = e ** 2 / var
    ed = np.degrees(e)
    return dict(rmse=float(np.sqrt(np.mean(ed ** 2))), mae=float(np.mean(np.abs(ed))),
                maxabs=float(np.max(np.abs(ed))), nees_mean=float(np.mean(nees)),
                nees_median=float(np.median(nees)), n=len(e))


def build_accel_ref(ar_runs):
    """Batch-mean roll/pitch metrics vs the quasi-static accel-tilt reference,
    shaped for print_evaluation_report(accel_ref=...). None if no still
    windows were found in any flight."""
    runs = [d for d in ar_runs if d and "filter_roll_rmse_vs_accel" in d]
    if not runs:
        return None

    def m(k):
        vals = [d[k] for d in runs if k in d and d[k] is not None]
        return float(np.mean(vals)) if vals else None

    return dict(
        n_windows=sum(d.get("n_static_windows", 0) for d in runs),
        qae=m("qae_accelref"),
        roll_rmse=m("filter_roll_rmse_vs_accel"), pitch_rmse=m("filter_pitch_rmse_vs_accel"),
        roll_mae=m("filter_roll_mae_vs_accel"), pitch_mae=m("filter_pitch_mae_vs_accel"),
        roll_max=m("filter_roll_max_vs_accel"), pitch_max=m("filter_pitch_max_vs_accel"),
        roll_nees=m("filter_roll_nees_vs_accel"), pitch_nees=m("filter_pitch_nees_vs_accel"),
        gt_roll=m("gt_roll_rmse_vs_accel"), gt_pitch=m("gt_pitch_rmse_vs_accel"),
        gyro_roll=m("filter_roll_gyro_consistency_rmse"),
        gyro_pitch=m("filter_pitch_gyro_consistency_rmse"),
        floor=m("ref_floor_deg"),
        n_flights=len(runs),
    )


def mean_histories_6(hist_runs, n_grid=1500):
    """Mean over flights of (t_rel, bg(M,3), Pdiag(M,L)) on a common relative time grid
    [0, shortest flight]. L is 6 (no mag-bias state) or 7 (with it), read from the data.
    Returns (t_grid, bg_mean, Pdiag_mean)."""
    t_end = min(float(t[-1]) for t, *_ in hist_runs)
    tg = np.linspace(0.0, t_end, n_grid)
    ncol = hist_runs[0][2].shape[1]
    bg = np.zeros((n_grid, 3))
    pdg = np.zeros((n_grid, ncol))
    for t_rel, bg_h, pd_h in hist_runs:
        for c in range(3):
            bg[:, c] += np.interp(tg, t_rel, bg_h[:, c])
        for c in range(ncol):
            pdg[:, c] += np.interp(tg, t_rel, pd_h[:, c])
    return tg, bg / len(hist_runs), pdg / len(hist_runs)


# --------------------------------------------------------------------------
# 6-state diagnostic plots (override the 15-state versions from metrics.py)
# --------------------------------------------------------------------------
def plot_bias_convergence(t, bg_hist, ba_hist=None, title_suffix=""):
    """Gyro-bias estimate over time. USQUE has no accel-bias state, so there is
    no second panel (ba_hist is accepted and ignored for drop-in compatibility)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    bg_deg = np.degrees(bg_hist)
    for k, (nm, col) in enumerate(zip("XYZ", "rgb")):
        ax.plot(t, bg_deg[:, k], label=f"Gyro Bias {nm}", color=col)
    ax.set_title("Gyroscope Bias Convergence (USQUE)")
    ax.set_ylabel("Bias (deg/s)")
    ax.set_xlabel("Time (s)")
    ax.grid(True)
    ax.legend(loc="upper right")
    if title_suffix:
        fig.suptitle(title_suffix)
    fig.tight_layout()
    return fig, ax


def plot_covariance_diagonals(t, P_hist, title_suffix=""):
    """Attitude, gyro-bias, and (if present) delta_psi_mag 1-sigma (from diag P) over
    time, log scale. P layout: [dtheta (0:3, body-frame GRP = rotation vector in rad),
    dbias (3:6, rad/s), dpsi_mag (6, rad, only if the mag-bias state is enabled)]."""
    import matplotlib.pyplot as plt
    P_hist = np.asarray(P_hist)
    has_psi = P_hist.shape[-1] >= 7
    fig, ax = plt.subplots(3 if has_psi else 2, 1, figsize=(11, 10 if has_psi else 7), sharex=True)
    for k, (nm, col) in enumerate(zip(("Roll", "Pitch", "Yaw"), "rgb")):
        ax[0].plot(t, np.degrees(np.sqrt(np.maximum(P_hist[:, k, k], 1e-30))),
                   label=f"sigma({nm})", color=col)
    ax[0].set_title("Attitude uncertainty  sqrt(P_att)")
    ax[0].set_ylabel("deg")
    for k, (nm, col) in enumerate(zip("XYZ", "rgb")):
        ax[1].plot(t, np.degrees(np.sqrt(np.maximum(P_hist[:, 3 + k, 3 + k], 1e-30))),
                   label=f"sigma(bg {nm})", color=col)
    ax[1].set_title("Gyro-bias uncertainty  sqrt(P_bg)")
    ax[1].set_ylabel("deg/s")
    if has_psi:
        ax[2].plot(t, np.degrees(np.sqrt(np.maximum(P_hist[:, 6, 6], 1e-30))), color='m')
        ax[2].set_title("Magnetic heading-bias uncertainty  sqrt(P_psi_mag)")
        ax[2].set_ylabel("deg")
    ax[-1].set_xlabel("Time (s)")
    for a_ in ax:
        a_.set_yscale("log")
        a_.grid(True, which="both", ls="--", alpha=0.5)
        a_.legend(loc="upper right") if a_.get_legend_handles_labels()[0] else None
    if title_suffix:
        fig.suptitle(title_suffix)
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------
# Trajectory: ground-truth path + GT vs estimated attitude
# --------------------------------------------------------------------------
def plot_attitude_trajectory(pos_gt, quat_gt, quat_est, n_arrows=40, title_suffix=""):
    """USQUE estimates no position, so the estimate-vs-GT 'trajectory' is the
    GT flight path with GT (blue) and estimated (orange) body-forward arrows
    drawn along it. Left: 3D (N/E/Up). Right: top-down with heading arrows,
    where a yaw error is easiest to read. Arrow pairs are evenly spaced."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pos = np.asarray(pos_gt, float)
    n = len(pos)
    idx = np.unique(np.linspace(0, n - 1, n_arrows).astype(int))
    span = max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1]), np.ptp(pos[:, 2]), 1.0)
    L = 0.06 * span

    def fwd(qq):
        return q.to_rotation_matrix(qq)[:, 0]      # body-x (forward) in NED

    gt_dir = np.array([fwd(quat_gt[i]) for i in idx])
    es_dir = np.array([fwd(quat_est[i]) for i in idx])
    P = pos[idx]

    fig = plt.figure(figsize=(14, 6))
    ax3 = fig.add_subplot(121, projection='3d')
    ax3.plot(pos[:, 0], pos[:, 1], -pos[:, 2], color='gray', lw=1, alpha=0.6, label='GT path')
    ax3.quiver(P[:, 0], P[:, 1], -P[:, 2], gt_dir[:, 0], gt_dir[:, 1], -gt_dir[:, 2],
               length=L, color='dodgerblue', label='GT forward axis')
    ax3.quiver(P[:, 0], P[:, 1], -P[:, 2], es_dir[:, 0], es_dir[:, 1], -es_dir[:, 2],
               length=L, color='darkorange', label='USQUE forward axis')
    ax3.set_xlabel('N [m]'); ax3.set_ylabel('E [m]'); ax3.set_zlabel('Up [m]')
    r = span / 2.0
    c = pos.mean(axis=0)
    ax3.set_xlim(c[0] - r, c[0] + r); ax3.set_ylim(c[1] - r, c[1] + r)
    ax3.set_zlim(-c[2] - r, -c[2] + r)
    ax3.set_title('3D: GT path + attitude (forward axis)')
    ax3.legend(loc='upper left')

    ax2 = fig.add_subplot(122)
    ax2.plot(pos[:, 1], pos[:, 0], color='gray', lw=1, alpha=0.6)
    ax2.quiver(P[:, 1], P[:, 0], gt_dir[:, 1], gt_dir[:, 0], color='dodgerblue',
               angles='xy', scale_units='xy', scale=1.0 / L, width=0.004, label='GT heading')
    ax2.quiver(P[:, 1], P[:, 0], es_dir[:, 1], es_dir[:, 0], color='darkorange',
               angles='xy', scale_units='xy', scale=1.0 / L, width=0.004, label='USQUE heading')
    ax2.set_xlabel('East [m]'); ax2.set_ylabel('North [m]')
    ax2.set_aspect('equal', adjustable='datalim')
    ax2.grid(True)
    ax2.set_title('Top-down: heading GT vs estimate')
    ax2.legend(loc='best')
    fig.suptitle('Flight path with attitude: ground truth vs USQUE estimate'
                 + (f'   [{title_suffix}]' if title_suffix else ''))
    fig.tight_layout()
    return fig, (ax3, ax2)


# --------------------------------------------------------------------------
# Honest reference handling
# --------------------------------------------------------------------------
# The roll/pitch "ground truth" outdoors (Wahba) is 7-8 deg off, and the
# accel-tilt reference is not independent of a filter that fuses the
# accelerometer every step. The helpers below make that explicit instead of
# printing numbers that look better (or worse) than the evidence supports.
def pooled_gt_own_error(ar_runs):
    """Pooled RMS (deg) of the Wahba-GT's OWN roll/pitch error, measured against the
    accel-tilt reference at every still window of every flight (per-window arrays,
    so flights with more windows weigh more). Returns (roll, pitch, n_windows) or None."""
    r, p = [], []
    for d in ar_runs:
        if d and d.get("gt_d_roll_vs_accel"):
            r += list(d["gt_d_roll_vs_accel"])
            p += list(d["gt_d_pitch_vs_accel"])
    if not r:
        return None
    return (float(np.sqrt(np.mean(np.square(r)))), float(np.sqrt(np.mean(np.square(p)))), len(r))


def nees_series_ref(q_est, q_true, P_hist, sigma_ref_deg):
    """NEES against a NOISY reference: theta' (P + P_ref)^-1 theta with
    P_ref = diag(sigma_ref^2) (roll, pitch, yaw; deg). Using P alone charges the
    reference's own noise to the filter, which is why ANEES against the Wahba GT
    reads in the hundreds. P_ref must be ESTIMATED (pooled_gt_own_error for
    roll/pitch, RTK sigma for yaw); errors of filter and reference are positively
    correlated (both use the magnetometer), so this is somewhat OPTIMISTIC."""
    Pref = np.diag(np.radians(np.asarray(sigma_ref_deg, float)) ** 2)
    out = np.zeros(len(q_est))
    for i in range(len(q_est)):
        theta, P_theta = _error_and_cov(q_est[i], q_true[i], np.asarray(P_hist[i])[0:3, 0:3])
        out[i] = theta @ np.linalg.solve(P_theta + Pref, theta)
    return out


def error_bounds(axis_err_runs, ar_runs, own=None):
    """Bracket the true roll/pitch error using two references of different kinds.
      upper : raw RMS vs the Wahba GT (includes the GT's own error)
      lower : sqrt(raw^2 - gt_own^2) -- quadrature subtraction. Exact only if the
              filter and GT errors are independent; they are positively correlated
              (shared magnetometer) so the TRUE error is >= this.
    Returns dict(roll=(lo, hi), pitch=(lo, hi), gt_own=(r, p, n)) or None."""
    own = own if own is not None else pooled_gt_own_error(ar_runs)
    if own is None:
        return None
    hi_r = float(np.sqrt(np.nanmean(np.concatenate([e[:, 0] for e in axis_err_runs]) ** 2)))
    hi_p = float(np.sqrt(np.nanmean(np.concatenate([e[:, 1] for e in axis_err_runs]) ** 2)))
    lo = lambda hi, g: float(np.sqrt(max(hi ** 2 - g ** 2, 0.0)))
    return dict(roll=(lo(hi_r, own[0]), hi_r), pitch=(lo(hi_p, own[1]), hi_p), gt_own=own)


def gyro_consistency_multi(t_imu, q_est, gyro_raw, bg_hist, wins=(2.0, 5.0, 15.0, 30.0)):
    """Gyro-integration cross-check at several window lengths. For each window:
    dead-reckon the bias-corrected gyro from the filter's own attitude at the window
    start and compare the roll/pitch INCREMENT with the filter's. This measures how far
    the filter's accel/mag corrections pull away from pure gyro propagation.
      grows with window -> real drift the corrections keep removing (error ~ that level)
      flat              -> the check's own noise floor: error is BELOW it, unresolved.
    Returns {win_s: (roll_rmse_deg, pitch_rmse_deg, n_windows)}."""
    import attitude_reference as _ar
    t_imu = np.asarray(t_imu)
    fs = 1.0 / np.median(np.diff(t_imu))
    dt = np.diff(t_imu, prepend=t_imu[0] - 1.0 / fs)
    gcorr = np.asarray(gyro_raw) - np.asarray(bg_hist)
    out = {}
    for w in wins:
        win = int(w * fs)
        stride = max(1, win)                      # non-overlapping: bounded cost
        r_res, p_res = [], []
        for s in range(0, len(t_imu) - win - 1, stride):
            e = s + win
            q_pred = _ar._gyro_integrate(q_est[s], gcorr[s:e], dt[s:e])
            rp0 = np.array(_ar._euler_deg(q_est[s])[:2])
            df = (np.array(_ar._euler_deg(q_est[e])[:2]) - rp0 + 180) % 360 - 180
            dp = (np.array(_ar._euler_deg(q_pred)[:2]) - rp0 + 180) % 360 - 180
            r_res.append(df[0] - dp[0])
            p_res.append(df[1] - dp[1])
        if r_res:
            out[w] = (float(np.sqrt(np.mean(np.square(r_res)))),
                      float(np.sqrt(np.mean(np.square(p_res)))), len(r_res))
    return out


# still-detector presets for the accel-tilt sensitivity table
STILL_PRESETS = {
    "tight":   dict(w_thresh=0.025, a_dev_thresh=0.20, a_std_thresh=0.25, min_dur=0.50),
    "default": dict(),
    "loose":   dict(w_thresh=0.06, a_dev_thresh=0.50, a_std_thresh=0.50, min_dur=0.25),
    "looser":  dict(w_thresh=0.10, a_dev_thresh=0.80, a_std_thresh=0.80, min_dur=0.20),
}


def accel_tilt_sensitivity(ar_module, t_imu, q_est, gyro, accel, bg_hist, P_hist, exclude_init_s,
                           t_gt=None, quat_gt=None):
    """Re-run the accel-tilt reference under several still-detector thresholds and
    return {preset: dict(n, roll, pitch, floor, gt_d_roll_vs_accel, gt_d_pitch_vs_accel)}. If the value is not
    stable across presets (or the window count is tiny) it is a sanity check only."""
    out = {}
    for name, kw in STILL_PRESETS.items():
        try:
            d = ar_module.attitude_reference_report(
                t_imu, q_est, gyro, accel, bg_hist, t_gt=t_gt, quat_gt=quat_gt, P_hist=P_hist,
                exclude_init_s=exclude_init_s,
                still_kwargs=kw, quiet=True, gyro_win_s=1e9)      # huge window: skip gyro block
        except Exception:                                              # noqa: BLE001
            continue
        if d and "filter_roll_rmse_vs_accel" in d:
            out[name] = dict(n=d["n_static_windows"], roll=d["filter_roll_rmse_vs_accel"],
                             pitch=d["filter_pitch_rmse_vs_accel"], floor=d.get("ref_floor_deg", float("nan")),
                             gt_d_roll_vs_accel=d.get("gt_d_roll_vs_accel", []),
                             gt_d_pitch_vs_accel=d.get("gt_d_pitch_vs_accel", []))
        else:
            out[name] = dict(n=(d.get("n_static_windows", 0) if d else 0), roll=float("nan"),
                             pitch=float("nan"), floor=float("nan"),
                             gt_d_roll_vs_accel=[], gt_d_pitch_vs_accel=[])
    return out


def plot_batch_error_bounds(names, bounds_rows, floor_deg=None, title_suffix=""):
    """Replaces the old 'accel-tilt bar next to the GT bar' chart, which put a
    0.09 deg bar (no resolving power, self-comparison) beside a 9 deg bar.
    Per flight and axis (roll, pitch): raw error vs the Wahba GT (upper bound),
    the quadrature-subtracted estimate (lower bound) and the gyro-integration
    cross-check (independent, drift-only). The accel-tilt reference floor is the
    shaded band: anything below it is unresolved by that instrument.
    bounds_rows: list of dicts(raw_r, raw_p, lb_r, lb_p, gyro_r, gyro_p)."""
    import matplotlib.pyplot as plt
    n = len(names)
    x = np.arange(n)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for k, (axn, key) in enumerate((("Roll", "r"), ("Pitch", "p"))):
        w = 0.26
        ax[k].bar(x - w, [b["raw_" + key] for b in bounds_rows], w, color="lightgray",
                  edgecolor="k", label="raw vs Wahba GT (upper bound)")
        ax[k].bar(x, [b["lb_" + key] for b in bounds_rows], w, color="tab:orange",
                  label="quadrature-subtracted (lower bound)")
        ax[k].bar(x + w, [b["gyro_" + key] for b in bounds_rows], w, color="tab:green",
                  label="gyro cross-check (independent, drift only)")
        if floor_deg is not None and np.isfinite(floor_deg):
            ax[k].axhspan(0, floor_deg, color="tab:blue", alpha=0.25,
                          label=f"accel-tilt reference floor ~{floor_deg:.2f} deg")
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(names, rotation=45, ha="right")
        ax[k].set_title(f"{axn} error: bounded by independent references")
        ax[k].grid(axis="y", ls="--", alpha=0.6)
    ax[0].set_ylabel("deg")
    ax[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Roll/pitch error bounds" + (f"  [{title_suffix}]" if title_suffix else ""))
    fig.tight_layout()
    return fig, ax


# --------------------------------------------------------------------------
# Approximate TRUE error
# --------------------------------------------------------------------------
def approx_true_error(raw_rmse, gt_own_rmse, rho_hi=0.5):
    """Estimate the filter's own RMS error e from the RMS error vs a noisy reference (raw)
    and the reference's own RMS error g:   raw^2 = e^2 + g^2 - 2 rho e g.
      rho = 0        -> e = sqrt(raw^2 - g^2)                (lower bound)
      rho = rho_hi   -> e = rho g + sqrt(rho^2 g^2 - g^2 + raw^2)
    rho > 0 because filter and Wahba GT both use the magnetometer. Returns (e_lo, e_hi, unresolved)
    where unresolved=True means g >= raw: the reference noise alone explains the discrepancy, so
    the filter error cannot be resolved (reported as 0 .. e_hi)."""
    g, R = float(gt_own_rmse), float(raw_rmse)
    unresolved = g >= R
    e_lo = 0.0 if unresolved else float(np.sqrt(R * R - g * g))
    disc = rho_hi ** 2 * g * g - g * g + R * R
    e_hi = float(rho_hi * g + np.sqrt(max(disc, 0.0)))
    return e_lo, e_hi, unresolved


def plot_batch_true_error(names, rows, rho_hi=0.5, title_suffix=""):
    """Per-flight APPROXIMATE TRUE error of the filter (RMSE, deg), not the raw error vs a noisy
    reference. Bars = point estimate (filter/reference errors independent); whisker = the same
    estimate if the errors are correlated with rho=rho_hi (shared magnetometer); grey outline = raw
    RMSE vs the reference; diamond = independent gyro-integration check (drift only).
    Roll/pitch reference = Wahba GT (own error subtracted). Yaw reference = dual-RTK heading
    (RTK noise subtracted). Flights where the reference noise >= the raw error are hatched:
    the filter error there is unresolved (0 .. whisker).
    rows: list of dict(raw_r, raw_p, raw_y, est_r, est_p, est_y, hi_r, hi_p, hi_y,
                       unres_r, unres_p, gyro_r, gyro_p)."""
    import matplotlib.pyplot as plt
    n = len(names)
    x = np.arange(n)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for k, (axn, key) in enumerate((("Roll", "r"), ("Pitch", "p"), ("Yaw", "y"))):
        raw = [b["raw_" + key] for b in rows]
        est = np.array([b["est_" + key] for b in rows])
        hi = np.array([b["hi_" + key] for b in rows])
        unres = [b.get("unres_" + key, False) for b in rows]
        ax[k].bar(x, raw, 0.75, fill=False, edgecolor="gray", ls="--",
                  label="raw RMSE vs reference")
        bars = ax[k].bar(x, est, 0.45, color="tab:orange", label="approx. true error (independent errors)")
        for bi, u in zip(bars, unres):
            if u:
                bi.set_hatch("//")
                bi.set_facecolor("none")
                bi.set_edgecolor("tab:orange")
        ax[k].errorbar(x, est, yerr=[np.zeros(n), np.maximum(hi - est, 0)], fmt="none",
                       ecolor="k", capsize=4, label=f"if errors correlated (rho={rho_hi})")
        if key in ("r", "p"):
            gy = [b["gyro_" + key] for b in rows]
            ax[k].plot(x, gy, "D", color="tab:green", ms=8, mec="k",
                       label="gyro cross-check (independent, drift only)")
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(names, rotation=45, ha="right")
        ax[k].set_title(f"{axn}")
        ax[k].grid(axis="y", ls="--", alpha=0.6)
    ax[0].set_ylabel("RMSE (deg)")
    ax[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Approximate TRUE filter error (reference noise removed)"
                 + (f"  [{title_suffix}]" if title_suffix else ""))
    fig.tight_layout()
    return fig, ax
