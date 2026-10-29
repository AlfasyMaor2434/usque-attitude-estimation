"""
attitude_reference.py
---------------------
Independent roll/pitch accuracy checks for the mars flights.

The INSANE outdoor ground truth (`ground_truth_*.csv`) gets its yaw from the
dual-RTK baseline (~1 deg, trustworthy) but its roll/pitch from
`mag_gps_wahba` -- a nearly horizontal RTK baseline weighted x50 plus the
~18%-distorted site magnetometer through cross products. Verified against
the raw gyro: GT roll/pitch-rate has ~0 correlation with the true gyro and
+-85..285 deg/s of noise. So "filter Roll RMSE vs GT" mostly measures GT
noise, not filter error.

Two references here do NOT use the Wahba GT:

  A. Quasi-static accelerometer tilt.  When |omega| ~ 0 and |f| ~ g the
     accelerometer specific force IS the gravity direction, giving true
     roll/pitch to ~0.5 deg (yaw unobservable). mars flights have short
     hover / on-ground segments scattered through the flight; the filter
     may have drifted since its last gravity update at those moments.

  B. Gyro-integration consistency.  Over short windows, dead-reckon the
     bias-corrected raw gyro from the filter's own attitude and compare
     the roll/pitch INCREMENT to the filter's. Checks the updates are not
     yanking roll/pitch against the gyro. Consistency, not accuracy
     (the gyro drifts), but < ~0.5 deg over 10-20 s.

Body frame: FRD. Attitude quaternion: body->nav (R_nb), Hamilton wxyz.
Static specific force in FRD is f_b = g * [ sin(th), -sin(ph)cos(th),
-cos(ph)cos(th) ]  =>  roll = atan2(-f_y, -f_z), pitch = atan2(f_x,
hypot(f_y, f_z)).  `sign_check` re-derives the sign from the filter at the
static windows themselves (where the filter is freshest) and flips the
formula if the dataset's Euler handedness disagrees.
"""
import numpy as np
import project_utils as pu
from scipy.spatial.transform import Rotation as _Rot

_G = 9.81


def _euler_deg(q_wxyz):
    """(roll, pitch, yaw) in degrees from a body->nav quaternion."""
    return np.degrees(pu.to_euler(q_wxyz))


def _wrap180(a):
    """Angular difference(s) wrapped to (-180, 180] degrees."""
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def find_quasi_static_windows(t, gyro_frd, accel_frd, *,
                              w_thresh=0.035, a_dev_thresh=0.30,
                              a_std_thresh=0.35, min_dur=0.35,
                              max_tilt_deg=20.0):
    """Index spans (i0, i1) where the vehicle is close to stationary AND
    in a plausible flight/rest attitude:
      per-sample  |omega| < w_thresh  and  ||accel| - g| < a_dev_thresh,
      held for >= min_dur s, low accel std, AND the accel-tilt magnitude
      (|roll|, |pitch|) < max_tilt_deg.
    The tilt gate rejects post-landing / handling windows: after touchdown
    the crew picks the vehicle up or lays it on its side, which also reads
    |omega|~0 and |accel|~g but with a 90-180 deg tilt -- a static reading
    of a NOT-flying attitude, useless as a hover reference (this is what
    produced the bogus 70-109 deg "GT error" on mars_3)."""
    t = np.asarray(t)
    wn = np.linalg.norm(gyro_frd, axis=1)
    an = np.linalg.norm(accel_frd, axis=1)
    ok = (wn < w_thresh) & (np.abs(an - _G) < a_dev_thresh)

    fs = 1.0 / np.median(np.diff(t))
    min_len = max(3, int(min_dur * fs))

    spans, i, n = [], 0, len(ok)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        if j - i >= min_len:
            seg = accel_frd[i:j]
            if np.max(np.std(seg, axis=0)) < a_std_thresh:
                r, p = _accel_tilt_deg(seg.mean(0))
                if abs(r) < max_tilt_deg and abs(p) < max_tilt_deg:
                    spans.append((i, j))
        i = j
    return spans


def _accel_tilt_deg(f_mean, flip_pitch=False, flip_roll=False):
    f = f_mean / (np.linalg.norm(f_mean) + 1e-12) * _G
    roll = np.degrees(np.arctan2(-f[1], -f[2]))
    pitch = np.degrees(np.arctan2(f[0], np.hypot(f[1], f[2])))
    if flip_roll:
        roll = -roll
    if flip_pitch:
        pitch = -pitch
    return roll, pitch


def _gyro_integrate(q0_wxyz, gyro_frd, dt):
    """Dead-reckon body->nav quaternion over a block of body-rate samples."""
    q = np.asarray(q0_wxyz, float)
    for k in range(len(gyro_frd)):
        q = pu.normalize(pu.multiply(q, pu.from_axis_angle(gyro_frd[k] * dt[k])))
    return q


def attitude_reference_report(t_imu, q_est, gyro_frd_raw, accel_frd_raw,
                              bg_hist, t_gt=None, quat_gt=None, P_hist=None, *,
                              ba_hist=None, grav_fire_times=None,
                              exclude_init_s=0.0, exclude_grav_margin_s=2.0,
                              gyro_win_s=15.0, gyro_stride_s=10.0, label="",
                              still_kwargs=None, quiet=False):
    """Print the independent roll/pitch block and return a metrics dict.

    t_imu          (M,)     seconds
    q_est          (M,4)    filter attitude, body->nav, wxyz
    gyro_frd_raw   (M,3)    RAW gyro, FRD  (rad/s)  -- NOT bias-corrected
    accel_frd_raw  (M,3)    RAW accel, FRD (m/s^2)
    bg_hist        (M,3)    filter gyro-bias estimate history (rad/s)
    t_gt, quat_gt  optional -- also score the Wahba-GT vs accel-tilt (contrast)
    P_hist         (M,15,15) optional -- enables NEES vs accel-tilt at the
                   still windows (P[0,0]=roll var, P[1,1]=pitch var, rad^2)
    ba_hist        (M,3) optional -- filter accel-bias history, for the
                   per-window reference uncertainty budget (task 4.2)
    grav_fire_times  list of abs t of grav_update firings; windows within
                   exclude_grav_margin_s of any are dropped (task 4.2)
    exclude_init_s   drop windows starting within this many s of t_imu[0]
                   (they score the TRIAD init against itself)
    """
    t_imu = np.asarray(t_imu)
    dt = np.diff(t_imu, prepend=t_imu[0] - np.median(np.diff(t_imu)))

    # ---------- A. quasi-static accel-tilt ----------
    spans_all = find_quasi_static_windows(t_imu, gyro_frd_raw, accel_frd_raw,
                                          **(still_kwargs or {}))

    # task 4.2: drop windows contaminated by the initialization or by a
    # grav_update firing -- otherwise the accel-tilt "error" is partly the
    # filter being scored against a signal it was just pulled onto.
    _gft = np.asarray(grav_fire_times, dtype=float) if grav_fire_times is not None else np.array([])
    _t0 = float(t_imu[0])
    spans, _n_excl_init, _n_excl_grav = [], 0, 0
    for (i0, i1) in spans_all:
        # spans are half-open [i0, i1); a window that runs to the END of the record has
        # i1 == len(t_imu), so index the last sample INSIDE it (was t_imu[i1] -> IndexError,
        # which silently dropped every flight that ends in a still period).
        ta0, ta1 = float(t_imu[i0]), float(t_imu[i1 - 1])
        if exclude_init_s > 0.0 and ta0 <= _t0 + exclude_init_s:
            _n_excl_init += 1
            continue
        if _gft.size and np.any((_gft >= ta0 - exclude_grav_margin_s) &
                                (_gft <= ta1 + exclude_grav_margin_s)):
            _n_excl_grav += 1
            continue
        spans.append((i0, i1))

    out = {"n_static_windows": len(spans),
           "n_static_windows_raw": len(spans_all),
           "n_excluded_init": _n_excl_init,
           "n_excluded_grav": _n_excl_grav}

    if spans:
        # sign calibration: at the static windows the filter has just been
        # gravity-updated, so filter tilt ~ true tilt there. Pick the
        # flip combo (of 4) that minimises |filter - accel| across windows.
        centres = [(i + j) // 2 for i, j in spans]
        f_means = [accel_frd_raw[i:j].mean(0) for i, j in spans]
        filt_rp = np.array([_euler_deg(q_est[c])[:2] for c in centres])

        best, best_err = (False, False), np.inf
        for fr in (False, True):
            for fp in (False, True):
                rp = np.array([_accel_tilt_deg(fm, flip_pitch=fp, flip_roll=fr)
                               for fm in f_means])
                e = np.sqrt(np.mean(_wrap180(rp - filt_rp) ** 2))
                if e < best_err:
                    best_err, best = e, (fr, fp)
        flip_roll, flip_pitch = best

        acc_rp = np.array([_accel_tilt_deg(fm, flip_pitch=flip_pitch,
                                           flip_roll=flip_roll) for fm in f_means])
        # window centre abs-times + the accel-tilt roll/pitch there, for
        # overlaying markers on plot_euler_comparison.
        out["accel_tilt_t"] = [float(t_imu[c]) for c in centres]
        out["accel_tilt_roll"] = acc_rp[:, 0].tolist()
        out["accel_tilt_pitch"] = acc_rp[:, 1].tolist()
        d_roll = _wrap180(acc_rp[:, 0] - filt_rp[:, 0])
        d_pitch = _wrap180(acc_rp[:, 1] - filt_rp[:, 1])
        out["filter_roll_rmse_vs_accel"] = float(np.sqrt(np.mean(d_roll ** 2)))
        out["filter_pitch_rmse_vs_accel"] = float(np.sqrt(np.mean(d_pitch ** 2)))
        out["filter_roll_mae_vs_accel"] = float(np.mean(np.abs(d_roll)))
        out["filter_pitch_mae_vs_accel"] = float(np.mean(np.abs(d_pitch)))
        out["filter_roll_max_vs_accel"] = float(np.max(np.abs(d_roll)))
        out["filter_pitch_max_vs_accel"] = float(np.max(np.abs(d_pitch)))
        # SIGNED mean residual (accel-tilt - filter), deg. Consistent sign
        # across windows/flights => a systematic lean (bias-like, fixable
        # at its source); sign that averages toward 0 with |mean| << rmse
        # => scatter (needs covariance work, not bias work). Positive =
        # filter under-reads that axis vs the true tilt.
        out["filter_roll_bias_vs_accel"] = float(np.mean(d_roll))
        out["filter_pitch_bias_vs_accel"] = float(np.mean(d_pitch))
        # per-window signed residuals (accel-tilt - filter), deg -- pooled
        # across flights for the accel-tilt CDF / per-flight RMSE plots.
        out["d_roll_vs_accel"] = d_roll.tolist()
        out["d_pitch_vs_accel"] = d_pitch.tolist()
        out["sign_flip_roll"] = flip_roll
        out["sign_flip_pitch"] = flip_pitch

        # ---- per-window reference uncertainty budget (task 4.2) ----
        # The accel-tilt reference uses RAW specific force, so:
        #   * a filter/sensor accel bias b_a directly tilts the reference by
        #     ~atan(||b_a|| / g)   (bias floor)
        #   * within-window accel noise adds ~atan(std / g)   (noise floor)
        #   * residual linear accel shows as ||mean|a|| - g|.
        # Report all three so "0.15 deg" can be read as "0.15 +/- floor".
        _budget = []
        for wi, (i0, i1) in enumerate(spans):
            seg = accel_frd_raw[i0:i1]
            a_std = float(np.max(np.std(seg, axis=0)))
            a_dev = float(abs(np.linalg.norm(seg.mean(0)) - _G))
            ba_mag = (float(np.linalg.norm(ba_hist[centres[wi]]))
                      if ba_hist is not None else float('nan'))
            bias_floor = np.degrees(np.arctan2(ba_mag, _G)) if np.isfinite(ba_mag) else float('nan')
            noise_floor = np.degrees(np.arctan2(a_std, _G))
            comb = (np.degrees(np.arctan2((0.0 if not np.isfinite(ba_mag) else ba_mag) + a_std, _G)))
            _budget.append(dict(a_std=a_std, a_dev=a_dev, ba_mag=ba_mag,
                                bias_floor=bias_floor, noise_floor=noise_floor,
                                ref_floor=comb))
        out["ref_budget_pw"] = _budget
        out["ref_floor_deg"] = float(np.nanmax([b["ref_floor"] for b in _budget]))
        out["ref_bias_floor_deg"] = float(np.nanmax([b["bias_floor"] for b in _budget]))

        # NEES vs accel-tilt at the still windows (dof=1 per axis, ideal~1):
        # (filter_error_deg -> rad)^2 / P_axis_var.  This is the ONE attitude
        # NEES not contaminated by Wahba-GT roll/pitch noise.
        if P_hist is not None:
            vr = np.array([max(P_hist[c][0, 0], 1e-12) for c in centres])
            vp = np.array([max(P_hist[c][1, 1], 1e-12) for c in centres])
            _rn_pw = np.deg2rad(d_roll) ** 2 / vr
            _pn_pw = np.deg2rad(d_pitch) ** 2 / vp
            out["filter_roll_nees_vs_accel"] = float(np.mean(_rn_pw))
            out["filter_pitch_nees_vs_accel"] = float(np.mean(_pn_pw))
            # per-window arrays, for overlaying on the NEES-over-time plot
            out["roll_nees_pw"] = _rn_pw.tolist()
            out["pitch_nees_pw"] = _pn_pw.tolist()

        if t_gt is not None and quat_gt is not None:
            t_gt = np.asarray(t_gt)
            gt_rp, gt_yaw = [], []
            for c in centres:
                k = int(np.clip(np.searchsorted(t_gt, t_imu[c]), 0, len(t_gt) - 1))
                e3 = _euler_deg(quat_gt[k])
                gt_rp.append(e3[:2])
                gt_yaw.append(e3[2])
            gt_rp = np.array(gt_rp)
            gd_roll = _wrap180(gt_rp[:, 0] - acc_rp[:, 0])
            gd_pitch = _wrap180(gt_rp[:, 1] - acc_rp[:, 1])
            out["gt_roll_rmse_vs_accel"] = float(np.sqrt(np.mean(gd_roll ** 2)))
            out["gt_pitch_rmse_vs_accel"] = float(np.sqrt(np.mean(gd_pitch ** 2)))
            out["gt_roll_mae_vs_accel"] = float(np.mean(np.abs(gd_roll)))
            out["gt_pitch_mae_vs_accel"] = float(np.mean(np.abs(gd_pitch)))
            out["gt_roll_max_vs_accel"] = float(np.max(np.abs(gd_roll)))
            out["gt_pitch_max_vs_accel"] = float(np.max(np.abs(gd_pitch)))
            # per-window Wahba-GT residual vs the same accel-tilt reference
            # -- the GT-noise floor drawn behind the filter curves.
            out["gt_d_roll_vs_accel"] = gd_roll.tolist()
            out["gt_d_pitch_vs_accel"] = gd_pitch.tolist()

            # qAE with roll/pitch from accel-tilt but YAW from the (trusted)
            # Wahba heading -- the total-orientation-error analogue of the
            # standard qAE, at the still windows. per-window angle between
            # q_est and q_ref(accel_roll, accel_pitch, gt_yaw).
            qerr = []
            for wi, c in enumerate(centres):
                qr = _Rot.from_euler('xyz', [np.deg2rad(acc_rp[wi, 0]),
                                             np.deg2rad(acc_rp[wi, 1]),
                                             np.deg2rad(gt_yaw[wi])]).as_quat()
                qr_w = np.array([qr[3], qr[0], qr[1], qr[2]])
                d = abs(float(np.dot(pu.normalize(q_est[c]), pu.normalize(qr_w))))
                qerr.append(np.degrees(2.0 * np.arccos(min(1.0, d))))
            out["qae_accelref"] = float(np.mean(qerr))
            out["qae_accelref_pw"] = [float(x) for x in qerr]   # per-window

    # ---------- B. gyro-integration consistency ----------
    fs = 1.0 / np.median(np.diff(t_imu))
    win = int(gyro_win_s * fs)
    stride = max(1, int(gyro_stride_s * fs))
    gyro_corr = gyro_frd_raw - bg_hist
    r_res, p_res = [], []
    for s in range(0, len(t_imu) - win, stride):
        e = s + win
        q_pred = _gyro_integrate(q_est[s], gyro_corr[s:e], dt[s:e])
        rp0 = np.array(_euler_deg(q_est[s])[:2])
        rp1_f = np.array(_euler_deg(q_est[e])[:2])
        rp1_p = np.array(_euler_deg(q_pred)[:2])
        # unwrap the increment near +-180
        df = (rp1_f - rp0 + 180) % 360 - 180
        dp = (rp1_p - rp0 + 180) % 360 - 180
        r_res.append(df[0] - dp[0])
        p_res.append(df[1] - dp[1])
    if r_res:
        out["n_gyro_windows"] = len(r_res)
        out["filter_roll_gyro_consistency_rmse"] = float(np.sqrt(np.mean(np.array(r_res) ** 2)))
        out["filter_pitch_gyro_consistency_rmse"] = float(np.sqrt(np.mean(np.array(p_res) ** 2)))

    if not quiet:
        _print_block(out, label)
    return out


def _print_block(o, label):
    """Compact per-flight line. The full two-reference table is printed
    once at the end of the run by print_evaluation_report(accel_ref=...)."""
    tag = f" [{label}]" if label else ""
    n = o.get("n_static_windows", 0)
    _nraw = o.get("n_static_windows_raw", n)
    _exi, _exg = o.get("n_excluded_init", 0), o.get("n_excluded_grav", 0)
    _exstr = (f"  [{_nraw} raw, excluded {_exi} init / {_exg} grav_update]"
              if (_exi or _exg) else "")
    if n and "filter_roll_rmse_vs_accel" in o:
        fl = [a for a in ("roll", "pitch") if o.get(f"sign_flip_{a}")]
        flt = f"  (euler sign-flip: {','.join(fl)})" if fl else ""
        _ff = o.get("ref_floor_deg", float('nan'))
        print(f"  accel-tilt ref{tag}: {n} still-window(s){_exstr}{flt} -- "
              f"filter Roll/Pitch vs accel-tilt: "
              f"RMSE {o['filter_roll_rmse_vs_accel']:.2f}/{o['filter_pitch_rmse_vs_accel']:.2f}  "
              f"(reference floor ~{_ff:.2f} deg)  "
              f"bias(signed) {o.get('filter_roll_bias_vs_accel', float('nan')):+.2f}/"
              f"{o.get('filter_pitch_bias_vs_accel', float('nan')):+.2f}  "
              f"NEES {o.get('filter_roll_nees_vs_accel', float('nan')):.1f}/"
              f"{o.get('filter_pitch_nees_vs_accel', float('nan')):.1f}  "
              f"(Wahba-GT vs accel-tilt: "
              f"{o.get('gt_roll_rmse_vs_accel', float('nan')):.2f}/"
              f"{o.get('gt_pitch_rmse_vs_accel', float('nan')):.2f})")
        # per-window detail -- an aggregate RMSE over 2-5 windows can be
        # dominated by a single outlier (e.g. one window sitting inside an
        # RTK/heading dropout on the transition flights); this line lets
        # you see WHICH window and WHEN (abs time, cross-reference against
        # the trajectory/RTK gaps) instead of only the mixed-in aggregate.
        ta = o.get("accel_tilt_t", [])
        dr = o.get("d_roll_vs_accel", [])
        dp = o.get("d_pitch_vs_accel", [])
        rn = o.get("roll_nees_pw", [None] * len(ta))
        pn = o.get("pitch_nees_pw", [None] * len(ta))
        bud = o.get("ref_budget_pw", [None] * len(ta))
        for i, t in enumerate(ta):
            _rn = f"{rn[i]:.1f}" if i < len(rn) and rn[i] is not None else "n/a"
            _pn = f"{pn[i]:.1f}" if i < len(pn) and pn[i] is not None else "n/a"
            _b = bud[i] if i < len(bud) and bud[i] is not None else None
            _bs = (f"  |  ref-budget: a_std={_b['a_std']:.3f} ||a|-g|={_b['a_dev']:.3f} "
                   f"|b_a|={_b['ba_mag']:.3f} floor~{_b['ref_floor']:.2f}deg" if _b else "")
            print(f"    window {i + 1}  t={t:.1f}  "
                  f"d_roll={dr[i]:+.2f}  d_pitch={dp[i]:+.2f}  NEES(R/P)={_rn}/{_pn}{_bs}")
    else:
        print(f"  accel-tilt ref{tag}: only {n} still-window(s){_exstr} -- too few for an estimate")
    if "filter_roll_gyro_consistency_rmse" in o:
        print(f"  gyro-integration consistency ({o['n_gyro_windows']} win): "
              f"Roll {o['filter_roll_gyro_consistency_rmse']:.2f}  "
              f"Pitch {o['filter_pitch_gyro_consistency_rmse']:.2f} deg  (drift-bounded cross-check)")
