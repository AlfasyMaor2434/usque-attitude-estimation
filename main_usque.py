"""
main_usque.py
-------------
Stage 3 for the pure-AHRS USQUE filter (gyro + accel + mag; no GPS, no INS):
run the filter on the INSANE flights, score ACCURACY and CONSISTENCY, plot.

Sites (FLIGHT_SITE): 'mars' | 'transition' | 'outdoor' | 'indoor'
  mars/transition/outdoor : RTK+Wahba ground truth (only YAW is trustworthy),
                            dual-RTK heading available, magnetometer on.
  indoor                  : Optitrack mocap ground truth (sub-degree, all three
                            axes), NO absolute heading and magnetometer OFF
                            (building distortion) -> accel-only updates, yaw is
                            seeded once from mocap and is a pure DRIFT metric.

Initial attitude (INIT_MODES, run back-to-back and compared):
  'mag_triad' : roll/pitch/yaw all from sensors -- TRIAD(accel, mag) against the
                site's WMM reference (inclination + declination). The magnetic
                reference used by later mag updates is then self-consistent
                with that start attitude.
  'rtk_yaw'   : TRIAD roll/pitch, but the YAW is seeded ONCE from the dual-RTK
                baseline heading, and the magnetic reference is rebuilt from
                that attitude (so the mag update damps toward the RTK heading
                instead of the ~34 deg-biased Mars magnetometer heading).
                Seed only: no RTK heading is fused afterwards, USQUE stays an
                AHRS. yaw-vs-RTK is reported as an independent diagnostic.
  indoor uses 'mocap_yaw' (roll/pitch from accel tilt, yaw from mocap at t0).

Everything is NED world / FRD body; quaternions Hamilton wxyz body->nav.
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, lfilter
from scipy.spatial.transform import Rotation as _Rot

import data_loader_USQUE as dl
import project_utils as q
import metrics_usque as mt
import attitude_reference as ar
import usque

# ======================================================================
# Run configuration
# ======================================================================
FLIGHT_SITE = 'mars'          # 'mars' | 'transition' | 'outdoor' | 'indoor'
RUN_MODE = 1                  # 0 = single flight (Monte-Carlo style plots), 1 = batch
N = 7                         # RUN_MODE=1: how many flights of the site; RUN_MODE=0: repeats
TARGET_FLIGHT = 1             # RUN_MODE=0: which flight number
from data_path import DATA_DIR as BASE_DIR     # the INSANE data set (see data_path.py)

# mars only. Flights 4 and 14 are genuine divergences (disturbed mag / missing
# data), not noise -- if you exclude them say so explicitly in the report.
SKIP_FLIGHTS = [4, 14]
INCLUDE_FLIGHTS = [1, 2, 3, 5,6,7,8]   # explicit list overrides SKIP_FLIGHTS (flight 4 is a known mag-disturbed flight)

# Initial-attitude strategies to run and compare (ignored for indoor).
INIT_MODES = ['rtk_yaw'] #'mag_triad' or 'rtk_yaw' or both
PLOT_INIT_MODE = 'rtk_yaw'    # which mode gets the full plot set
# 'rtk_yaw' seed: 0 = the single RTK heading sample at t0; > 0 = circular mean of the RTK heading
# over the leading still window, capped at this many seconds (see run_flight)
RTK_SEED_AVG_S = 0.0

# The old main_usque removed a constant world-frame offset (mean q_gt * q_est^-1
# over the first second) before scoring. That hides exactly what the two init
# modes differ in (a constant heading offset) and any mag heading bias, and
# main.py (hybrid) does not do it, so it is OFF by default.
ALIGN_TO_GT = False
ANCHOR_WINDOW_S = 1.0

# Seconds trimmed from the START / END of the scored window. Mars flights end
# with a touchdown/handling phase whose Wahba GT flips and whose RTK baseline
# glitches (non-physical, not filter error).
EVAL_TRIM_START_S = 20.0
EVAL_TRIM_END_S = 3.0
EVAL_MAX_HZ = 100.0           # decimate very dense GT (indoor mocap ~300 Hz) for scoring

# task 4.2 (same as main.py) -- accel-tilt still-windows starting within this
# many seconds of t_imu[0] score the TRIAD init against itself and are dropped.
AR_EXCLUDE_INIT_S = 5.0

# A flight's Wahba roll/pitch GT counts as VALIDATED when its own error vs the accel-tilt at still
# windows is below this (deg) on both axes, from >= 2 windows. Roll/pitch accuracy is only meaningful there.
GT_GOOD_MAX_DEG = 2.0

# Initial attitude uncertainty (1-sigma, deg, body-frame roll/pitch/yaw).
# Roll/pitch come from a 0.5 s accel average in every mode. Yaw depends on the
# source: mocap pose (~1 deg), dual-RTK baseline + lever/mount (~2 deg), or the
# magnetometer (~10 deg, and ~34 deg off at the Mars site -- see the report).
# Do NOT use a loose isotropic P0 (the old 0.1 rad^2 = 18 deg): it puts the
# sigma points ~48 deg away and lets the filter convert every tilt manoeuvre
# into a large spurious yaw correction.
INIT_SIGMA_DEG = dict(roll_pitch=2.0,
                      yaw={'mocap_yaw': 1.0, 'rtk_yaw': 2.0, 'mag_triad': 10.0})
# The initial gyro bias comes from the leading still window (accurate to a few
# 0.01 deg/s), so its prior must be tight; 1e-4 (0.57 deg/s) lets bias soak up
# unmodelled tilt/yaw error (bias z drifted to -2.6 deg/s in a test).
P0_BIAS_STATIC = 1.0e-6
P0_BIAS_NO_STATIC = 1.0e-4

# Indoor mocap boresight: {flight folder name: [roll, pitch, yaw] deg} to APPLY instead of fitting
# it on each run's own estimate. Empty = fit per run (default). A per-run fit absorbs any constant
# body-frame offset of the estimate, so comparisons between sensors must use one fixed rig value.
BORESIGHT_FIXED_RPY_DEG = {}

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_usque")

# ======================================================================
# Fusion-rate handling  (main ANEES lever)
# ----------------------------------------------------------------------
# The accel is low-passed to 15 Hz but fused at the full 200 Hz IMU rate and
# the 80 Hz magnetometer is interpolated onto the same grid, so consecutive
# measurements are heavily CORRELATED while the Kalman update treats each one
# as new information (P shrinks, NEES blows up).
#   'decimate' : predict at 200 Hz, update only at FUSION_RATE_HZ.
#   'inflate'  : update every step but scale R by the over-count factor.
#   'legacy'   : original behaviour, for A/B comparison.
# 30 Hz independent updates (accel is low-passed at 15 Hz): equal or better than fusing at 200 Hz with
# an oversample R inflation on every metric in the ablation, and its ANIS is ~6x closer to ideal.
FUSION_MODE = 'decimate'
FUSION_RATE_HZ = 30.0

# Pre-filtering: 'both' | 'accel_only' | 'gyro_only' | 'none'  (filtfilt is
# zero-phase / non-causal: offline-evaluation only.)
LPF_MODE = 'accel_only'
ACCEL_CUTOFF_HZ = 15.0
# How the accelerometer is low-passed (when LPF_MODE enables it):
#   'zero_phase' : filtfilt Butterworth -- no delay, but uses FUTURE samples (offline only; default,
#                  what the report uses)
#   'causal'     : the same Butterworth run forward only -- what a real-time system (the ESP32
#                  sketch) can do; adds a group delay of a few tens of ms
#   'boxcar'     : causal mean of the samples since the previous update (window 1 / FUSION_RATE_HZ)
ACCEL_LPF_KIND = 'zero_phase'
GYRO_CUTOFF_HZ = 15.0
VIB_PROXY_CUTOFF_HZ = 15.0

# Per-flight hard-iron mag correction (sphere fit on top of the site YAML calibration).
# See mag_bias_analysis.py for why: yaw-vs-RTK RMSE varies 1.3-18 deg across flights on
# identical hardware, consistent with per-flight payload/battery/motor distortion.
PER_FLIGHT_MAG_CALIB = True
PER_FLIGHT_MAG_CALIB_MIN_SPAN_DEG = 60.0

# ======================================================================
# Filter parameters
# ----------------------------------------------------------------------
# All noise entries are VARIANCES (hence *_var):
#   gyro_noise_var   [rad^2/s]    attitude Q = var * dt
#   bias_rw_var      [rad^2/s^3]  bias Q     = var * dt
#   accel_noise_var  [rad^2]      accel-direction measurement variance
#   mag_noise_var    [rad^2]      mag-direction measurement variance
# Starting point = physical/Allan-based values (see HANDOFF / main.py):
# sigma_g^2 + sigma_g_extra^2 ~ (3e-3)^2 for the attitude channel.
# Re-tuned after the h_a/TRIAD/mag_ref sign fixes: every number that was
# fitted before those fixes was fitted around a dead accelerometer channel.
# ======================================================================
# Final tuning (indoor_1-3 + mars_5-7): see evaluation/ (tuning_log.csv, report).
FILTER_PARAMS = dict(
    gyro_noise_var=3.0e-5,
    bias_rw_var=1.0e-10,
    accel_noise_var=2.0e-3,   # small R is safe only with the stronger PHYS_DIR inflation below
    mag_noise_var=0.05,  # was 0.5; shrunk now that delta_psi_mag absorbs the bias --
                          # chosen on the HELD-OUT test flights (9,10,11,12,13,15,16,17,18,19),
                          # not the 7 flights the rest of the params were tuned on.

    # Crassidis & Markley canonical UT: lambda = 1, Wm[0] = 1/7, spread sqrt(7)
    alpha=1.0,
    beta=2.0,
    kappa=1.0,
    gamma=1.0,                 # no synthetic fading memory

    # P0_att / P0_bias are overwritten per flight in run_flight() from
    # INIT_SIGMA_DEG / P0_BIAS_*: the initial uncertainty must reflect how the
    # start attitude was obtained (see the comment above INIT_SIGMA_DEG).
    P0_att=np.radians([2.0, 2.0, 2.0]) ** 2,
    P0_bias=1.0e-6,
    q_scale_factor=0.0,        # (1 + k*|gyro|) Q multiplier; 0 = off (tuned)

    adaptive_methods=['PHYSICAL', 'IAE', 'VIBRATION', 'MAG_GATE'],
    max_phys_inflation=20.0,
    max_vib_inflation=10.0,
    max_vib_inflation_mag=8.0,

    # Magnetic heading BIAS state (delta_psi_mag) -- absorbs a per-flight/per-site
    # magnetic heading BIAS (10-40 deg, confirmed non-random by mag_bias_analysis.py)
    # that no R value can fix. See usque.py's __init__ docstring block.
    use_mag_bias_state=True,
    mag_bias_tau_s=60.0,
    mag_bias_sigma_ss_deg=30.0,
    # Mag disturbance GATE -- catches a mid-flight TRANSIENT the slow bias state
    # doesn't react to in time (e.g. the mars_8 jump, transition's building pass).
    mag_gate_mag_frac=0.15,
    mag_gate_incl_deg=10.0,
    max_mag_gate_inflation=50.0,
    q_inflation_coupling=0.0,  # 0: R-inflation is 'this measurement is bad', not 'the state moved more'
                               # (coupling 1 fed max(accel,mag) inflation ~x2.6 into Q on every step)
    phys_mag_test=False,       # |a|-g magnitude test: no measurable benefit in the ablation
    iae_per_channel=True,      # IAE judged per accel / mag block instead of pooled over 6 dof
    vib_relative=True,         # vibration proxy = excursion above a causal 60 s baseline
    vib_rel_span=1.0,
    phys_deviation_thresh=0.05,
    phys_dir_thresh_deg=2.0,   # accel-direction test (None = off)
    vib_scale=2.0,

    use_corrected_Q=True,
    use_Q_cross_term=True,
    use_sigma0_reference=True,

    q_tune_scale=1.0,
    r_tune_scale=1.0,
)


# ======================================================================
# Helpers
# ======================================================================
def filter_accel(accel, fs):
    """Accelerometer pre-filter selected by ACCEL_LPF_KIND (see its comment)."""
    if ACCEL_LPF_KIND == 'zero_phase':
        return apply_low_pass_filter(accel, fs, ACCEL_CUTOFF_HZ)
    if ACCEL_LPF_KIND == 'causal':
        wn = ACCEL_CUTOFF_HZ / (0.5 * fs)
        b, a = butter(4, wn, btype='low', analog=False)
        # start the filter at rest on the first sample, as a real-time filter primed on it would be
        from scipy.signal import lfilter_zi
        zi = lfilter_zi(b, a)
        return np.column_stack([lfilter(b, a, accel[:, k], zi=zi * accel[0, k])[0] for k in range(accel.shape[1])])
    if ACCEL_LPF_KIND == 'boxcar':
        n = max(1, int(round(fs / FUSION_RATE_HZ)))
        c = np.cumsum(np.vstack([np.zeros((1, accel.shape[1])), accel]), axis=0)
        idx = np.arange(1, len(accel) + 1)
        lo = np.maximum(idx - n, 0)
        return (c[idx] - c[lo]) / (idx - lo)[:, None]
    raise ValueError(f"unknown ACCEL_LPF_KIND {ACCEL_LPF_KIND!r}")


def apply_low_pass_filter(data, sample_rate, cutoff_freq, order=4):
    wn = cutoff_freq / (0.5 * sample_rate)
    if not (0 < wn < 1):
        raise ValueError(f"LPF cutoff/nyquist = {wn:.3g} out of (0,1); check the sample rate.")
    b, a = butter(order, wn, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def set_quat_yaw(q_wxyz, yaw_rad):
    """q with its yaw replaced by yaw_rad, roll/pitch kept (this rotates about
    the world down axis only, so the body-frame gravity direction is preserved)."""
    roll, pitch, _ = _Rot.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_euler('xyz')
    x, y, z, w = _Rot.from_euler('xyz', [roll, pitch, yaw_rad]).as_quat()
    return np.array([w, x, y, z])


def tilt_quat_from_accel(accel, yaw_rad):
    """Roll/pitch from a specific-force sample (level at rest reads (0,0,-g)),
    yaw supplied. Used indoors, where there is no usable magnetometer."""
    fx, fy, fz = accel / np.linalg.norm(accel)
    roll = np.arctan2(-fy, -fz)
    pitch = np.arctan2(fx, np.hypot(fy, fz))
    x, y, z, w = _Rot.from_euler('xyz', [roll, pitch, yaw_rad]).as_quat()
    return np.array([w, x, y, z])


def left_matrix(quat):
    qw, qx, qy, qz = quat
    return np.array([[qw, -qx, -qy, -qz],
                     [qx, qw, -qz, qy],
                     [qy, qz, qw, -qx],
                     [qz, -qy, qx, qw]])


def world_offset(t_imu, q_est, t_gt, quat_gt, t_start, window_s):
    """Mean world-frame offset q_gt * q_est^-1 over a short window (eigenvector
    quaternion mean). Only used when ALIGN_TO_GT."""
    t_anchor = np.arange(t_start, t_start + window_s, 0.05)
    t_anchor = t_anchor[(t_anchor >= t_imu[0]) & (t_anchor <= t_imu[-1])
                        & (t_anchor >= t_gt[0]) & (t_anchor <= t_gt[-1])]
    if len(t_anchor) == 0:
        t_anchor = np.array([t_start])
    offs = []
    for ta in t_anchor:
        ii = np.clip(np.searchsorted(t_imu, ta), 0, len(t_imu) - 1)
        ig = np.clip(np.searchsorted(t_gt, ta), 0, len(t_gt) - 1)
        qe = q_est[ii] / np.linalg.norm(q_est[ii])
        qg = quat_gt[ig] / np.linalg.norm(quat_gt[ig])
        offs.append(q.multiply(qg, q.inverse(qe)))
    Mavg = np.zeros((4, 4))
    for qo in offs:
        if np.dot(qo, offs[0]) < 0:
            qo = -qo
        Mavg += np.outer(qo, qo)
    _, evecs = np.linalg.eigh(Mavg)
    qw = evecs[:, -1]
    return qw / np.linalg.norm(qw)


def site_modes(site):
    return ['mocap_yaw'] if dl.SITE_PROFILE[site]['gt'] == 'mocap' else list(INIT_MODES)


# ======================================================================
# One flight
# ======================================================================
def run_flight(dataset_dir, site, init_mode, params=None, verbose=True, want_ar=True,
               max_seconds=None, want_extras=False):
    """Run USQUE on one flight and score it. Returns a result dict (arrays are
    full-length IMU-rate histories plus the scored/trimmed comparison set)."""
    params = dict(FILTER_PARAMS if params is None else params)
    prof = dl.SITE_PROFILE[site]
    name = os.path.basename(dataset_dir)
    log = print if verbose else (lambda *a, **k: None)

    t_imu, gyro, accel, mag = dl.load_sensor_data(dataset_dir)
    t_gt, pos_gt, quat_gt = dl.load_groundtruth(dataset_dir, frame=prof['gt'])
    t_hdg, yaw_hdg, sig_hdg = (dl.load_rtk_heading(dataset_dir) if prof['heading'] == 'rtk'
                               else (None, None, None))
    use_mag = prof['mag']
    if not use_mag:
        # no magnetometer -> the heading-bias state never gets a measurement (no effect on
        # attitude, only an empty uncertainty panel), so drop it
        params['use_mag_bias_state'] = False
    if max_seconds is not None:      # truncate (used by the tuning harness)
        keep = t_imu <= t_imu[0] + max_seconds
        t_imu, gyro, accel, mag = t_imu[keep], gyro[keep], accel[keep], mag[keep]

    mag_hi_center = mag_hi_radius = None
    if use_mag and PER_FLIGHT_MAG_CALIB:
        # Residual per-flight hard-iron correction ON TOP OF the site (YAML) calibration --
        # different payload/battery/motor current per flight is a well-known multirotor
        # effect; yaw-vs-RTK RMSE varies 1.3-18 deg across flights on the SAME hardware,
        # which points at a per-flight, not a purely per-sensor/per-site, distortion (see
        # mag_bias_analysis.py). Sphere-fit needs heading spread to be well-conditioned; a
        # near-straight flight gives an unreliable fit, so skip it below a coverage floor.
        heading_span_deg = float(np.degrees(np.ptp(np.unwrap(np.arctan2(mag[:, 1], mag[:, 0])))))
        if heading_span_deg >= PER_FLIGHT_MAG_CALIB_MIN_SPAN_DEG:
            mag, mag_hi_center, mag_hi_radius = dl.per_flight_hard_iron_correct(mag)
            log(f"    [{name}] per-flight hard-iron: center={np.round(mag_hi_center, 3)} "
                f"|center|={np.linalg.norm(mag_hi_center):.3f} radius={mag_hi_radius:.3f} "
                f"(heading span {heading_span_deg:.0f} deg)")
        else:
            log(f"    [{name}] per-flight hard-iron SKIPPED: heading span only "
                f"{heading_span_deg:.0f} deg (< {PER_FLIGHT_MAG_CALIB_MIN_SPAN_DEG:.0f}), "
                f"sphere fit would be unreliable")

    dt_mean = np.mean(np.diff(t_imu))
    fs = 1.0 / dt_mean
    use_accel_lpf = LPF_MODE in ('both', 'accel_only')
    use_gyro_lpf = LPF_MODE in ('both', 'gyro_only')
    accel_in = filter_accel(accel, fs) if use_accel_lpf else accel
    gyro_in = apply_low_pass_filter(gyro, fs, GYRO_CUTOFF_HZ) if use_gyro_lpf else gyro
    accel_vib_ref = apply_low_pass_filter(accel, fs, VIB_PROXY_CUTOFF_HZ)
    mag_in = mag

    b0_static = dl.load_static_gyro_bias(dataset_dir, t_imu=t_imu, gyro=gyro, accel=accel)

    if FUSION_MODE == 'decimate':
        fusion_mask = dl.make_fusion_mask(t_imu, FUSION_RATE_HZ)
    else:
        fusion_mask = np.ones(len(t_imu), dtype=bool)
    if FUSION_MODE == 'inflate':
        eff_bw = ACCEL_CUTOFF_HZ if use_accel_lpf else fs / 2.0
        params['accel_oversample'] = max(1.0, fs / (2.0 * eff_bw))
        params['mag_oversample'] = max(1.0, fs / 80.0)
    else:
        params['accel_oversample'] = 1.0
        params['mag_oversample'] = 1.0

    throttle = dl.load_motor_data(dataset_dir, t_imu, accel=accel_vib_ref, accel_raw=accel,
                                  vib_scale=params.get('vib_scale', 2.0))

    if params.get('vib_relative', False):
        # excursion-above-baseline proxy: 0 in ordinary flight, so the VIBRATION branch adapts
        # instead of inflating R by a constant (see data_loader_USQUE.relative_proxy)
        throttle = dl.relative_proxy(t_imu, throttle, span=params.get('vib_rel_span', 3.0),
                                     floor_pct=params.get('vib_floor_pct', 20.0))

    # ---------------- initial attitude ----------------
    n_static = min(max(1, int(0.5 * fs)), len(accel_in))
    accel0 = np.mean(accel_in[:n_static], axis=0)
    mag0 = np.mean(mag_in[:n_static], axis=0)
    mag_ref_site = dl.load_mag_reference(dataset_dir)
    g0 = int(np.clip(np.searchsorted(t_gt, t_imu[0]), 0, len(quat_gt) - 1))

    mode = init_mode
    if mode == 'rtk_yaw' and (t_hdg is None or len(t_hdg) == 0):
        log(f"    [{name}] no RTK heading -> falling back to mag_triad")
        mode = 'mag_triad'

    sy = INIT_SIGMA_DEG['yaw'][mode]
    params['P0_att'] = np.radians([INIT_SIGMA_DEG['roll_pitch'], INIT_SIGMA_DEG['roll_pitch'], sy]) ** 2
    params['P0_bias'] = P0_BIAS_STATIC if np.any(b0_static != 0.0) else P0_BIAS_NO_STATIC

    if mode == 'mocap_yaw':
        q0 = tilt_quat_from_accel(accel0, float(q.to_euler(quat_gt[g0])[2]))
        # with the mag on (sensor-comparison runs), the magnetic reference is built from
        # this mocap-seeded attitude + mag0, as in 'rtk_yaw'
        f = usque.USQUE(params, initial_accel=accel0, initial_mag=mag0 if use_mag else None, init_q=q0)
    elif mode == 'mag_triad':
        q0 = usque.USQUE._triad_to_quat(accel0, mag0, mag_ref_site)
        f = usque.USQUE(params, initial_accel=accel0, initial_mag=mag0, init_q=q0)
    elif mode == 'rtk_yaw':
        q0 = usque.USQUE._triad_to_quat(accel0, mag0, mag_ref_site)
        h0 = int(np.clip(np.searchsorted(t_hdg, t_imu[0]), 0, len(t_hdg) - 1))
        yaw_seed = float(yaw_hdg[h0])
        if RTK_SEED_AVG_S > 0:
            # circular mean of the RTK heading over the leading still window (capped at
            # RTK_SEED_AVG_S): one 8 Hz sample carries ~0.2-0.4 deg of noise, and the seed error
            # is locked in, because the mag reference is rebuilt from this attitude
            still = (np.linalg.norm(gyro, axis=1) < 0.05) & (np.abs(np.linalg.norm(accel, axis=1) - 9.81) < 0.5)
            n_st = int(np.argmin(still)) if not still.all() else len(still)
            t_end = min(t_imu[max(n_st - 1, 0)], t_imu[0] + RTK_SEED_AVG_S)
            w = (t_hdg >= t_imu[0]) & (t_hdg <= t_end)
            if w.sum() >= 3:
                yaw_seed = float(np.arctan2(np.mean(np.sin(yaw_hdg[w])), np.mean(np.cos(yaw_hdg[w]))))
        q0 = set_quat_yaw(q0, yaw_seed)
        f = usque.USQUE(params, initial_accel=accel0, initial_mag=mag0, init_q=q0)
    else:
        raise ValueError(f"unknown init mode {init_mode!r}")

    if b0_static is not None:
        f.bias_nom = np.array(b0_static, dtype=float)

    e_init = np.degrees(q.to_euler(f.q_nom))
    e_gt0 = np.degrees(q.to_euler(quat_gt[g0]))
    init_err = (e_init - e_gt0 + 180.0) % 360.0 - 180.0
    log(f"    [{name}] init={mode}  q0 euler(deg)={np.round(e_init, 2)}  "
        f"err vs GT(t0)={np.round(init_err, 2)}  "
        f"model residual accel={f.init_residual_accel:.4f} mag={f.init_residual_mag:.3f}  "
        f"gyro bias0={np.round(np.degrees(b0_static), 3)} deg/s")

    # ---------------- filter loop ----------------
    M = len(t_imu)
    q_est = np.zeros((M, 4))
    bg_hist = np.zeros((M, 3))
    P_hist = np.zeros((M, f.P.shape[0], f.P.shape[0]))   # L=6 (no mag-bias state) or 7 (with it)
    nis_val, nis_adapt, nis_idx = [], [], []
    nis_acc, nis_mag = [], []          # per-block NIS (2 effective dof each)
    # Two UNIT vectors carry 2 degrees of freedom each (the radial component of the innovation
    # is ~0 by construction), so the chi-square dof of the NIS is 4 with mag / 2 accel-only,
    # not 6 / 3.
    dof_meas = 4 if use_mag else 2

    for i in range(M):
        dt = t_imu[i] - t_imu[i - 1] if i > 0 else (t_imu[1] - t_imu[0])
        f.predict(gyro_in[i], dt)
        if fusion_mask[i]:
            innov, S_base, S_adapt = f.update(accel_in[i], mag_in[i] if use_mag else None,
                                              throttle_pct=throttle[i])
            # ANIS is reported with the UN-inflated R; the inflated S is only for the gain.
            nis_val.append(float(innov @ np.linalg.solve(S_base, innov)))
            nis_adapt.append(float(innov @ np.linalg.solve(S_adapt, innov)))
            nis_acc.append(float(innov[0:3] @ np.linalg.solve(S_base[0:3, 0:3], innov[0:3])))
            if use_mag:
                nis_mag.append(float(innov[3:6] @ np.linalg.solve(S_base[3:6, 3:6], innov[3:6])))
            nis_idx.append(i)
        q_est[i] = f.q_nom
        bg_hist[i] = f.bias_nom
        P_hist[i] = f.P
    nis = np.array(nis_val)
    nis_adapt = np.array(nis_adapt)
    t_nis = t_imu[np.array(nis_idx, dtype=int)]

    # ---------------- evaluation ----------------
    e0 = max(t_gt[0], t_imu[0]) + EVAL_TRIM_START_S
    e1 = min(t_gt[-1], t_imu[-1]) - EVAL_TRIM_END_S
    mask = (t_gt >= e0) & (t_gt <= e1)
    sel = np.where(mask)[0]
    gt_hz = (len(t_gt) - 1) / (t_gt[-1] - t_gt[0])
    stride = max(1, int(round(gt_hz / EVAL_MAX_HZ)))
    sel = sel[::stride]
    t_cmp, quat_gt_c, pos_gt_c = t_gt[sel], quat_gt[sel], pos_gt[sel]
    idx = np.clip(np.searchsorted(t_imu, t_cmp), 0, M - 1)

    q_est_al, Pqq_all = q_est, P_hist
    if ALIGN_TO_GT:
        q_off = world_offset(t_imu, q_est, t_gt, quat_gt, t_gt[0] + 1.0, ANCHOR_WINDOW_S)
        q_est_al = np.array([q.multiply(q_off, qi) for qi in q_est])
        # error state is body-frame (right); a left world alignment does not rotate P
    q_est_c = q_est_al[idx]
    Pqq_c = P_hist[idx][:, 0:3, 0:3]

    # Indoor (mocap): the ~2.2-2.5 deg constant OFFSET between the mocap rigid-body frame
    # and the IMU/gravity frame (marker mounting) is a rig-calibration artifact, not filter
    # error -- see mocap_boresight()'s docstring. Fit it FIRST and score the boresight-
    # CORRECTED estimate, so the headline indoor numbers are not inflated by a known,
    # constant, non-filter offset. The raw (uncorrected) residual is still in 'boresight'.
    bs = None
    if prof['gt'] == 'mocap':
        try:
            bs = mt.mocap_boresight(q_est_c, quat_gt_c, omega=(gyro[idx] - bg_hist[idx]), label=name)
            # a fixed rig boresight (e.g. fitted once on the PX4 run) instead of this run's own fit
            bs['applied_rpy_deg'] = np.asarray(BORESIGHT_FIXED_RPY_DEG.get(name, bs['boresight_rpy_deg']), float)
            r_, p_, y_ = np.radians(bs['applied_rpy_deg'])
            xr, yr_, zr, wr = _Rot.from_euler('xyz', [r_, p_, y_]).as_quat()
            q_rhat = np.array([wr, xr, yr_, zr])
            # R_mocap = R_filter @ R_hat  =>  q_mocap = q_filter (x) q_rhat (body-side correction)
            q_est_c = np.array([q.multiply(qi, q_rhat) for qi in q_est_c])
        except Exception as e:   # noqa: BLE001
            log(f"    [{name}] mocap boresight skipped: {e}")

    qae = mt.qae_deg(q_est_c, quat_gt_c)
    axis_err = mt.axis_error_deg(q_est_c, quat_gt_c)
    nees, _, _ = mt.nees_series(q_est_c, quat_gt_c, Pqq_c)
    nees_ax = mt.nees_per_axis(q_est_c, quat_gt_c, Pqq_c)
    eul_est = np.degrees(np.array([q.to_euler(e) for e in q_est_c]))
    eul_gt = np.degrees(np.array([q.to_euler(g) for g in quat_gt_c]))

    yr = None
    if t_hdg is not None and len(t_hdg) > 0:
        hm = (t_hdg >= e0) & (t_hdg <= e1)
        yr = mt.yaw_vs_rtk(t_imu, q_est_al, P_hist, t_hdg[hm], yaw_hdg[hm])

    extras = {}
    if want_extras:
        # instrument resolution: gyro cross-check vs window length, and the accel-tilt
        # value vs still-detector thresholds (see metrics_usque docstrings)
        extras['gyro_multi'] = mt.gyro_consistency_multi(t_imu, q_est, gyro, bg_hist)
        extras['tilt_sens'] = mt.accel_tilt_sensitivity(ar, t_imu, q_est, gyro, accel, bg_hist,
                                                        P_hist, AR_EXCLUDE_INIT_S,
                                                        t_gt=t_gt, quat_gt=quat_gt)

    ar_d = None
    if want_ar:
        try:
            ar_d = ar.attitude_reference_report(
                t_imu, q_est, gyro, accel, bg_hist, t_gt=t_gt, quat_gt=quat_gt,
                P_hist=P_hist, ba_hist=None, grav_fire_times=[],
                exclude_init_s=AR_EXCLUDE_INIT_S, label=name)
        except Exception as e:   # noqa: BLE001
            log(f"    [{name}] attitude_reference skipped: {e}")

    if verbose:
        rm = lambda x: float(np.sqrt(np.nanmean(np.asarray(x) ** 2)))
        line = (f"    [{name}] qAE mean={np.mean(qae):.2f} deg | RMSE r/p/y = "
                f"{rm(axis_err[:, 0]):.2f}/{rm(axis_err[:, 1]):.2f}/{rm(axis_err[:, 2]):.2f} | "
                f"ANEES={np.mean(nees):.1f} | ANIS={np.mean(nis):.2f} (dof {dof_meas}) | "
                f"final bg={np.round(np.degrees(bg_hist[-1]), 3)} deg/s | "
                f"P repaired {f.pd_clip_count}/{f.pd_calls}")
        if getattr(f, 'use_mag_bias_state', False):
            line += f" | psi_mag={np.degrees(f.psi_mag_nom):+.2f} deg (sigma {np.degrees(np.sqrt(f.P[6, 6])):.2f})"
        if yr:
            line += f" | yaw vs RTK RMSE={yr['rmse']:.2f} deg"
        print(line)
        for ln in f.adaptation_summary().split("\n"):
            print("    " + ln)

    return dict(
        name=name, site=site, mode=mode, t_imu=t_imu, q_est=q_est, bg_hist=bg_hist, P_hist=P_hist,
        t_cmp=t_cmp, qae=qae, axis_err=axis_err, nees=nees, nees_ax=nees_ax, nis=nis, t_nis=t_nis,
        dof_meas=dof_meas, eul_est=eul_est, eul_gt=eul_gt, q_est_c=q_est_c, quat_gt_c=quat_gt_c,
        pos_gt_c=pos_gt_c, yaw_rtk=yr, boresight=bs, ar=ar_d, init_err=init_err,
        pd_repair=(f.pd_clip_count, f.pd_calls), gyro=gyro, accel=accel, idx=idx,
        Pqq_c=Pqq_c, nis_adapt=nis_adapt, nis_acc=np.array(nis_acc), nis_mag=np.array(nis_mag), adapt=f.adaptation_summary(), extras=extras,
        psi_mag_deg=float(np.degrees(f.psi_mag_nom)) if getattr(f, 'use_mag_bias_state', False) else None,
        mag_hi_center=mag_hi_center, mag_hi_radius=mag_hi_radius,
        sigma_yaw_ref_deg=(float(np.degrees(sig_hdg[0])) if sig_hdg is not None else None))


# ======================================================================
# Reporting / plots for a set of flights run in one init mode
# ======================================================================
def report_and_plot(results, mode_label, do_plots=True):
    if not results:
        return
    n = len(results)
    qae_runs = [r['qae'] for r in results]
    axis_runs = [r['axis_err'] for r in results]
    nees_runs = [r['nees'] for r in results]
    nis_runs = [r['nis'] for r in results]
    nees_ax_runs = [r['nees_ax'] for r in results]
    names = [r['name'] for r in results]
    ar_runs = [r['ar'] for r in results]
    dof_meas = results[0]['dof_meas']

    accel_ref = mt.build_accel_ref(ar_runs)

    # ---- GT-noise-aware NEES: P_ref from the reference's own error ----
    # roll/pitch: pooled Wahba-vs-accel-tilt residuals (outdoor GT); indoor mocap is
    # sub-degree so a nominal 0.5 deg is used. yaw: dual-RTK sigma (~1 deg) / 0.5 deg.
    # The strict default still-detector leaves 1-3 windows in total: too few to estimate the
    # Wahba GT's own error. That error is 7-9 deg, so the larger (~0.7 deg) accel-tilt floor of
    # the 'loose' preset is irrelevant for it -- pool ITS windows instead when they exist.
    loose = [dict(gt_d_roll_vs_accel=r['extras']['tilt_sens']['loose']['gt_d_roll_vs_accel'],
                  gt_d_pitch_vs_accel=r['extras']['tilt_sens']['loose']['gt_d_pitch_vs_accel'])
             for r in results if r['extras'].get('tilt_sens', {}).get('loose')]
    own = mt.pooled_gt_own_error(loose) if loose else None
    if own is None:
        own = mt.pooled_gt_own_error(ar_runs)
    indoor = dl.SITE_PROFILE[FLIGHT_SITE]['gt'] == 'mocap'
    sy = [r['sigma_yaw_ref_deg'] for r in results if r['sigma_yaw_ref_deg'] is not None]
    sig_yaw = float(np.mean(sy)) if sy else 0.5
    if indoor:
        sigma_ref = (0.5, 0.5, 0.5)
    elif own is not None:
        sigma_ref = (own[0], own[1], sig_yaw)
    else:
        sigma_ref = None
    nees_ref = None
    if sigma_ref is not None:
        ref_runs = [mt.nees_series_ref(r['q_est_c'], r['quat_gt_c'], r['Pqq_c'], sigma_ref) for r in results]
        Pref = np.radians(np.asarray(sigma_ref)) ** 2
        ax_runs = []
        for r in results:
            th = np.array([mt._error_and_cov(r['q_est_c'][i], r['quat_gt_c'][i], r['Pqq_c'][i])[0]
                           for i in range(0, len(r['q_est_c']), 5)])
            Pd = np.array([np.diag(r['Pqq_c'][i]) for i in range(0, len(r['q_est_c']), 5)])
            ax_runs.append(np.mean(th ** 2 / (Pd + Pref), axis=0))
        nees_ref = dict(mean=float(np.mean([np.mean(x) for x in ref_runs])),
                        sigma_ref_deg=sigma_ref, axis=np.mean(ax_runs, axis=0))
    mt.print_evaluation_report(qae_runs, axis_runs, nees_runs, nis_runs, n,
                               dof_att=3, dof_meas=dof_meas, accel_ref=accel_ref,
                               nees_axis_runs=nees_ax_runs, label=mode_label, nees_ref=nees_ref)

    # ---- measurement-only additions: per-flight P90 / CDF(5 deg), GT-quality screen, per-block ANIS ----
    def _gt_quality(r):
        """('good'|'bad'|'thin'|'none', roll_own, pitch_own, n_windows): the Wahba GT's own roll/pitch
        error vs the accel-tilt at still windows (loose preset when it has >= 2 windows, else default)."""
        if dl.SITE_PROFILE[FLIGHT_SITE]['gt'] == 'mocap':
            return 'mocap', float('nan'), float('nan'), 0
        ls_ = (r['extras'].get('tilt_sens') or {}).get('loose') or {}
        dr, dp = list(ls_.get('gt_d_roll_vs_accel', [])), list(ls_.get('gt_d_pitch_vs_accel', []))
        if len(dr) < 2:
            a_ = r['ar'] or {}
            dr, dp = list(a_.get('gt_d_roll_vs_accel', [])), list(a_.get('gt_d_pitch_vs_accel', []))
        if not dr:
            return 'none', float('nan'), float('nan'), 0
        gr, gp = float(np.sqrt(np.mean(np.square(dr)))), float(np.sqrt(np.mean(np.square(dp))))
        if max(gr, gp) >= GT_GOOD_MAX_DEG:
            return 'bad', gr, gp, len(dr)
        return ('good' if len(dr) >= 2 else 'thin'), gr, gp, len(dr)

    print(f"\n--- PER-FLIGHT CDF(5 deg) / P90 / GT QUALITY [{mode_label}] ---")
    print(f"{'flight':<16} {'GT':>5} {'own r/p':>10} {'n':>2} | {'CDF5 r/p/y':>17} | {'P90 r/p/y (deg)':>20} | "
          f"{'yawRTK':>6} | {'ANIS acc/mag (ideal 2)':>22}")
    quality = []
    for r, ax_e in zip(results, axis_runs):
        qk, gr, gp, gn = _gt_quality(r)
        quality.append(qk)
        ae = np.abs(ax_e)
        c5 = [float(np.nanmean(ae[:, k] < 5.0)) for k in range(3)]
        p90 = [float(np.nanpercentile(ae[:, k], 90)) for k in range(3)]
        yr_ = f"{r['yaw_rtk']['rmse']:6.2f}" if r['yaw_rtk'] else "   n/a"
        na = f"{np.mean(r['nis_acc']):.2f}" if len(r['nis_acc']) else "n/a"
        nm = f"{np.mean(r['nis_mag']):.2f}" if len(r['nis_mag']) else "n/a"
        print(f"{r['name']:<16} {qk:>5} {gr:5.1f}/{gp:<4.1f} {gn:>2} | {c5[0]:5.2f} {c5[1]:5.2f} {c5[2]:5.2f}  | "
              f"{p90[0]:6.1f} {p90[1]:6.1f} {p90[2]:6.1f}  | {yr_} | {na:>10} / {nm:<8}")
    good = [r for r, qk in zip(results, quality) if qk in ('good', 'mocap')]
    if good and any(qk == 'good' for qk in quality):
        ge = [np.abs(r['axis_err']) for r in good]
        pool = [np.concatenate([e[:, k] for e in ge]) for k in range(3)]
        pool = [p[np.isfinite(p)] for p in pool]
        print(f"GT-validated flights ({', '.join(r['name'].split('_sensors')[0] for r in good)}; own error < "
              f"{GT_GOOD_MAX_DEG:.0f} deg): RMSE roll/pitch/yaw "
              f"{np.sqrt(np.mean(pool[0] ** 2)):.2f}/{np.sqrt(np.mean(pool[1] ** 2)):.2f}/{np.sqrt(np.mean(pool[2] ** 2)):.2f}  "
              f"CDF(5) {np.mean(pool[0] < 5):.2f}/{np.mean(pool[1] < 5):.2f}/{np.mean(pool[2] < 5):.2f}  "
              f"P90 {np.percentile(pool[0], 90):.1f}/{np.percentile(pool[1], 90):.1f}/{np.percentile(pool[2], 90):.1f}")
    nb = [r for r, qk in zip(results, quality) if qk == 'bad']
    if nb:
        print(f"Flights whose Wahba roll/pitch GT is itself off by >= {GT_GOOD_MAX_DEG:.0f} deg "
              f"({', '.join(r['name'].split('_sensors')[0] for r in nb)}): their roll/pitch numbers are "
              f"reference-limited and are NOT filter error.")
    all_na = np.concatenate([r['nis_acc'] for r in results])
    all_nm = np.concatenate([r['nis_mag'] for r in results if len(r['nis_mag'])]) if any(len(r['nis_mag']) for r in results) else np.array([])
    print(f"Per-block ANIS (base R, ideal 2 each): accel {all_na.mean():.2f}" + (f" / mag {all_nm.mean():.2f}" if all_nm.size else ""))

    # ---- roll/pitch error bounds from references of different kinds ----
    bounds = mt.error_bounds(axis_runs, ar_runs, own=own) if not indoor else None
    if bounds:
        gr, gp, gn = bounds['gt_own']
        print(f"\n--- ROLL/PITCH ERROR BOUNDS [{mode_label}] ---")
        print(f"raw vs Wahba GT (upper)      : roll {bounds['roll'][1]:.2f} / pitch {bounds['pitch'][1]:.2f} deg")
        print(f"Wahba-GT own error vs tilt   : roll {gr:.2f} / pitch {gp:.2f} deg   ({gn} still windows pooled)")
        print(f"quadrature-subtracted (LOWER): roll >= {bounds['roll'][0]:.2f} / pitch >= {bounds['pitch'][0]:.2f} deg  "
              f"(filter/GT errors correlated via the magnetometer -> true error is at least this)")
        gy = [a for a in (r['ar'] for r in results) if a and 'filter_roll_gyro_consistency_rmse' in a]
        if gy:
            print(f"gyro cross-check (independent): roll {np.mean([a['filter_roll_gyro_consistency_rmse'] for a in gy]):.2f} / "
                  f"pitch {np.mean([a['filter_pitch_gyro_consistency_rmse'] for a in gy]):.2f} deg (15 s windows; drift only)")

    # ---- instrument resolution ----
    gm = [r['extras'].get('gyro_multi') for r in results if r['extras'].get('gyro_multi')]
    if gm:
        print(f"\n--- GYRO CROSS-CHECK vs WINDOW LENGTH [{mode_label}] (grows = real drift; flat = check's own floor) ---")
        print(f"{'window':>8} {'roll':>7} {'pitch':>7} {'n':>6}")
        for w in sorted(gm[0]):
            rr = [g[w][0] for g in gm if w in g]
            pp = [g[w][1] for g in gm if w in g]
            nn = sum(g[w][2] for g in gm if w in g)
            print(f"{w:7.0f}s {np.mean(rr):7.2f} {np.mean(pp):7.2f} {nn:6d}")
    ts = [r['extras'].get('tilt_sens') for r in results if r['extras'].get('tilt_sens')]
    if ts:
        print(f"\n--- ACCEL-TILT vs STILL-DETECTOR THRESHOLDS [{mode_label}] (stable value + many windows, or sanity check only) ---")
        print(f"{'preset':>8} {'windows':>8} {'roll':>7} {'pitch':>7} {'floor':>7}")
        for name in mt.STILL_PRESETS:
            rows_ = [t[name] for t in ts if name in t and np.isfinite(t[name]['roll'])]
            nw = sum(t[name]['n'] for t in ts if name in t)
            if rows_:
                print(f"{name:>8} {nw:8d} {np.mean([x['roll'] for x in rows_]):7.2f} "
                      f"{np.mean([x['pitch'] for x in rows_]):7.2f} {np.nanmean([x['floor'] for x in rows_]):7.2f}")
            else:
                print(f"{name:>8} {nw:8d}     n/a     n/a     n/a")
    print(f"\n--- ADAPTIVE-R FIRING [{mode_label}] ---")
    for r in results:
        print(f"  {r['name']}: " + r['adapt'].split("\n")[0])
    print(results[-1]['adapt'].split("\n", 1)[1] if "\n" in results[-1]['adapt'] else "", end="")
    print("  (per-branch table above is for the last flight)")
    yrs = [r['yaw_rtk'] for r in results if r['yaw_rtk']]
    if yrs:
        print(f"Yaw vs dual-RTK heading [{mode_label}]: RMSE {np.mean([y['rmse'] for y in yrs]):.2f} deg  "
              f"MAE {np.mean([y['mae'] for y in yrs]):.2f}  max {np.max([y['maxabs'] for y in yrs]):.2f}  "
              f"NEES mean {np.mean([y['nees_mean'] for y in yrs]):.1f} (dof=1, ideal 1)")
    bs_ok = [r['boresight'] for r in results if r['boresight']]
    if bs_ok:
        res = np.array([b['resid_rms_deg'] for b in bs_ok])
        rx = np.array([b['resid_rms_rpy_deg'] for b in bs_ok])
        print(f"Mocap boresight (accel-independent) [{mode_label}]: residual {res.mean():.3f} deg RMS "
              f"(roll {rx[:, 0].mean():.3f}, pitch {rx[:, 1].mean():.3f}, yaw {rx[:, 2].mean():.3f})")
    rep = sum(r['pd_repair'][0] for r in results)
    tot = sum(r['pd_repair'][1] for r in results)
    print(f"P positive-definite repairs [{mode_label}]: {rep}/{tot} calls "
          f"({100.0 * rep / max(tot, 1):.3f}%)")

    if not do_plots:
        return

    tag = f"{FLIGHT_SITE} | init={mode_label}"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- bias convergence + covariance diagnostics ----
    hist_runs = [(r['t_imu'] - r['t_imu'][0], r['bg_hist'],
                  np.diagonal(r['P_hist'], axis1=1, axis2=2)) for r in results]
    if n > 1:
        t_g, bg_m, Pd_m = mt.mean_histories_6(hist_runs)
        L_ = Pd_m.shape[1]
        P_m = np.zeros((len(t_g), L_, L_))
        for k in range(L_):
            P_m[:, k, k] = Pd_m[:, k]
        sfx = f"{tag} | MEAN of {n} flights (common {t_g[-1]:.0f}s)"
        mt.plot_bias_convergence(t_g, bg_m, title_suffix=sfx)
        mt.plot_covariance_diagonals(t_g, P_m, title_suffix=sfx)
    else:
        r = results[-1]
        mt.plot_bias_convergence(r['t_imu'], r['bg_hist'], title_suffix=tag)
        mt.plot_covariance_diagonals(r['t_imu'], r['P_hist'], title_suffix=tag)

    # ---- trajectory: GT path + GT vs estimated attitude (last flight) ----
    r = results[-1]
    mt.plot_attitude_trajectory(r['pos_gt_c'], r['quat_gt_c'], r['q_est_c'],
                                title_suffix=f"{r['name']} | {tag}")

    if RUN_MODE == 0:
        _ad = ar_runs[-1]
        apts = None
        if _ad and _ad.get("accel_tilt_t"):
            apts = (np.array(_ad["accel_tilt_t"]), np.array(_ad["accel_tilt_roll"]),
                    np.array(_ad["accel_tilt_pitch"]))
        apw = None
        if _ad and _ad.get("accel_tilt_t") and "roll_nees_pw" in _ad:
            apw = dict(t=_ad["accel_tilt_t"], d_roll=_ad["d_roll_vs_accel"],
                       d_pitch=_ad["d_pitch_vs_accel"], qae=_ad.get("qae_accelref_pw", []),
                       roll_nees=_ad["roll_nees_pw"], pitch_nees=_ad["pitch_nees_pw"])
        t_cmp = results[-1]['t_cmp']
        mt.plot_euler_comparison(t_cmp, results[0]['eul_est'], results[0]['eul_gt'], accel_pts=apts)
        mt.plot_monte_carlo_qae(t_cmp, qae_runs, accel_pts=apw)
        mt.plot_monte_carlo_axis_errors(t_cmp, axis_runs, accel_pts=apw)
        mt.plot_monte_carlo_consistency(t_cmp, nees_runs, None, nis_runs, n,
                                        dof_att=3, dof_meas=dof_meas, accel_pts=apw)
    else:
        # batch: RMSE bars + CDF vs the Wahba/mocap GT ...
        mt.plot_batch_rmse_bars(axis_runs, dataset_names=names)
        mt.plot_batch_cdf(axis_runs)
        # ... the roll/pitch error BOUNDS (replaces the misleading "0.09 deg accel-tilt bar
        # next to a 9 deg GT bar" chart): raw vs GT, quadrature-subtracted lower bound,
        # independent gyro cross-check, and the accel-tilt reference floor as a band.
        if bounds:
            rows_ = []
            for r, ax_e in zip(results, axis_runs):
                a_ = r['ar'] or {}
                gr_, gp_ = (a_.get('gt_roll_rmse_vs_accel'), a_.get('gt_pitch_rmse_vs_accel'))
                raw_r = float(np.sqrt(np.nanmean(ax_e[:, 0] ** 2)))
                raw_p = float(np.sqrt(np.nanmean(ax_e[:, 1] ** 2)))
                # per-flight GT-own error is only 1-2 windows; use the pooled value
                lb = lambda raw, g: float(np.sqrt(max(raw ** 2 - g ** 2, 0.0)))
                rows_.append(dict(raw_r=raw_r, raw_p=raw_p,
                                  lb_r=lb(raw_r, bounds['gt_own'][0]), lb_p=lb(raw_p, bounds['gt_own'][1]),
                                  gyro_r=a_.get('filter_roll_gyro_consistency_rmse', np.nan),
                                  gyro_p=a_.get('filter_pitch_gyro_consistency_rmse', np.nan)))
            floor = (accel_ref or {}).get('floor')
            mt.plot_batch_error_bounds(names, rows_, floor_deg=floor, title_suffix=tag)
        # ---- approximate TRUE error per flight (reference noise removed) ----
        # roll/pitch: raw RMSE vs Wahba GT with the GT's own error subtracted (per-flight value
        # when that flight has >= 3 'loose' still windows, else the pooled one);
        # yaw: RMSE vs the dual-RTK heading with the ~1 deg RTK noise subtracted.
        if bounds:
            te_rows = []
            print(f"\n--- APPROXIMATE TRUE ERROR PER FLIGHT [{mode_label}] (deg; est = independent errors, "
                  f"hi = correlated rho=0.5) ---")
            print(f"{'flight':<16} {'axis':<6} {'raw':>7} {'ref err':>8} {'est':>7} {'hi':>7}")
            for r, ax_e in zip(results, axis_runs):
                ls_ = (r['extras'].get('tilt_sens') or {}).get('loose') or {}
                dr, dp = ls_.get('gt_d_roll_vs_accel', []), ls_.get('gt_d_pitch_vs_accel', [])
                if len(dr) >= 2:
                    g_r, g_p, src = (float(np.sqrt(np.mean(np.square(dr)))),
                                     float(np.sqrt(np.mean(np.square(dp)))), 'own')
                else:
                    g_r, g_p, src = bounds['gt_own'][0], bounds['gt_own'][1], 'pooled'
                raw_r = float(np.sqrt(np.nanmean(ax_e[:, 0] ** 2)))
                raw_p = float(np.sqrt(np.nanmean(ax_e[:, 1] ** 2)))
                er = mt.approx_true_error(raw_r, g_r)
                ep = mt.approx_true_error(raw_p, g_p)
                if r['yaw_rtk']:
                    raw_y = r['yaw_rtk']['rmse']
                    est_y = float(np.sqrt(max(raw_y ** 2 - sig_yaw ** 2, 0.0)))
                else:
                    raw_y = float(np.sqrt(np.nanmean(ax_e[:, 2] ** 2))); est_y = raw_y
                a_ = r['ar'] or {}
                te_rows.append(dict(raw_r=raw_r, raw_p=raw_p, raw_y=raw_y,
                                    est_r=er[0], est_p=ep[0], est_y=est_y,
                                    hi_r=er[1], hi_p=ep[1], hi_y=est_y,
                                    unres_r=er[2], unres_p=ep[2],
                                    gyro_r=a_.get('filter_roll_gyro_consistency_rmse', np.nan),
                                    gyro_p=a_.get('filter_pitch_gyro_consistency_rmse', np.nan)))
                for nm, raw_, g_, e_ in (('roll', raw_r, g_r, er), ('pitch', raw_p, g_p, ep)):
                    print(f"{r['name']:<16} {nm:<6} {raw_:7.2f} {g_:8.2f} "
                          f"{e_[0]:7.2f} {e_[1]:7.2f}" + ("  (ref noise >= raw: unresolved)" if e_[2] else "")
                          + f"  [ref err: {src}]")
                print(f"{r['name']:<16} {'yaw':<6} {raw_y:7.2f} {sig_yaw:8.2f} {est_y:7.2f} {est_y:7.2f}  [ref: dual-RTK]")
            mt.plot_batch_true_error(names, te_rows, title_suffix=tag)
        # accel-tilt CDF only when there are enough windows for a CDF to mean anything
        if accel_ref and accel_ref.get('n_windows', 0) >= 8:
            mt.plot_batch_cdf_accel(ar_runs, axis_err_runs=axis_runs)
        else:
            print("[accel-tilt CDF] skipped: fewer than 8 still windows in total")
        mt.export_batch_results_to_csv(
            qae_runs, axis_runs, nees_runs, nis_runs, filter_name=f"USQUE_{FLIGHT_SITE}_{mode_label}",
            filename=os.path.join(RESULTS_DIR, f"usque_{FLIGHT_SITE}_{mode_label}.csv"))
        mt.export_accel_ref_csv(
            ar_runs, dataset_names=names, filter_name=f"USQUE_{FLIGHT_SITE}_{mode_label}",
            filename=os.path.join(RESULTS_DIR, f"usque_{FLIGHT_SITE}_{mode_label}_acceltilt.csv"))


def compare_modes(all_results):
    """Side-by-side summary of the init strategies over the same flights."""
    modes = list(all_results)
    if len(modes) < 2:
        return
    print("\n================ INIT-MODE COMPARISON (same flights, same params) ================")
    hdr = f"{'mode':<10} {'qAE':>7} {'roll':>7} {'pitch':>7} {'yaw':>7} {'yawRTK':>8} {'ANEES':>8} {'init yaw err':>13}"
    print(hdr)
    rows = {}
    for m in modes:
        R = all_results[m]
        rm = lambda e, k: float(np.mean([np.sqrt(np.nanmean(a[:, k] ** 2)) for a in e]))
        yrs = [r['yaw_rtk']['rmse'] for r in R if r['yaw_rtk']]
        rows[m] = dict(
            qae=float(np.mean([np.mean(r['qae']) for r in R])),
            roll=rm([r['axis_err'] for r in R], 0), pitch=rm([r['axis_err'] for r in R], 1),
            yaw=rm([r['axis_err'] for r in R], 2),
            yawrtk=float(np.mean(yrs)) if yrs else float('nan'),
            anees=float(np.mean([np.mean(r['nees']) for r in R])),
            iyaw=float(np.mean([abs(r['init_err'][2]) for r in R])))
        v = rows[m]
        print(f"{m:<10} {v['qae']:7.2f} {v['roll']:7.2f} {v['pitch']:7.2f} {v['yaw']:7.2f} "
              f"{v['yawrtk']:8.2f} {v['anees']:8.1f} {v['iyaw']:13.2f}")
    print("  (deg; qAE/roll/pitch/yaw vs the site GT, yawRTK vs dual-RTK heading, init yaw err vs GT at t0)")

    names = [r['name'] for r in all_results[modes[0]]]
    x = np.arange(len(names))
    w = 0.8 / len(modes)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for j, m in enumerate(modes):
        R = all_results[m]
        ax[0].bar(x + (j - (len(modes) - 1) / 2) * w, [np.mean(r['qae']) for r in R], w, label=m)
        ax[1].bar(x + (j - (len(modes) - 1) / 2) * w,
                  [np.sqrt(np.nanmean(r['axis_err'][:, 2] ** 2)) for r in R], w, label=m)
    for a_, ttl in zip(ax, ("Mean qAE per flight (deg)", "Yaw RMSE vs GT per flight (deg)")):
        a_.set_xticks(x)
        a_.set_xticklabels(names, rotation=45, ha='right')
        a_.set_title(ttl)
        a_.grid(axis='y', ls='--', alpha=0.6)
        a_.legend()
    fig.suptitle(f"Init-mode comparison | {FLIGHT_SITE}")
    fig.tight_layout()


# ======================================================================
# main
# ======================================================================
def main():
    total_start = time.perf_counter()

    # Permanent sanity check of the measurement model / TRIAD at non-trivial
    # attitudes (the constructor also runs it and additionally checks the model
    # against the real data at t=0).
    worst = usque.USQUE.self_check()
    print(f"USQUE.self_check OK (worst residual {worst:.1e})")

    if RUN_MODE == 1:
        datasets = dl.flight_dirs(FLIGHT_SITE, BASE_DIR, n=N, skip=SKIP_FLIGHTS, include=INCLUDE_FLIGHTS)
    else:
        datasets = dl.flight_dirs(FLIGHT_SITE, BASE_DIR, include=[TARGET_FLIGHT]) * N
    modes = site_modes(FLIGHT_SITE)
    print(f"*** USQUE | site={FLIGHT_SITE} | {'BATCH' if RUN_MODE == 1 else 'SINGLE'} | "
          f"{len(datasets)} flight(s) | init modes={modes} | align_to_gt={ALIGN_TO_GT} ***")
    if FLIGHT_SITE == 'mars' and RUN_MODE == 1 and SKIP_FLIGHTS and not INCLUDE_FLIGHTS:
        print(f"    EXCLUDED flights {SKIP_FLIGHTS}  <-- report this explicitly")

    all_results = {}
    for mode in modes:
        res = []
        print(f"\n########## init mode: {mode} ##########")
        for k, d in enumerate(datasets):
            print(f"\n--- Run {k + 1}/{len(datasets)} : {os.path.basename(d)} ---")
            try:
                res.append(run_flight(d, FLIGHT_SITE, mode, want_extras=True))
            except FileNotFoundError as e:
                print(f"Error loading {d}: {e}")
        all_results[mode] = res

    print(f"\n>>> Evaluation completed in {time.perf_counter() - total_start:.1f} s")
    for mode, res in all_results.items():
        report_and_plot(res, mode, do_plots=(mode == PLOT_INIT_MODE or len(modes) == 1))
    compare_modes(all_results)
    plt.show()


if __name__ == "__main__":
    main()
