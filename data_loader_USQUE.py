"""
data_loader_USQUE.py
--------------------
Data layer for main_usque.py (pure AHRS: gyro + accel + mag, no GPS/INS).

This used to be a stand-alone, mars-only loader with its own ENU->FRD
conversion (a left-multiplication by a fixed quaternion). It is now a thin
layer on top of data_loader.py -- the loader main.py already validates --
so USQUE sees exactly the same data as the hybrid filter:

  * all four INSANE sites: mars, transition, outdoor, indoor
  * ground truth via the rigid similarity R_ned_frd = M R M
    (RTK/Wahba GT for mars/transition/outdoor, Optitrack mocap GT indoor)
  * per-site magnetometer calibration + world-frame magnetic reference
    (inclination AND declination) from the site YAML
  * dual-RTK heading (used only to SEED the initial yaw and as a diagnostic)

Everything is NED world / FRD body, quaternions Hamilton wxyz body->nav.
(On Windows data_loader_USQUE.py and data_loader_usque.py are the same file.)
"""

import os
import numpy as np

import data_loader as _dl

# ----------------------------------------------------------------------
# Sites
# ----------------------------------------------------------------------
# folder pattern + available flight numbers per site
SITE_FLIGHTS = {
    'mars':       ("mars_{}_sensors",       list(range(1, 20))),
    'transition': ("transition_{}_sensors", [1, 2, 3]),
    'outdoor':    ("outdoor_{}_sensors",    [1]),
    'indoor':     ("indoor_{}_sensors",     [1, 2, 3]),
}

# gt   : 'enu'   -> RTK + Wahba ground truth (only yaw is trustworthy)
#        'mocap' -> Optitrack pose, sub-degree on all three axes
# heading: 'rtk' -> dual-RTK baseline heading available; None -> none
# mag  : magnetometer usable (False indoors: building distortion)
SITE_PROFILE = {
    'mars':       dict(gt='enu',   heading='rtk', mag=True),
    'transition': dict(gt='enu',   heading='rtk', mag=True),
    'outdoor':    dict(gt='enu',   heading='rtk', mag=True),
    'indoor':     dict(gt='mocap', heading=None,  mag=False),
}

# Fallback NED magnetic reference (Ramon Crater / Negev, WMM 2019-2024) used
# only if the site YAML carries no mag_var.
MAG_REF_NED_WMM = np.array([0.68872216, 0.05750074, 0.72274162])


def flight_dirs(site, base_dir, n=None, skip=None, include=None):
    """Dataset folders for a site. `skip` only applies to mars (flights 4 and
    14 have disturbed mag / missing data); `include` overrides the range."""
    if site not in SITE_FLIGHTS:
        raise ValueError(f"site must be one of {list(SITE_FLIGHTS)}, got {site!r}")
    pattern, ids = SITE_FLIGHTS[site]
    ids = list(include) if include else list(ids)
    if skip and site == "mars" and not include:
        ids = [i for i in ids if i not in set(skip)]
    if n is not None:
        ids = ids[:n]
    return [os.path.join(base_dir, pattern.format(i)) for i in ids]


def to_seconds(t):
    """Best-effort timestamp normalisation to seconds (see main.py)."""
    t = np.asarray(t, dtype=float)
    dt_med = np.median(np.diff(t))
    if dt_med > 1e6:
        return t * 1e-9
    if dt_med > 1e2:
        return t * 1e-6
    return t


# ----------------------------------------------------------------------
# Sensors / ground truth
# ----------------------------------------------------------------------
def load_sensor_data(dataset_dir):
    """(t_imu, gyro_frd, accel_frd, mag_frd_on_imu_grid).

    accel is SPECIFIC FORCE (reads (0,0,-g) at rest, level). The
    magnetometer is calibrated with the site YAML and linearly interpolated
    onto the IMU time grid (native rate ~80 Hz)."""
    t_imu, gyro, accel, t_mag, mag_calib = _dl.load_sensor_data(dataset_dir)
    t_imu = to_seconds(t_imu)
    t_mag = to_seconds(t_mag)
    mag = np.column_stack([np.interp(t_imu, t_mag, mag_calib[:, k]) for k in range(3)])
    return t_imu, gyro, accel, mag


def load_groundtruth(dataset_dir, frame='enu'):
    """(t_gt, pos_ned, quat_wxyz_body_to_nav) -- frame in {'enu','mocap'}."""
    t_gt, pos, quat = _dl.load_groundtruth(dataset_dir, frame=frame)
    return to_seconds(t_gt), pos, quat


def load_rtk_heading(dataset_dir):
    """(t, yaw_rad, sigma) from the dual-RTK baseline, or (None, None, None)."""
    t, yaw, sig = _dl.load_rtk_heading(dataset_dir)
    if t is None:
        return None, None, None
    return to_seconds(t), yaw, sig


def load_mag_reference(dataset_dir):
    """Unit NED magnetic reference for the site (inclination + declination)
    from the site YAML; falls back to the Negev WMM vector."""
    ref = _dl.load_mag_calibration_yaml(dataset_dir).get('mag_ref_ned')
    return np.asarray(MAG_REF_NED_WMM if ref is None else ref, float)


def load_motor_data(dataset_dir, t_imu, accel=None, accel_raw=None, vib_scale=2.0):
    """
    Motor RPM (px4_rpm.csv, normalised 0..1) when the file exists, else a
    vibration proxy built from the HIGH-FREQUENCY accelerometer residual
    (accel_raw - accel_lowpassed), RMS-averaged over ~0.5 s and normalised by
    `vib_scale`. There is deliberately no constant floor: a calm hover gives a
    proxy near 0, so the VIBRATION adaptive-R stays quiet until real vibration
    arrives. `vib_scale` needs tuning against the dataset's vibration levels.
    """
    motor_path = os.path.join(dataset_dir, "px4_rpm.csv")

    if os.path.exists(motor_path):
        try:
            try:
                motor_data = np.loadtxt(motor_path, delimiter=',', comments='#')
            except ValueError:
                motor_data = np.loadtxt(motor_path, delimiter=',', comments='#', skiprows=1)

            t_motor = motor_data[:, 0]
            motors_mean = np.mean(motor_data[:, 1:5], axis=1)
            if np.max(motors_mean) > 2.0:
                motors_mean = motors_mean / 15500.0
            motors_mean = np.clip(motors_mean, 0.0, 1.0)
            return np.interp(t_imu, t_motor, motors_mean)
        except Exception:
            pass

    if accel_raw is not None and accel is not None:
        vib_energy = np.linalg.norm(accel_raw - accel, axis=1)
        if len(t_imu) > 1:
            dt_mean = np.mean(np.diff(t_imu))
            fs = 1.0 / dt_mean if dt_mean > 0 else 1.0
            window = max(1, int(round(0.5 * fs)))
        else:
            window = 1
        if window > 1:
            kernel = np.ones(window) / window
            vib_rms = np.sqrt(np.convolve(vib_energy ** 2, kernel, mode='same'))
        else:
            vib_rms = vib_energy
        return np.clip(vib_rms / vib_scale, 0.0, 1.0)

    if accel is not None:
        norm_a = np.linalg.norm(accel, axis=1)
        return np.clip(np.abs(norm_a - 9.81) / 10.0, 0.0, 1.0)

    return np.zeros(len(t_imu))


def make_fusion_mask(t_imu, fusion_rate_hz):
    """Boolean mask selecting IMU steps at which a measurement update runs, so
    updates happen at ~fusion_rate_hz instead of the full IMU rate."""
    mask = np.zeros(len(t_imu), dtype=bool)
    if fusion_rate_hz is None or fusion_rate_hz <= 0:
        mask[:] = True
        return mask
    period = 1.0 / fusion_rate_hz
    next_t = t_imu[0]
    for i, t in enumerate(t_imu):
        if t >= next_t:
            mask[i] = True
            next_t = t + period
    return mask


def load_static_gyro_bias(dataset_dir, t_imu=None, gyro=None, accel=None,
                          max_window_s=4.0, gyro_still_thresh=0.05,
                          accel_still_thresh=0.5):
    """Static gyro bias = mean gyro over the leading still window (gyro norm and
    | |accel| - g | both below threshold, capped at max_window_s). Zeros if no
    still window is found or only dataset_dir is given."""
    if t_imu is None or gyro is None:
        return np.array([0.0, 0.0, 0.0])

    gyro = np.asarray(gyro)
    n = len(t_imu)
    still = np.linalg.norm(gyro, axis=1) < gyro_still_thresh
    if accel is not None:
        accel = np.asarray(accel)
        still &= np.abs(np.linalg.norm(accel, axis=1) - 9.81) < accel_still_thresh

    end = 0
    while end < n and still[end]:
        end += 1
    if end < 10:
        return np.array([0.0, 0.0, 0.0])

    end = min(end, int(np.searchsorted(t_imu, t_imu[0] + max_window_s)))
    if end < 10:
        return np.array([0.0, 0.0, 0.0])
    return np.mean(gyro[:end], axis=0)


def relative_proxy(t_imu, x, span=3.0, floor_pct=20.0, window_s=60.0, decim_hz=10.0):
    """Turn an absolute vibration/RPM proxy into an EXCURSION-above-baseline proxy in [0, 1].

    The absolute proxies (RMS of the high-frequency accel residual, or mean motor RPM) sit at
    a large, roughly constant level for the whole flight of a running quad, so
    R * (1 + k * proxy) is a constant inflation, not adaptation. Here the baseline is the
    `floor_pct` percentile of the proxy over the PAST `window_s` seconds (causal: implementable
    on the ESP32), and the output is  clip((x / baseline - 1) / span, 0, 1):
    0 in ordinary flight (no inflation), 1 when the proxy is (1 + span) x its recent floor.
    """
    import pandas as pd
    t = np.asarray(t_imu, float)
    x = np.asarray(x, float)
    step = max(1, int(round((len(t) - 1) / (t[-1] - t[0]) / decim_hz)))
    idx = np.arange(0, len(t), step)
    xs = pd.Series(x[idx])
    win = max(10, int(window_s * decim_hz))
    floor = xs.rolling(win, min_periods=10).quantile(floor_pct / 100.0)
    floor = floor.fillna(xs.expanding(min_periods=1).quantile(floor_pct / 100.0)).to_numpy()
    floor_full = np.interp(t, t[idx], floor)
    floor_full = np.maximum(floor_full, 1e-3)
    return np.clip((x / floor_full - 1.0) / span, 0.0, 1.0)


def fit_hard_iron_sphere(mag):
    """Least-squares sphere fit: returns (center, radius) minimising
    sum((|m_i - center| - radius)^2) via the standard linearised form
    2*c.x + (r^2-|c|^2) = |x|^2. Used for PER-FLIGHT hard-iron correction on top
    of the global (site) calibration -- absorbs a per-flight residual offset
    (different payload/battery/motor current) that the fixed YAML calibration
    cannot see. Needs enough heading/attitude spread to be well-conditioned;
    with too little (e.g. a near-straight-line flight) the fit is unreliable --
    callers should sanity-check the returned radius against the expected field
    magnitude (data-dependent, so no fixed threshold is imposed here)."""
    x, y, z = mag[:, 0], mag[:, 1], mag[:, 2]
    A = np.column_stack([2 * x, 2 * y, 2 * z, np.ones_like(x)])
    b = x ** 2 + y ** 2 + z ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r2 = sol[3] + float(c @ c)
    return c, float(np.sqrt(max(r2, 0.0)))


def per_flight_hard_iron_correct(mag):
    """(mag_corrected, center, radius) -- subtracts the per-flight fitted sphere
    center from every sample. See fit_hard_iron_sphere()."""
    c, r = fit_hard_iron_sphere(mag)
    return mag - c, c, r
