"""
data_loader.py
--------------
Loads IMU, Magnetometer, Ground-Truth, Calibration, and RTK GPS data.
Includes dual-quaternion mapping to perfectly align BOTH World (ENU->NED) and Body frames!
"""

import os
import pandas as pd
import numpy as np
import yaml
import project_utils as q
from scipy.spatial.transform import Rotation as _Rot

IMU_T_COL   = 0
IMU_ACC     = [1, 2, 3]
IMU_GYRO    = [4, 5, 6]

MAG_T_COL   = 0
MAG_XYZ     = [1, 2, 3]

GT_T_COL    = 0
GT_POS      = [1, 2, 3]
GT_QUAT_WXYZ = [4, 5, 6, 7]

def _to_vec3(val):
    """Parse a YAML vector that may be a proper list, or a bracketed
    string with ';'/',' separators (some INSANE mag YAMLs use ';')."""
    if isinstance(val, (list, tuple)) and len(val) == 1 and isinstance(val[0], str):
        val = val[0]
    if isinstance(val, str):
        s = val.strip().lstrip('[').rstrip(']').replace(';', ' ').replace(',', ' ')
        return np.fromstring(s, sep=' ')
    return np.asarray(val, dtype=float).ravel()


def _mag_ref_ned_from_var(inc, dec):
    """Unit magnetic-field reference in NED from inclination/declination.
    INSANE's mag_calibration_*.yaml stores inc/dec with the opposite sign
    of the down-positive / east-positive NED convention used here (the
    Mars values -0.8086 / -0.0842 rad reproduce the WMM reference vector
    for the Negev site to ~1e-3). Only the direction is used."""
    inc = -float(inc)
    dec = -float(dec)
    v = np.array([np.cos(inc) * np.cos(dec),
                  np.cos(inc) * np.sin(dec),
                  np.sin(inc)])
    return v / np.linalg.norm(v)


def _find_file_flexible(base_dir, possible_names):
    for name in possible_names:
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            return path
    if os.path.exists(base_dir):
        for root, _, files in os.walk(base_dir):
            for name in possible_names:
                if name in files:
                    return os.path.join(root, name)
    raise FileNotFoundError(f"Cannot find {possible_names} in {base_dir} or its subdirectories.")

def _find_ground_truth_file(dataset_dir):
    possible_names = [
        "ground_truth_80hz.csv", "ground_truth_8hz.csv",
        "ground_truth.csv", "ground_truth.txt",
        "mocap.csv", "mocap.txt", "gt.csv", "gt.txt"
    ]
    try:
        return _find_file_flexible(dataset_dir, possible_names)
    except FileNotFoundError:
        for root, dirs, files in os.walk(dataset_dir):
            if 'ground_truth' in os.path.basename(root).lower() or 'ground_truth' in dirs:
                gt_sub = os.path.join(root, 'ground_truth') if 'ground_truth' in dirs else root
                for f in os.listdir(gt_sub):
                    if f.endswith(('.csv', '.txt')) and not f.endswith('.yaml'):
                        return os.path.join(gt_sub, f)
        raise FileNotFoundError(f"Could not find any Ground Truth file in {dataset_dir}")

def _find_root_dir(dataset_dir):
    current = dataset_dir
    while current:
        if os.path.basename(current) == '9DOF':
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    raise FileNotFoundError("Could not locate '9DOF' root directory.")

def load_mag_calibration_yaml(dataset_dir):
    calib_data = {
        'hard_iron': np.zeros(3),
        'soft_iron': np.eye(3),
        'R_extrinsic': np.eye(3),
        'mag_ref_ned': None,
    }

    # Pick the calibration recorded at the same site as the dataset.
    # mars_* -> Mars analog; indoor/outdoor/transition -> Klagenfurt.
    name = os.path.basename(os.path.normpath(dataset_dir)).lower()
    if ('klu' in name or 'klagenfurt' in name or 'indoor' in name
            or 'outdoor' in name or 'transition' in name):
        site_order = ["mag_calibration_klu.yaml", "mag_calibration_mars.yaml"]
    else:
        site_order = ["mag_calibration_mars.yaml", "mag_calibration_klu.yaml"]
    possible_yaml = site_order + ["sensor_calibration.yaml", "mag_calibration.yaml"]

    # Build an ordered list of candidate YAML paths. The 9DOF-root copies
    # are tried before insane_sensor_calib_preprocessed/ because the
    # latter ships a copy with a malformed flow sequence (missing commas
    # in R_lsmmag_pximu) that yaml.safe_load cannot parse.
    search_dirs = [dataset_dir]
    try:
        root_dir = _find_root_dir(dataset_dir)
        search_dirs += [root_dir,
                        os.path.join(root_dir, "calibrations"),
                        os.path.join(root_dir, "insane_sensor_calib_preprocessed")]
    except Exception:
        pass

    candidate_paths = []
    for sd in search_dirs:
        if not os.path.isdir(sd):
            continue
        try:
            candidate_paths.append(_find_file_flexible(sd, possible_yaml))
        except FileNotFoundError:
            pass

    content = None
    for yaml_path in candidate_paths:
        try:
            with open(yaml_path, 'r') as f:
                parsed = yaml.safe_load(f)
        except Exception:
            continue
        if isinstance(parsed, dict) and (
                'R_pxmag_pximu' in parsed or 'px4_mag_intr' in parsed
                or 'mag_var' in parsed or 'soft_iron' in parsed):
            content = parsed
            break

    if content:
        if True:
            if 'R_pxmag_pximu' in content:
                calib_data['R_extrinsic'] = np.array(content['R_pxmag_pximu'], dtype=float).reshape(3, 3)

            # px4_mag_intr is a YAML list: [{transform: 3x3}, {offset: 3}].
            # Intrinsic model:  m_corr = transform @ (m_raw - offset)
            # (paper Sec. III-B2). The previous code only read [0].transform
            # and looked for a 'px4_mag_bias' key that INSANE never emits,
            # so the hard-iron offset was silently dropped.
            if 'px4_mag_intr' in content and isinstance(content['px4_mag_intr'], list):
                for it in content['px4_mag_intr']:
                    if not isinstance(it, dict):
                        continue
                    if 'transform' in it:
                        calib_data['soft_iron'] = np.array(it['transform'], dtype=float).reshape(3, 3)
                    if 'offset' in it:
                        calib_data['hard_iron'] = _to_vec3(it['offset'])
            if 'soft_iron' in content:
                calib_data['soft_iron'] = np.array(content['soft_iron'], dtype=float).reshape(3, 3)
            if 'px4_mag_bias' in content:
                calib_data['hard_iron'] = _to_vec3(content['px4_mag_bias'])
            elif 'hard_iron' in content:
                calib_data['hard_iron'] = _to_vec3(content['hard_iron'])

            # Local magnetic variation -> world-frame reference direction
            # for the TRIAD attitude bootstrap (paper Tables II-IV).
            if 'mag_var' in content and isinstance(content['mag_var'], list):
                inc = dec = None
                for it in content['mag_var']:
                    if isinstance(it, dict):
                        inc = it.get('inc', inc)
                        dec = it.get('dec', dec)
                if inc is not None and dec is not None:
                    calib_data['mag_ref_ned'] = _mag_ref_ned_from_var(inc, dec)
    return calib_data

def apply_mag_calibration(mag_raw, calib_data):
    hard_iron = calib_data.get('hard_iron', np.zeros(3))
    soft_iron = calib_data.get('soft_iron', np.eye(3))
    R_extrinsic = calib_data.get('R_extrinsic', np.eye(3))
    # Intrinsic: m = t_sp @ (m_raw - b_ct)   (paper Sec. III-B2).
    mag_corrected = (mag_raw - hard_iron) @ soft_iron.T
    # Extrinsic: R_pxmag_pximu maps IMU-frame -> mag-frame vectors (YAML
    # convention: R_A_B takes a vector from frame B to frame A). The
    # measurement is in the mag frame, so rotating it into the IMU frame
    # needs the transpose: v_imu = R_pxmag_pximu.T @ v_mag, i.e. (row
    # vectors) v @ R_extrinsic. Previously used R_extrinsic.T -- the
    # inverse rotation, ~14 deg wrong for the Mars extrinsic.
    mag_corrected = mag_corrected @ R_extrinsic
    return mag_corrected

def load_sensor_data(dataset_dir):
    mag_calib = load_mag_calibration_yaml(dataset_dir)
    imu_path = _find_file_flexible(dataset_dir, ["px4_imu_data.csv", "px4_imu.txt", "px4_imu.csv", "imu.txt", "imu.csv"])
    try:
        imu_data = np.loadtxt(imu_path, delimiter=',', comments='#')
    except ValueError:
        imu_data = np.loadtxt(imu_path, delimiter=',', comments='#', skiprows=1)

    t_imu = imu_data[:, IMU_T_COL]
    accel_raw = imu_data[:, IMU_ACC]
    gyro_raw = imu_data[:, IMU_GYRO]

    # Convert PX4 ENU to Body FRD
    accel_px4_frd = np.zeros_like(accel_raw)
    accel_px4_frd[:, 0] =  accel_raw[:, 1]
    accel_px4_frd[:, 1] =  accel_raw[:, 0]
    accel_px4_frd[:, 2] = -accel_raw[:, 2]

    gyro_px4_frd = np.zeros_like(gyro_raw)
    gyro_px4_frd[:, 0] =  gyro_raw[:, 1]
    gyro_px4_frd[:, 1] =  gyro_raw[:, 0]
    gyro_px4_frd[:, 2] = -gyro_raw[:, 2]

    # PX4 IMU only. The previous code averaged in the LSM9DS1 IMU after a
    # bare ENU->FRD relabel, but LSM is mounted ~42 deg rotated from the
    # PX4 IMU (sensor_calibration.yaml: T_pximu_lsimu) and that rotation
    # was never applied, so the average mixed two misaligned triads. It
    # also used a hard-coded gyro bias of ~1 rad/s (57 deg/s -- not a
    # physical MEMS bias) with no documented origin. Re-enable only after
    # rotating LSM into the PX4 frame with T_pximu_lsimu and deriving the
    # bias properly (e.g. from insane_static_sensor_data).
    accel_frd = accel_px4_frd
    gyro_frd = gyro_px4_frd

    mag_path = _find_file_flexible(dataset_dir, ["px4_mag_data_revised.csv", "px4_mag.txt", "px4_mag.csv"])
    try:
        mag_data = np.loadtxt(mag_path, delimiter=',', comments='#')
    except ValueError:
        mag_data = np.loadtxt(mag_path, delimiter=',', comments='#', skiprows=1)

    t_mag_px4 = mag_data[:, MAG_T_COL]
    mag_px4_raw = mag_data[:, MAG_XYZ]

    mag_px4_calibrated = apply_mag_calibration(mag_px4_raw, mag_calib)
    mag_px4_frd = np.zeros_like(mag_px4_calibrated)
    mag_px4_frd[:, 0] =  mag_px4_calibrated[:, 1]
    mag_px4_frd[:, 1] =  mag_px4_calibrated[:, 0]
    mag_px4_frd[:, 2] = -mag_px4_calibrated[:, 2]

    # PX4 magnetometer only. The previous code averaged in the LSM9DS1
    # magnetometer, but the two streams are in different units (PX4 in
    # Tesla ~3e-5, LSM in Gauss ~0.8) so the mean was ~4 orders of
    # magnitude off, and apply_mag_calibration() was applying the PX4
    # intrinsics/extrinsics (px4_mag_intr, R_pxmag_pximu) to LSM data
    # instead of the LSM ones (lsm_mag_cal_intr, R_lsmmag_pximu).
    mag_out_frd = mag_px4_frd
    t_mag_final = t_mag_px4

    return t_imu, gyro_frd, accel_frd, t_mag_final, mag_out_frd

def load_lsm_imu(dataset_dir):
    """Raw LSM9DS1 IMU (t, accel_frd, gyro_frd), relabelled ENU->FRD like
    the PX4 IMU.

    The LSM runs at ~826 Hz vs the PX4's ~196 Hz, so it RESOLVES the
    motor/prop vibration resonance (~150-260 Hz at hover, bench-confirmed)
    that is above the PX4 Nyquist and ALIASES into the PX4 estimation
    band. It is NOT rotated into the PX4 frame (it sits ~42 deg off,
    T_pximu_lsimu) and is NOT used by the filter -- only as a per-flight
    frequency oracle for vibration_notch.py. A static frame offset does
    not move spectral peaks, so the bare relabel is enough for peak
    tracking. Accel is in m/s^2 here (unlike the LSM magnetometer, which
    is in Gauss -- see load_sensor_data).
    """
    path = _find_file_flexible(dataset_dir, ["lsm_imu.csv", "lsm_imu.txt"])
    try:
        d = np.loadtxt(path, delimiter=',', comments='#')
    except ValueError:
        d = np.loadtxt(path, delimiter=',', comments='#', skiprows=1)
    t = d[:, IMU_T_COL]
    a = d[:, IMU_ACC]
    w = d[:, IMU_GYRO]
    a_frd = np.column_stack((a[:, 1], a[:, 0], -a[:, 2]))
    w_frd = np.column_stack((w[:, 1], w[:, 0], -w[:, 2]))
    return t, a_frd, w_frd

_M_ENU_NED = np.array([[0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0]])


def _enu_rfu_to_ned_frd(pos_enu, quat_raw_wxyz):
    """Shared ENU-world / RFU-body  ->  NED-world / FRD-body conversion:
    pos via M, orientation via the rigid similarity R_ned_frd = M R M
    (M is its own inverse). Used by both the RTK/Wahba GT
    (ground_truth_*.csv) and the mocap GT (mocap_vehicle_data.csv) --
    checked empirically for the indoor mocap: at rest the transformed
    Euler angles are ~level (<2 deg), altitude comes out on -Down."""
    pos_ned = np.column_stack((pos_enu[:, 1], pos_enu[:, 0], -pos_enu[:, 2]))
    quat_ned = np.zeros_like(quat_raw_wxyz)
    for i in range(len(quat_raw_wxyz)):
        n = np.linalg.norm(quat_raw_wxyz[i])
        if n == 0.0:
            quat_ned[i] = np.array([1.0, 0.0, 0.0, 0.0])
            continue
        qn = quat_raw_wxyz[i] / n
        R_enu_rfu = _Rot.from_quat([qn[1], qn[2], qn[3], qn[0]]).as_matrix()
        R_ned_frd = _M_ENU_NED @ R_enu_rfu @ _M_ENU_NED
        x, y_, z, w = _Rot.from_matrix(R_ned_frd).as_quat()
        quat_ned[i] = np.array([w, x, y_, z])
        if i > 0 and np.dot(quat_ned[i], quat_ned[i - 1]) < 0.0:
            quat_ned[i] = -quat_ned[i]
    return pos_ned, quat_ned


def load_mocap_groundtruth(dataset_dir):
    """Indoor GT: the Optitrack motion-capture pose, delivered raw as
    `mocap_vehicle_data.csv` in the flight root (no `ground_truth/` folder
    for the indoor datasets -- that folder is the RTK+Wahba product).
    Format `[t, p_x, p_y, p_z, q_w, q_x, q_y, q_z]` at ~300 Hz.

    The Dronehall Optitrack world is Z-up (ENU-like) and its rigid-body
    frame is RFU (at rest the raw quaternion is ~identity), so the SAME
    `M R M` map as the ENU path applies. This GT is millimeter/sub-degree
    on ALL THREE axes (paper Sec. III-B), unlike the mars/outdoor Wahba GT
    where only yaw+position are trustworthy. Its yaw origin is the
    (arbitrary) Optitrack calibration heading, not true North -- so
    indoor yaw is a DRIFT metric, anchored by seeding the filter's
    initial attitude from mocap (see main.py)."""
    path = _find_file_flexible(dataset_dir, ["mocap_vehicle_data.csv", "mocap.csv"])
    try:
        d = np.loadtxt(path, delimiter=',', comments='#')
    except ValueError:
        d = np.loadtxt(path, delimiter=',', comments='#', skiprows=1)
    pos_ned, quat_ned = _enu_rfu_to_ned_frd(d[:, 1:4], d[:, 4:8])
    return d[:, 0], pos_ned, quat_ned


def load_mocap_as_gps(dataset_dir, decimate_hz=25.0, sigma_pos=0.01, sigma_vel=0.05):
    """Mocap pose as a position+velocity aiding source, shaped exactly
    like load_gps_data()'s return so main.py can feed it through the same
    gps_posvel_update path for the indoor flights (which have no RTK).
    Position = mocap position in NED (M-relabelled), velocity = finite
    difference, decimated to `decimate_hz`. `cov` is a constant small
    diagonal; `fix_info` reports a permanent 3D fix. NOTE: mocap is then
    BOTH the aid and the GT -> indoor position ATE is near-circular and
    not a headline number; the point of the indoor runs is ATTITUDE."""
    t, pos_ned, _q = load_mocap_groundtruth(dataset_dir)
    fs = (len(t) - 1) / (t[-1] - t[0])
    step = max(1, int(round(fs / decimate_hz)))
    t_d, p_d = t[::step], pos_ned[::step]
    v_d = np.gradient(p_d, t_d, axis=0)
    cov = np.tile([sigma_pos ** 2] * 3, (len(t_d), 1))
    fix_info = {'gps_fix_type': np.full(len(t_d), 3, dtype=int),
                'rtk_status': np.full(len(t_d), 1, dtype=int)}
    return t_d, p_d, v_d, cov, fix_info


def load_groundtruth(dataset_dir, frame='enu'):
    """
    Loads GT and converts it from the INSANE toolbox convention
    (ENU world, RFU body, q = R_world_from_body, Hamilton scalar-first --
    verified from aau-cns/insane_dataset_tools: gps_mag_orientation.m,
    mag_gps_wahba.m) to this project's convention (NED world, FRD body).

    frame='mocap' -> load the raw Optitrack pose instead (indoor flights);
    see load_mocap_groundtruth. Same M R M conversion.

    Orientation conversion is the single rigid similarity transform
        R_ned_frd = M @ R_enu_rfu @ M ,   M = [[0,1,0],[1,0,0],[0,0,-1]]
    where M is the ENU<->NED axis relabel and simultaneously the
    RFU<->FRD one (M @ M = I). Checked against the toolbox on mars_1/2/3:
      * yaw then matches the independent dual-RTK baseline heading
        (load_rtk_heading) to <1 deg RMS on mars_1/2;
      * GT body yaw-rate stays positively correlated with the measured
        FRD gyro (applying M on the left only flips that sign).

    NOTE: the toolbox GT roll/pitch come from mag_gps_wahba -- a nearly
    horizontal RTK baseline (weighted x50) plus the ~18%-distorted site
    magnetometer through cross products -- so they carry almost no true
    roll/pitch dynamics (near-zero correlation with the gyro under ANY
    transform). Treat GT roll/pitch as a low-quality reference; only GT
    yaw is trustworthy here. The earlier "conjugate + negate roll&pitch
    in Euler space" mapping was fitted to minimise error against that
    noisy reference, is not a rigid rotation, added a roll handedness
    error and degraded GT yaw-rate consistency; it is replaced here.
    This only affects the GT used for EVALUATION -- the filter is unchanged.
    """
    if frame == 'mocap':
        return load_mocap_groundtruth(dataset_dir)

    gt_path = _find_ground_truth_file(dataset_dir)
    try:
        gt_data = np.loadtxt(gt_path, delimiter=',', comments='#')
    except ValueError:
        gt_data = np.loadtxt(gt_path, delimiter=',', comments='#', skiprows=1)

    t_gt = gt_data[:, GT_T_COL]
    pos_ned, quat_ned = _enu_rfu_to_ned_frd(gt_data[:, GT_POS], gt_data[:, GT_QUAT_WXYZ])
    return t_gt, pos_ned, quat_ned

def _load_gps_time_offset(dataset_dir):
    """Seconds to SUBTRACT from raw rtk_gps*.csv timestamps to put them on
    the PX4 IMU / magnetometer clock (paper Sec. V-C, "Sensor Time
    Calibration"). The value is `t_mag_gps` (raw RTK leads the PX4 clock
    by a few seconds; unshifted, every GPS update is applied that far out
    of sync -- ~10 m at 6 m/s).

    mars / outdoor: it is in the flight-root `time_info.yaml`.
    transition:     the root `time_info.yaml` carries only t_begin/t_end
                    (transition_3 has none), and `t_mag_gps` lives in
                    `ground_truth/time_offset.yaml` -- checked as a
                    fallback here. Returns 0.0 with a warning if neither
                    source has it."""
    for p in (os.path.join(dataset_dir, "time_info.yaml"),
              os.path.join(dataset_dir, "ground_truth", "time_offset.yaml")):
        if not os.path.isfile(p):
            continue
        try:
            with open(p, 'r') as f:
                y = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[data_loader] WARNING: could not read {os.path.basename(p)} ({e})")
            continue
        if 't_mag_gps' in y:
            return float(y['t_mag_gps'])
    print(f"[data_loader] WARNING: no t_mag_gps for {dataset_dir}; "
          f"GPS left un-synchronised to the IMU clock.")
    return 0.0

def load_motor_data(dataset_dir, t_imu, accel=None):
    try:
        motor_path = _find_file_flexible(dataset_dir, ["px4_rpm.txt", "px4_rpm.csv", "rpm.csv"])
        try:
            motor_data = np.loadtxt(motor_path, delimiter=',', comments='#')
        except ValueError:
            motor_data = np.loadtxt(motor_path, delimiter=',', comments='#', skiprows=1)

        t_motor = motor_data[:, 0]
        motors_mean = np.mean(motor_data[:, 1:5], axis=1)
        if np.max(motors_mean) > 2.0:
            motors_mean = motors_mean / 15500.0
        return np.interp(t_imu, t_motor, np.clip(motors_mean, 0.0, 1.0))
    except Exception:
        pass

    if accel is not None:
        norm_a = np.linalg.norm(accel, axis=1)
        return np.clip(np.abs(norm_a - 9.81) / 10.0 + 0.3, 0.0, 1.0)
    return np.zeros(len(t_imu))

def load_gps_data(dataset_dir):
    """
    Loads RTK GPS position + velocity, PLUS the receiver's own reported
    position covariance and fix-quality flags (README.txt: rtk_gps1 format
    is [t, t_gps, lat, long, alt, p_x, p_y, p_z, v_x, v_y, v_z, cov_p_x,
    cov_p_y, cov_p_z, gps_fix_type, rtk_status]).

    Using the receiver's own covariance is a more principled substitute
    for the paper's CASEF/image-quality mechanism (which has no analogue
    for GPS) -- the receiver already tells you how much to trust each
    epoch, per axis, in real units.

    Returns:
        t_gps    : (N,) timestamps
        pos_ned  : (N,3) position [North, East, Down]
        vel_ned  : (N,3) velocity [North, East, Down]
        cov_ned  : (N,3) position variance [North, East, Down], or None
                   if the covariance columns are not present in this file
        fix_info : dict with 'gps_fix_type' and 'rtk_status' arrays, or
                   None if not present
    """
    gps_path = os.path.join(dataset_dir, "rtk_gps1.csv")
    if not os.path.exists(gps_path):
        gps_path = os.path.join(dataset_dir, "rtk_gps1.txt")
    if not os.path.exists(gps_path):
        raise FileNotFoundError(f"Cannot find raw RTK GPS file in {dataset_dir}")

    df = pd.read_csv(gps_path)
    df.columns = df.columns.str.replace('#', '').str.strip()

    # Sensor time calibration (paper Sec. V-C): put raw RTK timestamps on
    # the PX4 IMU / magnetometer clock. ground_truth/rtk_gps1_data_revised
    # .csv is the already-shifted stream but drops the ENU position +
    # covariance + fix columns, so shift the raw timestamps here instead.
    t_gps = df['t'].values - _load_gps_time_offset(dataset_dir)
    p_e, p_n, p_u = df['p_x'].values, df['p_y'].values, df['p_z'].values
    v_e, v_n, v_u = df['v_x'].values, df['v_y'].values, df['v_z'].values

    # Convert to NED World
    pos_ned = np.column_stack((p_n, p_e, -p_u))
    vel_ned = np.column_stack((v_n, v_e, -v_u))

    cov_ned = None
    cov_cols = ['cov_p_x', 'cov_p_y', 'cov_p_z']
    if all(c in df.columns for c in cov_cols):
        cov_e, cov_n, cov_u = df['cov_p_x'].values, df['cov_p_y'].values, df['cov_p_z'].values
        # Variances don't flip sign under the Up->Down negation, only the
        # axis order changes (matching the North/East/Down reorder above).
        cov_ned = np.column_stack((cov_n, cov_e, cov_u))

    fix_info = None
    if 'gps_fix_type' in df.columns or 'rtk_status' in df.columns:
        fix_info = {
            'gps_fix_type': df['gps_fix_type'].values if 'gps_fix_type' in df.columns else None,
            'rtk_status': df['rtk_status'].values if 'rtk_status' in df.columns else None,
        }

    return t_gps, pos_ned, vel_ned, cov_ned, fix_info


# Antenna positions in the PX4 IMU frame (sensor_calibration.yaml:
# p_pximu_rtk1 / p_pximu_rtk2). Same physical vehicle for every INSANE
# recording, so these are constants. Hard-coded because that YAML also
# contains a malformed block (T_pximu_rsimu, spaces instead of commas)
# that makes yaml.safe_load raise on the whole file.
_P_PXIMU_RTK1 = np.array([0.350121933088198, 0.410121933088197, 0.0])
_P_PXIMU_RTK2 = np.array([-0.470121933088198, -0.410121933088197, 0.0])

# Same vector expressed in the filter's FRD body frame: load_sensor_data()
# relabels the PX4 native ("ENU") axes to FRD as (x,y,z) -> (y,x,-z), so
# the RTK1-antenna lever arm from the IMU, in FRD, is (p_y, p_x, -p_z).
# ~0.54 m -- the RTK1 antenna is not the IMU; gps_posvel_update() uses
# this to refer the GPS position/velocity back to the IMU.
P_PXIMU_RTK1_FRD = np.array([_P_PXIMU_RTK1[1], _P_PXIMU_RTK1[0], -_P_PXIMU_RTK1[2]])


def load_rtk_heading(dataset_dir, min_fix_type=3, max_baseline_err=0.3):
    """
    Vehicle heading (NED yaw, radians) from the dual-RTK-GNSS baseline
    rtk_gps2 - rtk_gps1 (paper Sec. V-B) -- independent of the
    magnetometer. Returns (t, yaw, sigma_yaw) on the PX4 IMU clock, or
    (None, None, None) if rtk_gps2 is missing or no epoch is usable.

    Kept only where BOTH receivers report a >=3D fix and the measured
    baseline length matches the known ||p_pximu_rtk2 - p_pximu_rtk1||
    (~1.16 m) to within max_baseline_err.
    """
    p1p = os.path.join(dataset_dir, "rtk_gps1.csv")
    p2p = os.path.join(dataset_dir, "rtk_gps2.csv")
    if not (os.path.isfile(p1p) and os.path.isfile(p2p)):
        return None, None, None

    d1 = pd.read_csv(p1p); d1.columns = d1.columns.str.replace('#', '').str.strip()
    d2 = pd.read_csv(p2p); d2.columns = d2.columns.str.replace('#', '').str.strip()

    toff = _load_gps_time_offset(dataset_dir)
    t1 = d1['t'].values - toff
    t2 = d2['t'].values - toff

    # rtk_gps*.csv: p_x = East, p_y = North, p_z = Up.
    e1, n1 = d1['p_x'].values, d1['p_y'].values
    e2 = np.interp(t1, t2, d2['p_x'].values)
    n2 = np.interp(t1, t2, d2['p_y'].values)

    b_e, b_n = e2 - e1, n2 - n1
    baseline_len = np.hypot(b_e, b_n)
    bearing = np.arctan2(b_e, b_n)                       # world bearing from North

    b_body = _P_PXIMU_RTK2 - _P_PXIMU_RTK1
    body_ang = np.arctan2(b_body[0], b_body[1])          # baseline bearing in body
    b_nom = np.linalg.norm(b_body[:2])

    yaw = (bearing - body_ang + np.pi) % (2.0 * np.pi) - np.pi

    good = np.abs(baseline_len - b_nom) < max_baseline_err
    if 'gps_fix_type' in d1.columns:
        good &= d1['gps_fix_type'].values >= min_fix_type
    if 'gps_fix_type' in d2.columns:
        fix2 = np.round(np.interp(t1, t2, d2['gps_fix_type'].values)).astype(int)
        good &= fix2 >= min_fix_type

    if not np.any(good):
        return None, None, None

    # 1-sigma heading noise from ~1.4 cm (RTK-fix) per-antenna position
    # error over the baseline -> arctan(sqrt(2)*0.014 / 1.16) ~= 1 deg
    # (paper: worst case 0.95 deg).
    sigma_yaw = float(np.arctan2(np.sqrt(2.0) * 0.014, b_nom))
    n_good = int(np.count_nonzero(good))
    return t1[good], yaw[good], np.full(n_good, sigma_yaw)