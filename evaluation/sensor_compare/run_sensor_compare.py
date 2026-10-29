"""
run_sensor_compare.py -- USQUE on indoor_1..3 with the PX4 IMU/mag vs the LSM9DS1 IMU/mag.

Same filter, same final config (main_usque.FILTER_PARAMS), same mocap-seeded start and
scoring (usque_eval.run_one); only the sensor streams change.

Variants
  px4            PX4 IMU, no mag           (the report's indoor configuration)
  px4_mag        PX4 IMU + PX4 mag
  lsm            LSM IMU, no mag           (raw: unit conversion + YAML extrinsic only)
  lsm_mag        LSM IMU + LSM mag         (raw)
  lsm_sc         LSM IMU, gyro scale x LSM_GYRO_SCALE, no mag
  lsm_sc_mag     LSM IMU scale-corrected + LSM mag

LSM preprocessing (checked in check_lsm_frames*.py / check_lsm_scale.py):
  * gyro is in deg/s (bias ~1 deg/s) -> rad/s
  * IMU extrinsic: v_px4 = R v_lsm with R = T_pximu_lsimu[:3,:3] (Kabsch fit agrees within 2-3 deg)
  * mag: m = T (m_raw - b) (lsm_mag_cal_intr), then v_px4 = R_lsmmag_pximu v -- the direction the
    data supports (18 deg median vs the PX4 mag; the name-convention direction gives 52 deg)
  * then the same ENU-label -> FRD relabel as the PX4 streams, native rate (~900 Hz IMU, 20 Hz mag)
  * static-bias still detection: gyro threshold 0.1 rad/s (the LSM's own bias is up to 3 deg/s) and
    no |a| test (indoor_3's LSM accel reads |a| = 6.7 m/s^2 at rest)
LSM_GYRO_SCALE = PX4/LSM gyro ratio from a least-squares fit on all 3 flights (1.15-1.18 on every
axis) -- it uses the PX4 as the reference, so the *_sc variants show what a gyro scale calibration
would recover, not a result the LSM achieves on its own.
"""
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # evaluation/ (usque_eval)

sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # the repository root
from data_path import DATA_DIR as BASE  # noqa: E402
FLIGHTS = ['indoor_1', 'indoor_2', 'indoor_3']
VARIANTS = ['px4', 'px4_mag', 'lsm', 'lsm_mag', 'lsm_sc', 'lsm_sc_mag']
LSM_GYRO_SCALE = 1.171

R_PX_LS = np.array([[0.7430, 0.6692, 0.0115], [-0.6693, 0.7430, 0.0038], [-0.0060, -0.0106, 0.9999]])
R_LSMAG_PX = np.array([[0.732181316045914, 0.680823204661011, -0.0197556176411876],
                       [-0.680199708586795, 0.732394264252452, 0.0304466439619994],
                       [0.0351976827605520, -0.00885469848278986, 0.999341141674386]])
T_LSMAG = np.array([[0.947106840979225, -0.00393083215420979, -0.0154279452288296],
                    [-0.00393083215420980, 0.989242063563763, 0.000610967073493479],
                    [-0.0154279452288296, 0.000610967073493424, 0.995204818108842]])
B_LSMAG = np.array([0.184132371546752, 0.423699254058575, -0.0693055332559639])


def _frd(v):
    """ENU-labelled sensor axes -> FRD, same relabel as data_loader for the PX4 streams."""
    return np.column_stack((v[:, 1], v[:, 0], -v[:, 2]))


def load_lsm(dataset_dir, gyro_scale=1.0):
    ls = np.loadtxt(os.path.join(dataset_dir, "lsm_imu.csv"), delimiter=',', skiprows=1)
    lm = np.loadtxt(os.path.join(dataset_dir, "lsm_mag.csv"), delimiter=',', skiprows=1)
    t = ls[:, 0]
    accel = _frd(ls[:, 1:4] @ R_PX_LS.T)
    gyro = _frd(np.radians(ls[:, 4:7]) @ R_PX_LS.T) * gyro_scale
    m = ((lm[:, 1:4] - B_LSMAG) @ T_LSMAG.T) @ R_LSMAG_PX.T
    m = _frd(m)
    mag = np.column_stack([np.interp(t, lm[:, 0], m[:, k]) for k in range(3)])
    return t, gyro, accel, mag


def task(args):
    name, variant, boresight = args
    import usque_eval as ue
    mu, dl = ue.mu, ue.dl
    mu.BORESIGHT_FIXED_RPY_DEG = {} if boresight is None else {f"{name}_sensors": boresight}
    orig_load, orig_bias = dl.load_sensor_data, dl.load_static_gyro_bias
    use_mag = variant.endswith('_mag')
    dl.SITE_PROFILE['indoor']['mag'] = use_mag
    if variant.startswith('lsm'):
        scale = LSM_GYRO_SCALE if variant.startswith('lsm_sc') else 1.0
        dl.load_sensor_data = lambda d, _s=scale: load_lsm(d, _s)
        dl.load_static_gyro_bias = lambda d, t_imu=None, gyro=None, accel=None, **k: orig_bias(
            d, t_imu=t_imu, gyro=gyro, accel=None, gyro_still_thresh=0.1)
    t0 = time.perf_counter()
    try:
        o = ue.run_one(name, 'indoor', {}, keep_arrays=True)
    finally:
        dl.load_sensor_data, dl.load_static_gyro_bias = orig_load, orig_bias
        dl.SITE_PROFILE['indoor']['mag'] = False
    o['variant'] = variant
    o['runtime_s'] = time.perf_counter() - t0
    a = o['arrays']
    o['yaw_err'] = np.abs(a['axis_err'][:, 2])
    o['bg_final_dps'] = a['bg'][-1].tolist()
    return o


FIELDS = ['variant', 'name', 'err_lt1', 'err_lt3', 'err_p50', 'err_p90', 'err_rmse',
          'rmse_roll', 'rmse_pitch', 'rmse_yaw', 'anees', 'anis', 'dof_meas', 'in3sig',
          'bg_final_dps', 'boresight_fit_rpy_deg', 'runtime_s']


def row(o):
    r = {k: o.get(k) for k in FIELDS}
    r['rmse_roll'], r['rmse_pitch'], r['rmse_yaw'] = o['att_rmse_rpy']
    return r


if __name__ == '__main__':
    t0 = time.perf_counter()
    with Pool(9) as p:
        # 1) PX4 runs fit each flight's mocap boresight (rig calibration: mocap body frame vs PX4 IMU)
        px4 = p.map(task, [(n, 'px4', None) for n in FLIGHTS], chunksize=1)
        bore = {o['name']: o['boresight_fit_rpy_deg'] for o in px4}
        print("PX4 boresight (deg):", {k: np.round(v, 2).tolist() for k, v in bore.items()}, flush=True)
        # 2) every variant scored with that SAME boresight (LSM is rotated into the PX4 frame first)
        jobs = [(n, v, bore[n]) for v in VARIANTS for n in FLIGHTS]
        jobs.sort(key=lambda j: (not j[1].startswith('lsm'), j[0] != 'indoor_1'))   # longest first
        res = p.map(task, jobs, chunksize=1)
    print(f"all runs {time.perf_counter() - t0:.0f} s")
    res.sort(key=lambda o: (VARIANTS.index(o['variant']), o['name']))

    out_csv = os.path.join(HERE, 'sensor_compare.csv')
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for o in res:
            w.writerow(row(o))
        # pooled per variant (all scored samples of the 3 flights together)
        for v in VARIANTS:
            rs = [o for o in res if o['variant'] == v]
            e = np.concatenate([o['arrays']['err'] for o in rs])
            ax = np.concatenate([o['arrays']['axis_err'] for o in rs])
            nees = np.concatenate([o['arrays']['nees'] for o in rs])
            nis = np.concatenate([o['arrays']['nis'] for o in rs])
            w.writerow(dict(variant=v, name='pooled', err_lt1=np.mean(e < 1), err_lt3=np.mean(e < 3),
                            err_p50=np.median(e), err_p90=np.percentile(e, 90),
                            err_rmse=np.sqrt(np.mean(e ** 2)),
                            rmse_roll=np.sqrt(np.nanmean(ax[:, 0] ** 2)),
                            rmse_pitch=np.sqrt(np.nanmean(ax[:, 1] ** 2)),
                            rmse_yaw=np.sqrt(np.nanmean(ax[:, 2] ** 2)),
                            anees=np.mean(nees), anis=np.mean(nis), dof_meas=rs[0]['dof_meas'],
                            in3sig=np.mean([o['in3sig'] for o in rs])))
    print(f"wrote {out_csv}")

    # arrays for the figure script
    np.savez_compressed(os.path.join(HERE, 'sensor_compare_arrays.npz'),
                        **{f"{o['variant']}__{o['name']}__{k}": np.asarray(o['arrays'][k])
                           for o in res for k in ('t_cmp', 'err', 'axis_err', 't_nis', 'nis', 'nees',
                                                  'sig_att', 't_bg', 'bg')})
    with open(os.path.join(HERE, 'sensor_compare_meta.json'), 'w') as fh:
        json.dump(dict(flights=FLIGHTS, variants=VARIANTS, lsm_gyro_scale=LSM_GYRO_SCALE), fh, indent=1)
