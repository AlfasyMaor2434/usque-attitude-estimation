"""How much of the yaw-vs-RTK error is the comparison itself?
(a) Euler yaw vs baseline heading (current metric) vs (b) baseline bearing predicted from the full
estimated attitude (tilt-consistent), and (c) the RTK-vs-IMU time lag that minimises the error."""
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FLIGHTS = [f'mars_{i}' for i in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11)]


def task(name):
    import usque_eval as ue
    import data_loader as dl0
    mu = ue.mu
    mu.FUSION_RATE_HZ = 30.0; mu.PER_FLIGHT_MAG_CALIB = True; mu.RTK_SEED_AVG_S = 0.0
    r = mu.run_flight(ue.dataset_dir(name), 'mars', 'rtk_yaw', params=ue.make_params({}),
                      verbose=False, want_ar=False, want_extras=False)
    t_hdg, yaw_hdg, _ = ue.dl.load_rtk_heading(ue.dataset_dir(name))
    e0, e1 = r['t_cmp'][0], r['t_cmp'][-1]
    b = dl0._P_PXIMU_RTK2 - dl0._P_PXIMU_RTK1
    b_frd = np.array([b[1], b[0], -b[2]])
    body_ang = np.arctan2(b[0], b[1])
    q = r['q_est']
    R = Rot.from_quat(np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]]))
    yaw_eul = R.as_euler('ZYX')[:, 0]
    bw = R.apply(b_frd)
    yaw_base = np.arctan2(bw[:, 1], bw[:, 0]) - body_ang
    tilt = np.degrees(np.arccos(np.clip(R.apply([0, 0, 1])[:, 2], -1, 1)))
    wrap = lambda x: (x + np.pi) % (2 * np.pi) - np.pi
    out = dict(name=name)
    for lag in np.arange(-0.20, 0.2001, 0.02):
        m = (t_hdg >= e0) & (t_hdg <= e1)
        th = t_hdg[m] + lag
        ye = np.interp(th, r['t_imu'], np.unwrap(yaw_eul)); yb = np.interp(th, r['t_imu'], np.unwrap(yaw_base))
        ee = np.degrees(np.abs(wrap(ye - yaw_hdg[m]))); eb = np.degrees(np.abs(wrap(yb - yaw_hdg[m])))
        out[round(lag, 2)] = (np.mean(ee < 1), np.mean(eb < 1), np.sqrt(np.mean(eb ** 2)))
    out['tilt_p50'] = float(np.median(tilt[(r['t_imu'] >= e0) & (r['t_imu'] <= e1)]))
    out['tilt_p90'] = float(np.percentile(tilt[(r['t_imu'] >= e0) & (r['t_imu'] <= e1)], 90))
    return out


if __name__ == '__main__':
    with Pool(10) as p:
        res = p.map(task, FLIGHTS)
    lags = [k for k in res[0] if isinstance(k, float)]
    print("flight   tilt p50/p90 | <1 euler@0 | <1 baseline@0 | best lag (baseline) -> <1, rmse")
    for o in res:
        best = max(lags, key=lambda L: o[L][1])
        print(f"{o['name']:8s} {o['tilt_p50']:5.1f}/{o['tilt_p90']:5.1f} | {o[0.0][0]:.2f} | {o[0.0][1]:.2f} | "
              f"{best:+.2f}s -> {o[best][1]:.2f}, {o[best][2]:.2f}")
    for L in lags:
        print(f"lag {L:+.2f}s: mean <1 euler {np.mean([o[L][0] for o in res]):.3f}  baseline {np.mean([o[L][1] for o in res]):.3f}")
