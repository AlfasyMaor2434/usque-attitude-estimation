"""Verify LSM9DS1 units, frame rotation direction and time alignment against the PX4 IMU/mag."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # the repository root
sys.path.insert(0, ROOT)
import data_loader as dl0

from data_path import DATA_DIR as BASE  # noqa: E402
R_px_ls = np.array([[0.7430, 0.6692, 0.0115], [-0.6693, 0.7430, 0.0038], [-0.0060, -0.0106, 0.9999]])
R_lsmag_px = np.array([[0.732181316045914, 0.680823204661011, -0.0197556176411876],
                       [-0.680199708586795, 0.732394264252452, 0.0304466439619994],
                       [0.0351976827605520, -0.00885469848278986, 0.999341141674386]])
T_ls = np.array([[0.947106840979225, -0.00393083215420979, -0.0154279452288296],
                 [-0.00393083215420980, 0.989242063563763, 0.000610967073493479],
                 [-0.0154279452288296, 0.000610967073493424, 0.995204818108842]])
b_ls = np.array([0.184132371546752, 0.423699254058575, -0.0693055332559639])

for fl in (1, 2, 3):
    d = os.path.join(BASE, f"indoor_{fl}_sensors")
    px = np.loadtxt(os.path.join(d, "px4_imu.csv"), delimiter=',', skiprows=1)
    ls = np.loadtxt(os.path.join(d, "lsm_imu.csv"), delimiter=',', skiprows=1)
    pm = np.loadtxt(os.path.join(d, "px4_mag.csv"), delimiter=',', skiprows=1)
    lm = np.loadtxt(os.path.join(d, "lsm_mag.csv"), delimiter=',', skiprows=1)
    tp, wp = px[:, 0], px[:, 4:7]
    tl, wl, al = ls[:, 0], ls[:, 4:7], ls[:, 1:4]
    print(f"\n== indoor_{fl}: PX4 {len(tp)} @ {1/np.median(np.diff(tp)):.0f} Hz, LSM {len(tl)} @ {1/np.median(np.diff(tl)):.0f} Hz, "
          f"LSM mag {len(lm)} @ {1/np.median(np.diff(lm[:,0])):.1f} Hz; t0 diff LSM-PX4 {tl[0]-tp[0]:+.3f}s")
    # first-second means (raw, ENU-labelled sensor frames)
    print("  PX4 gyro mean first 1s", np.round(wp[tp < tp[0] + 1].mean(0), 4), " LSM gyro mean first 1s", np.round(wl[tl < tl[0] + 1].mean(0), 4))
    print("  LSM |accel| first 1s", np.round(np.linalg.norm(al[tl < tl[0] + 1].mean(0)), 3))
    # resample LSM gyro onto PX4 grid, try R and R.T, degrees-vs-rad, find lag
    for name, R in (("R", R_px_ls), ("R.T", R_px_ls.T)):
        wl_px = wl @ R.T
        best = None
        for lag in np.arange(-0.2, 0.2001, 0.005):
            wi = np.column_stack([np.interp(tp, tl + lag, wl_px[:, k]) for k in range(3)])
            m = (tp > tl[0] + 1) & (tp < tl[-1] - 1)
            wpm, wim = wp[m] - wp[m].mean(0), wi[m] - wi[m].mean(0)
            c = np.sum(wpm * wim) / np.sqrt(np.sum(wpm ** 2) * np.sum(wim ** 2))
            if best is None or c > best[1]:
                scale = np.sum(wpm * wim) / np.sum(wim ** 2)
                best = (lag, c, scale)
        print(f"  gyro {name:3s}: best lag {best[0]:+.3f}s corr {best[1]:.4f} scale PX4/LSM {best[2]:.4f}")
    # magnetometer: PX4 calibrated (as in data_loader) vs LSM calibrated, both in PX4 IMU frame (raw ENU-labelled)
    cal = dl0.load_mag_calibration_yaml(d)
    pmc = dl0.apply_mag_calibration(pm[:, 1:4], cal)
    lmc = (lm[:, 1:4] - b_ls) @ T_ls.T
    pu = pmc / np.linalg.norm(pmc, axis=1, keepdims=True)
    for name, R in (("v@R (R.T v)", R_lsmag_px), ("v@R.T (R v)", R_lsmag_px.T)):
        l_px = lmc @ R
        lu = l_px / np.linalg.norm(l_px, axis=1, keepdims=True)
        lui = np.column_stack([np.interp(pm[:, 0], lm[:, 0], lu[:, k]) for k in range(3)])
        lui /= np.linalg.norm(lui, axis=1, keepdims=True)
        ang = np.degrees(np.arccos(np.clip(np.sum(lui * pu, 1), -1, 1)))
        print(f"  mag {name}: angle LSM vs PX4 direction median {np.median(ang):.2f} deg, p90 {np.percentile(ang, 90):.2f}")
    print(f"  |mag| PX4 cal median {np.median(np.linalg.norm(pmc,axis=1))*1e6:.1f} uT (std {np.std(np.linalg.norm(pmc,axis=1))/np.median(np.linalg.norm(pmc,axis=1))*100:.1f}%), "
          f"LSM cal median {np.median(np.linalg.norm(lmc,axis=1))*100:.1f} uT (std {np.std(np.linalg.norm(lmc,axis=1))/np.median(np.linalg.norm(lmc,axis=1))*100:.1f}%)")
