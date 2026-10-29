import os, numpy as np
from scipy.signal import butter, filtfilt
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the repository root
from data_path import DATA_DIR as BASE  # noqa: E402
R = np.array([[0.7430, 0.6692, 0.0115], [-0.6693, 0.7430, 0.0038], [-0.0060, -0.0106, 0.9999]])
def lpf(x, fs, fc):
    b, a = butter(4, fc / (0.5 * fs)); return filtfilt(b, a, x, axis=0)
for fl in (1, 2, 3):
    d = os.path.join(BASE, f"indoor_{fl}_sensors")
    px = np.loadtxt(os.path.join(d, "px4_imu.csv"), delimiter=',', skiprows=1)
    ls = np.loadtxt(os.path.join(d, "lsm_imu.csv"), delimiter=',', skiprows=1)
    tp, wp = px[:, 0], lpf(px[:, 4:7], 200, 5)
    tl = ls[:, 0]; wl = lpf(np.radians(ls[:, 4:7]), 1 / np.median(np.diff(tl)), 5) @ R.T
    m = (tp > tl[0] + 1) & (tp < tl[-1] - 1)
    wi = np.column_stack([np.interp(tp[m], tl, wl[:, k]) for k in range(3)])
    b = np.median(wi[np.linalg.norm(wp[m], axis=1) < 0.02], 0)
    wi -= b
    A = np.column_stack([wi, np.ones(len(wi))])
    M, *_ = np.linalg.lstsq(A, wp[m], rcond=None)   # PX4 = LSM @ M[:3] + c
    print(f"indoor_{fl}: per-axis scale PX4/LSM (diag of LS 3x3 fit) {np.round(np.diag(M[:3]), 3)}; "
          f"integrated yaw-rate (z) PX4 {np.degrees(np.sum(np.abs(wp[m][:,2]))*0.005):.0f} deg vs LSM {np.degrees(np.sum(np.abs(wi[:,2]))*0.005):.0f} deg; "
          f"net z PX4 {np.degrees(np.sum(wp[m][:,2])*0.005):.0f}, LSM {np.degrees(np.sum(wi[:,2])*0.005):.0f}")
