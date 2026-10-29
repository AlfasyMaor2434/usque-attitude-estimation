import os, numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # the repository root
from data_path import DATA_DIR as BASE  # noqa: E402
for fl in (1, 2, 3):
    d = os.path.join(BASE, f"indoor_{fl}_sensors")
    ls = np.loadtxt(os.path.join(d, "lsm_imu.csv"), delimiter=',', skiprows=1)
    rpm = np.loadtxt(os.path.join(d, "px4_rpm.csv"), delimiter=',', skiprows=1)
    t_on = rpm[np.argmax(rpm[:, 1:5].mean(1) > 0.05), 0]
    a, w, t = ls[:, 1:4], ls[:, 4:7], ls[:, 0]
    on = t >= t_on
    print(f"indoor_{fl}: accel min {np.round(a.min(0),2)} max {np.round(a.max(0),2)} | gyro min {np.round(w.min(0),1)} max {np.round(w.max(0),1)}")
    for k, ax in enumerate('xyz'):
        mx = np.abs(a[:, k]).max()
        print(f"   accel {ax}: |a|>= 0.99*max in {np.mean(np.abs(a[on, k]) >= 0.99*mx)*100:.2f}% of motors-on samples; "
              f"std on {a[on,k].std():.2f} off {a[~on,k].std():.3f}; mean on {a[on,k].mean():.2f} off {a[~on,k].mean():.2f}")
    uniq = len(np.unique(np.round(a[:, 2], 6)))
    print(f"   distinct accel-z values {uniq}, smallest step {np.min(np.diff(np.unique(a[:,2]))):.5f}")
