"""Empirical LSM->PX4 rotations (Kabsch) for gyro and mag, vs the YAML extrinsics; LSM units; mag quality."""
import os, sys
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as Rot
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

def lpf(x, fs, fc):
    b, a = butter(4, fc / (0.5 * fs)); return filtfilt(b, a, x, axis=0)

def kabsch(A, B):
    """R minimising |B - A R^T| (rows): b_i = R a_i."""
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T

def rotdiff(R1, R2):
    return np.degrees(np.linalg.norm(Rot.from_matrix(R1 @ R2.T).as_rotvec()))

for fl in (1, 2, 3):
    d = os.path.join(BASE, f"indoor_{fl}_sensors")
    px = np.loadtxt(os.path.join(d, "px4_imu.csv"), delimiter=',', skiprows=1)
    ls = np.loadtxt(os.path.join(d, "lsm_imu.csv"), delimiter=',', skiprows=1)
    pm = np.loadtxt(os.path.join(d, "px4_mag.csv"), delimiter=',', skiprows=1)
    lm = np.loadtxt(os.path.join(d, "lsm_mag.csv"), delimiter=',', skiprows=1)
    rpm = np.loadtxt(os.path.join(d, "px4_rpm.csv"), delimiter=',', skiprows=1)
    t_on = rpm[np.argmax(rpm[:, 1:5].mean(1) > 0.05), 0]
    tp, wp, ap = px[:, 0], px[:, 4:7], px[:, 1:4]
    tl, wl, al = ls[:, 0], ls[:, 4:7], ls[:, 1:4]
    fsl = 1 / np.median(np.diff(tl))
    wlf, alf = lpf(wl, fsl, 20), lpf(al, fsl, 20)
    wpf, apf = lpf(wp, 200, 20), lpf(ap, 200, 20)
    m = (tp > tl[0] + 1) & (tp < tl[-1] - 1)
    wli = np.column_stack([np.interp(tp[m], tl, wlf[:, k]) for k in range(3)])
    ali = np.column_stack([np.interp(tp[m], tl, alf[:, k]) for k in range(3)])
    # units: rotation-rate magnitude ratio during motion
    mov = np.linalg.norm(wpf[m], axis=1) > 0.3
    ratio = np.median(np.linalg.norm(wli[mov] - np.median(wli[~mov], 0), axis=1) / np.linalg.norm(wpf[m][mov], axis=1))
    print(f"\n== indoor_{fl}: motors on at +{t_on - tp[0]:.1f}s; |LSM gyro|/|PX4 gyro| while turning = {ratio:.2f} "
          f"(57.3 => LSM in deg/s); LSM gyro bias (still, deg/s) {np.round(np.median(wli[~mov], 0), 3)}")
    wl_rad = np.radians(wli - np.median(wli[~mov], 0))
    Rg = kabsch(wl_rad, wpf[m])
    res = np.linalg.norm(wpf[m] - wl_rad @ Rg.T, axis=1)
    print(f"  gyro Kabsch R(LSM->PX4): rpy {np.round(Rot.from_matrix(Rg).as_euler('xyz', True), 2)}  "
          f"vs YAML R_px_ls {rotdiff(Rg, R_px_ls):.2f} deg, vs YAML^T {rotdiff(Rg, R_px_ls.T):.2f} deg; "
          f"residual rms {np.degrees(np.sqrt(np.mean(res**2))):.2f} deg/s")
    Ra = kabsch(ali, apf[m])
    print(f"  accel Kabsch R(LSM->PX4): rpy {np.round(Rot.from_matrix(Ra).as_euler('xyz', True), 2)}  vs gyro-Kabsch {rotdiff(Ra, Rg):.2f} deg; "
          f"|LSM a| median {np.median(np.linalg.norm(ali,axis=1)):.3f}, |PX4 a| {np.median(np.linalg.norm(apf[m],axis=1)):.3f}")
    # mag
    cal = dl0.load_mag_calibration_yaml(d)
    pmc = dl0.apply_mag_calibration(pm[:, 1:4], cal)
    lmc = (lm[:, 1:4] - b_ls) @ T_ls.T
    pmi = np.column_stack([np.interp(lm[:, 0], pm[:, 0], pmc[:, k]) for k in range(3)])
    ok = (lm[:, 0] > pm[0, 0]) & (lm[:, 0] < pm[-1, 0])
    for lab, sel in (("motors OFF", ok & (lm[:, 0] < t_on)), ("motors ON ", ok & (lm[:, 0] >= t_on))):
        nl = np.linalg.norm(lmc[sel], axis=1); npx = np.linalg.norm(pmi[sel], axis=1)
        print(f"  {lab}: |LSM mag| {np.median(nl)*100:.1f} uT (std {np.std(nl)/np.median(nl)*100:.1f}%), "
              f"|PX4 mag| {np.median(npx)*1e6:.1f} uT (std {np.std(npx)/np.median(npx)*100:.1f}%)  n={sel.sum()}")
    lu = lmc[ok] / np.linalg.norm(lmc[ok], axis=1, keepdims=True)
    pu = pmi[ok] / np.linalg.norm(pmi[ok], axis=1, keepdims=True)
    Rm = kabsch(lu, pu)
    ang = np.degrees(np.arccos(np.clip(np.sum((lu @ Rm.T) * pu, 1), -1, 1)))
    print(f"  mag Kabsch R(LSMmag->PX4): rpy {np.round(Rot.from_matrix(Rm).as_euler('xyz', True), 2)} vs YAML R_lsmag_px^T {rotdiff(Rm, R_lsmag_px.T):.2f}, "
          f"vs YAML R_lsmag_px {rotdiff(Rm, R_lsmag_px):.2f} deg; residual angle median {np.median(ang):.2f} p90 {np.percentile(ang,90):.2f}")
    angY = np.degrees(np.arccos(np.clip(np.sum((lu @ R_lsmag_px) * pu, 1), -1, 1)))
    offm = lm[ok, 0] < t_on
    print(f"  with YAML (v_px = R^T v): angle median motors-off {np.median(angY[offm]):.2f}, motors-on {np.median(angY[~offm]):.2f} deg")
