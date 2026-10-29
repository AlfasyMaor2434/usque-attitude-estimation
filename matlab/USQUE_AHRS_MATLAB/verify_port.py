"""
verify_port.py -- is USQUE_AHRS.m the same filter as usque.py?

Runs one raw phone recording through BOTH implementations (MATLAB in batch mode, Python here),
in four configurations that together exercise every branch of the filter, and compares the
outputs update by update: attitude (angle between the two quaternions), gyro bias, the
magnetic heading-bias state and the 1-sigma values from the covariance.

    python verify_port.py <usque_phone_raw_*.csv>

Both sides integrate the same gyro samples with the same trapezoid rule, and split a gap longer
than 0.2 s into equal sub-steps (USQUE_AHRS.predict ignores a single step longer than 0.2 s).
"""
import os
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import main_usque as mu   # noqa: E402
import usque              # noqa: E402

MATLAB = os.environ.get('MATLAB_EXE', 'matlab')     # path to matlab(.exe) if it is not on PATH
raw = os.path.abspath(sys.argv[1])
TMP = os.path.join(HERE, 'phone_tests', 'verify')
os.makedirs(TMP, exist_ok=True)
MODES = ['live', 'psi', 'accel', 'seeded']
frd = lambda v: np.array([v[1], v[0], -v[2]])

hdr, rows = {}, []
for ln in open(raw):
    p = ln.strip().split(',')
    if ln.startswith('#'):
        try:
            hdr[p[0][1:].strip()] = np.array([float(x) for x in p[1:]])
        except ValueError:
            pass
    elif p[0] in ('G', 'U'):
        rows.append((p[0], int(p[1]), [float(x) if x else np.nan for x in p[2:]]))


def py_replay(mode):
    p = dict(mu.FILTER_PARAMS)
    p.update(gyro_noise_var=9.0e-5, bias_rw_var=1.0e-10, accel_noise_var=1.0e-2,
             mag_noise_var=3.0 if mode == 'seeded' else 0.05,
             use_mag_bias_state=(mode == 'psi'), mag_bias_tau_s=300.0, mag_bias_sigma_ss_deg=15.0,
             mag_gate_mag_frac=0.15, mag_gate_incl_deg=10.0, max_mag_gate_inflation=50.0,
             phys_dir_thresh_deg=5.0, max_phys_inflation=5.0, max_vib_inflation=3.0, max_vib_inflation_mag=8.0,
             adaptive_methods=['PHYSICAL', 'IAE', 'VIBRATION', 'MAG_GATE'],
             P0_att=np.radians([2.0, 2.0, 2.0 if mode == 'seeded' else 10.0]) ** 2, P0_bias=float(hdr['P0_bg'][0]),
             accel_oversample=1.0, mag_oversample=1.0, alpha=1.0, beta=2.0, kappa=1.0, gamma=1.0,
             q_scale_factor=0.0, q_inflation_coupling=0.0, phys_mag_test=False, iae_per_channel=True,
             use_corrected_Q=True, use_Q_cross_term=True, use_sigma0_reference=True, q_tune_scale=1.0, r_tune_scale=1.0)
    a0, m0, g0 = hdr['boot_accel_frd'], hdr['boot_mag_frd_calibrated'], hdr['boot_gyro_frd']
    down = -a0 / np.linalg.norm(a0)
    incl = np.arcsin(np.clip(m0 / np.linalg.norm(m0) @ down, -1, 1))
    dec = np.radians(hdr['mag_declination_deg'][0])
    ref = np.array([np.cos(incl) * np.cos(dec), np.cos(incl) * np.sin(dec), np.sin(incl)])
    if mode == 'seeded':
        az0 = next(v[7] for typ, _, v in rows if typ == 'U' and not np.isnan(v[7]))
        q0 = mu.set_quat_yaw(usque.USQUE._triad_to_quat(a0, m0, ref), np.radians(az0) + dec)
        f = usque.USQUE(p, initial_accel=a0, initial_mag=m0, init_q=q0)
    else:
        f = usque.USQUE(p, initial_accel=a0, initial_mag=m0, mag_ref=ref)
    f.bias_nom = np.array(g0, float)
    off, sc = hdr['mag_cal_offset_frd'], hdr['mag_cal_scale_frd']
    out, long_steps = [], 0
    last_t = last_g = None; last_seg = -1
    for typ, seg, v in rows:
        if typ == 'G':
            tg, gi = v[0], frd(v[1:4])
            if seg != last_seg:
                last_t, last_seg = None, seg
            if last_t is not None and 0 < tg - last_t < 0.5:
                n_sub = int(np.ceil((tg - last_t) / 0.2))       # same sub-stepping as the MATLAB side
                long_steps += n_sub > 1
                for _ in range(n_sub):
                    f.predict(0.5 * (gi + last_g), (tg - last_t) / n_sub)
            last_t, last_g = tg, gi
        else:
            acc = frd(v[1:4]); mag = (frd(v[4:7]) - off) * sc
            f.update(acc, None if mode == 'accel' else mag, throttle_pct=0.0)
            out.append(np.r_[f.q_nom, f.bias_nom, f.psi_mag_nom, np.sqrt(np.maximum(np.diag(f.P), 0))])
    return np.array(out), long_steps


# ---------------------------------------------------------------- MATLAB side (one batch call)
cmd = "cd('%s'); " % HERE.replace('\\', '/') + ' '.join(
    "verify_replay_dump('%s','%s','%s');" % (raw.replace('\\', '/'), os.path.join(TMP, f'ml_{m}.csv').replace('\\', '/'), m) for m in MODES)
print('running MATLAB ...', flush=True)
r = subprocess.run([MATLAB, '-batch', cmd], capture_output=True, text=True)
print(r.stdout.strip()[-600:], r.stderr.strip()[-600:])

print(f"\n{'mode':8s} {'updates':>8s} | attitude diff (deg): {'median':>10s} {'max':>10s} | bias diff (deg/s) max | psi diff (deg) max | sigma diff max (rel)")
worst = 0.0
for m in MODES:
    ml = np.loadtxt(os.path.join(TMP, f'ml_{m}.csv'), delimiter=',')
    py, long_steps = py_replay(m)
    n = min(len(ml), len(py))
    Rm = Rot.from_quat(np.column_stack([ml[:n, 1], ml[:n, 2], ml[:n, 3], ml[:n, 0]]))
    Rp = Rot.from_quat(np.column_stack([py[:n, 1], py[:n, 2], py[:n, 3], py[:n, 0]]))
    ang = np.degrees((Rm.inv() * Rp).magnitude())
    db = np.degrees(np.abs(ml[:n, 4:7] - py[:n, 4:7])).max()
    dpsi = np.degrees(np.abs(ml[:n, 7] - py[:n, 7])).max()
    sig_rel = (np.abs(ml[:n, 8:15] - py[:n, 8:15]) / np.maximum(np.abs(py[:n, 8:15]), 1e-12))
    sig_rel = sig_rel[:, :6].max() if m != 'psi' else sig_rel.max()      # psi sigma is a fixed 1e-6 floor when the state is off
    worst = max(worst, ang.max())
    print(f"{m:8s} {n:8d} | {'':21s}{np.median(ang):10.2e} {ang.max():10.2e} | {db:21.2e} | {dpsi:18.2e} | {sig_rel:.2e}"
          + (f"   ({long_steps} gyro gaps > 0.2 s, sub-stepped on both sides)" if long_steps else ''))
print(f"\nworst attitude difference over all modes: {worst:.3e} deg")
