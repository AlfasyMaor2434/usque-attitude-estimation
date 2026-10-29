"""
replay_phone_raw.py <usque_phone_raw_*.csv> [usque_phone_log_*.txt] [out_dir]

Replays a live_phone_usque raw recording through the Python reference filter (usque.py), in the
same order the live loop ran (trapezoidal gyro integration between samples, update at every 'U'
row), with the same parameters the MATLAB phone script uses. Then:
  * graphs the replay vs the phone app's orientation (same layout as plot_phone_log.py)
  * if the .txt log is given, compares the replay with the MATLAB live estimate
  * runs a few one-change variants and lists, for every resting pose, the heading from the
    filter, from the gyro alone, and from the app -- i.e. how big each turn was according to each.
"""
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import main_usque as mu   # noqa: E402
import usque              # noqa: E402

args = [x for x in sys.argv[1:] if not x.startswith('--')]
SWEEP = '--sweep' in sys.argv
T_END = next((float(x.split('=')[1]) for x in sys.argv if x.startswith('--tend=')), None)
raw = args[0]
log = next((x for x in args[1:] if x.endswith('.txt')), None)
out = next((x for x in args[1:] if not x.endswith(('.txt', '.csv'))), os.path.join(HERE, 'phone_tests'))
os.makedirs(out, exist_ok=True)
tag = os.path.splitext(os.path.basename(raw))[0].replace('usque_phone_raw_', '').replace(' (1)', '')

# ---------------------------------------------------------------- load
hdr, rows = {}, []
for ln in open(raw):
    p = ln.strip().split(',')
    if ln.startswith('#'):
        key = p[0][1:].strip()
        try:
            hdr[key] = np.array([float(x) for x in p[1:]])
        except ValueError:
            pass
    elif p[0] in ('G', 'U'):
        rows.append((p[0], int(p[1]), [float(x) for x in p[2:]]))
frd = lambda v: np.array([v[1], v[0], -v[2]])
off, sc = hdr['mag_cal_offset_frd'], hdr['mag_cal_scale_frd']


def phone_params(**ov):
    """main_usque defaults with the values live_phone_usque / USQUE_AHRS.m use."""
    p = dict(mu.FILTER_PARAMS)
    p.update(gyro_noise_var=9.0e-5, bias_rw_var=1.0e-10, accel_noise_var=1.0e-2, mag_noise_var=0.05,
             use_mag_bias_state=bool(hdr['use_mag_bias_state'][0]), mag_bias_tau_s=300.0,
             mag_bias_sigma_ss_deg=15.0, phys_dir_thresh_deg=5.0, max_phys_inflation=5.0,
             max_vib_inflation=3.0, max_vib_inflation_mag=8.0,
             adaptive_methods=['PHYSICAL', 'IAE', 'VIBRATION', 'MAG_GATE'],
             P0_att=np.radians([2.0, 2.0, 10.0]) ** 2, P0_bias=float(hdr['P0_bg'][0]),
             **({'mag_noise_var': float(hdr['mag_noise_var'][0])} if 'mag_noise_var' in hdr else {}),
             accel_oversample=1.0, mag_oversample=1.0)
    p.update(ov)
    return p


def replay(use_mag=True, declination_deg=None, seed_app_yaw=None, **ov):
    a0, m0, g0 = hdr['boot_accel_frd'], hdr['boot_mag_frd_calibrated'], hdr['boot_gyro_frd']
    down = -a0 / np.linalg.norm(a0)
    incl = np.arcsin(np.clip(m0 / np.linalg.norm(m0) @ down, -1, 1))        # boot inclination
    dec = np.radians(hdr['mag_declination_deg'][0] if declination_deg is None else declination_deg)
    ref = np.array([np.cos(incl) * np.cos(dec), np.cos(incl) * np.sin(dec), np.sin(incl)])
    live_seed = 'init_heading_deg' in hdr and np.isfinite(hdr['init_heading_deg'][0])
    if seed_app_yaw is None:
        seed_app_yaw = live_seed          # default: whatever the live run did
    if seed_app_yaw:
        # Same idea as 'rtk_yaw' / 'mocap_yaw' on the flights: roll/pitch from the sensors, the
        # START heading taken once from the reference (app azimuth + declination), and the
        # magnetic reference rebuilt from that attitude. Nothing from the app is used afterwards.
        az0 = next(v[7] for typ, _, v in rows if typ == 'U' and not np.isnan(v[7]))
        yaw0 = np.radians(hdr['init_heading_deg'][0]) if live_seed else np.radians(az0) + dec
        q0 = mu.set_quat_yaw(usque.USQUE._triad_to_quat(a0, m0, ref), yaw0)
        ov = dict(ov); ov.setdefault('P0_att', np.radians([2.0, 2.0, 2.0]) ** 2)
        f = usque.USQUE(phone_params(**ov), initial_accel=a0, initial_mag=m0, init_q=q0)
    else:
        f = usque.USQUE(phone_params(**ov), initial_accel=a0, initial_mag=m0, mag_ref=ref)
    f.bias_nom = np.array(g0, float)
    T, Q, B, AZ, MAGN, SY = [], [], [], [], [], []
    last_t = last_g = None; last_seg = -1; t_off = 0.0; t_seg0 = 0.0; t_now = 0.0
    for typ, seg, v in rows:
        if typ == 'G':
            tg, gi = v[0], frd(v[1:4])
            if seg != last_seg:
                t_off, t_seg0, last_t, last_seg = t_now, tg, None, seg
            if last_t is not None and 0 < tg - last_t < 0.5:
                f.predict(0.5 * (gi + last_g), tg - last_t)
            last_t, last_g = tg, gi
            t_now = t_off + tg - t_seg0
        else:
            acc = frd(v[1:4]); mag = (frd(v[4:7]) - off) * sc
            f.update(acc, mag if use_mag else None, throttle_pct=0.0)
            T.append(t_now); Q.append(f.q_nom.copy()); B.append(np.degrees(f.bias_nom))
            AZ.append(v[7:10]); MAGN.append(np.linalg.norm(mag)); SY.append(np.degrees(np.sqrt(f.P[2, 2])))
    return dict(t=np.array(T), q=np.array(Q), bg=np.array(B), app=np.array(AZ), magn=np.array(MAGN),
                sig_yaw=np.array(SY), adapt=f.adaptation_summary())


def rot(q):
    return Rot.from_quat(np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]]))


def vs_app(r, remove_offset=True):
    R = rot(r['q']); app = r['app']; ok = ~np.isnan(app[:, 0])
    ypr = R.as_euler('ZYX', degrees=True)
    off0 = np.nanmedian(((ypr[:60, 0] - app[:60, 0] + 180) % 360) - 180) if remove_offset else 0.0
    Ra = Rot.from_euler('ZYX', np.column_stack([app[:, 0] + off0, -app[:, 1], app[:, 2]])[ok], degrees=True)
    ang = np.full(len(ypr), np.nan); ang[ok] = np.degrees((R[ok].inv() * Ra).magnitude())
    return ypr, ang, off0


def wrap_break(y):
    y = np.asarray(y, float).copy()
    y[np.r_[False, np.abs(np.diff(y)) > 180]] = np.nan
    return y


# ---------------------------------------------------------------- replay with the final settings + figure
# Recordings made before the script wrote 'init_heading_deg' / 'mag_noise_var' are replayed with the
# settings the script uses now (start heading seeded once from the app, mag_noise_var 3.0).
FINAL = {} if 'init_heading_deg' in hdr else dict(seed_app_yaw=True, mag_noise_var=3.0)
DECL = float(hdr['mag_declination_deg'][0])
base = replay(**FINAL)
t, app = base['t'], base['app'].copy()
app[:, 0] += DECL                                             # app azimuth: magnetic -> true north
Rb = rot(base['q'])
ypr = Rb.as_euler('ZYX', degrees=True)
okk = ~np.isnan(app[:, 0])
ang = np.full(len(t), np.nan)
ang[okk] = np.degrees((Rb[okk].inv() * Rot.from_euler('ZYX', np.column_stack([app[:, 0], -app[:, 1], app[:, 2]])[okk],
                                                      degrees=True)).magnitude())
gl = np.abs(ypr[:, 1]) > 75


def continuous(angle, bad):
    o = np.full(len(angle), np.nan); last = None
    for i, a_ in enumerate(angle):
        if bad[i] or np.isnan(a_):
            continue
        o[i] = a_ if last is None else last + ((a_ - last + 180) % 360 - 180)
        last = o[i]
    return o


def shade(ax, mask, color, label=None):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for a_, b_ in zip(e[::2], e[1::2]):
        ax.axvspan(t[a_], t[min(b_, len(t) - 1)], color=color, alpha=0.18, lw=0, label=label); label = None


yaw_c, roll_c = continuous(ypr[:, 0], gl), continuous(ypr[:, 2], gl)
app_yaw_c = yaw_c + ((app[:, 0] - ypr[:, 0] + 180) % 360 - 180)
app_roll_c = roll_c + ((app[:, 2] - ypr[:, 2] + 180) % 360 - 180)
sel = np.ones(len(t), bool) if T_END is None else t <= T_END
suffix = '' if T_END is None else f'_0-{T_END:.0f}s'
fig, axs = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
for ax, est, ap_, lab in ((axs[0], yaw_c, app_yaw_c, 'heading (deg, continuous)'),
                          (axs[1], ypr[:, 1], -app[:, 1], 'pitch (deg)'),
                          (axs[2], roll_c, app_roll_c, 'roll (deg, continuous)')):
    ax.plot(t[sel], est[sel], color='tab:blue', lw=1.4, label='USQUE estimate')
    ax.plot(t[sel], ap_[sel], color='tab:orange', lw=1.1, ls='--', label="phone's built-in orientation (MATLAB Mobile)")
    shade(ax, gl & sel, 'grey', '|pitch| > 75 deg: heading and roll not defined')
    ax.set_ylabel(lab); ax.grid(alpha=0.3)
axs[0].legend(fontsize=8, ncol=3, loc='best')
axs[3].plot(t[sel], ang[sel], color='tab:red', lw=1.0, label='total angle between the two orientations (absolute)')
axs[3].set_ylim(0, max(5.0, min(40.0, np.nanpercentile(ang[sel], 99.5) * 1.2)))
axs[3].set_ylabel('difference (deg)'); axs[3].grid(alpha=0.3); axs[3].legend(fontsize=8, loc='upper left')
axs[4].plot(t[sel], base['bg'][sel, 0], lw=1.0, label='x'); axs[4].plot(t[sel], base['bg'][sel, 1], lw=1.0, label='y')
axs[4].plot(t[sel], base['bg'][sel, 2], lw=1.3, label='z')
axs[4].set_ylabel('gyro bias estimate (deg/s)'); axs[4].set_xlabel('time (s)'); axs[4].grid(alpha=0.3)
axs[4].legend(fontsize=8, ncol=3)
fig.suptitle(f'Phone test {tag}: USQUE (offline replay of the recorded sensor data) vs the phone orientation')
fig.tight_layout(); fig.savefig(os.path.join(out, f'phone_{tag}{suffix}_4_replay.png'), dpi=160); plt.close(fig)
m_ = sel & ~gl
print(f"final settings {FINAL or '(as recorded)'}: absolute difference to the app, {t[sel][-1]:.0f} s"
      f"{'' if T_END is None else ' (cropped)'}: median {np.nanmedian(ang[m_]):.2f}, p90 {np.nanpercentile(ang[m_], 90):.2f}, "
      f"max {np.nanmax(ang[m_]):.2f} deg, above 10 deg {np.nanmean(ang[m_] > 10) * 100:.1f}% | "
      f"bgZ {base['bg'][0, 2]:+.4f} -> {base['bg'][sel][-1, 2]:+.4f} deg/s")

# ---------------------------------------------------------------- replay vs the MATLAB live estimate
if log and not FINAL:          # only meaningful when the replay uses the settings of the live run
    pat = re.compile(r"bgZ:\s*(\S+) dps.*?app az:\s*(\S+) p:\s*(\S+) r:\s*(\S+)\s+\|\s+q:\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")
    L = np.array([[float(x) for x in m.groups()] for m in pat.finditer(open(log, encoding='utf-8', errors='ignore').read())])
    # The log has no timestamps. Each print line is matched to the update row carrying the same
    # app orientation (searched forward from the previous match), which pins it to within one
    # app-orientation sample even during fast motion.
    idx = np.zeros(len(L), dtype=int); j = 0
    guess = np.linspace(0, len(t) - 1, len(L)).astype(int)
    for i in range(len(L)):
        lo, hi = max(j, guess[i] - 150), min(len(t), max(j, guess[i]) + 150)
        dd = np.nansum(np.abs(base["app"][lo:hi] - L[i, 1:4]), axis=1) if hi > lo else np.array([0.0])
        near = np.flatnonzero(dd <= dd.min() + 0.06)
        j = lo + int(near[np.argmin(np.abs(near + lo - guess[i]))])
        idx[i] = j
    tl = t[idx]
    L = L[:, [0, 4, 5, 6, 7]]
    Rl = rot(L[:, 1:5])
    d = np.degrees((rot(base['q'][idx]).inv() * Rl).magnitude())
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axs[0].plot(tl, d, color='tab:red', lw=1.0); axs[0].set_ylabel('deg'); axs[0].grid(alpha=0.3)
    axs[0].set_title('angle between the Python replay and the MATLAB live estimate')
    axs[1].plot(tl, L[:, 0], label='MATLAB live bgZ'); axs[1].plot(t, base['bg'][:, 2], ls='--', label='Python replay bgZ')
    axs[1].set_ylabel('deg/s'); axs[1].set_xlabel('time (s)'); axs[1].grid(alpha=0.3); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, f'phone_{tag}_5_replay_vs_matlab.png'), dpi=140); plt.close(fig)
    print(f"replay vs MATLAB live: median {np.median(d):.2f} deg, p90 {np.percentile(d, 90):.2f}, max {d.max():.2f}")

if SWEEP:
    # ---------------------------------------------------------------- variants
    # ABSOLUTE comparison: the app azimuth (magnetic north) + site declination = true north, the same
    # reference the filter uses. No offset is fitted.


    def abs_vs_app(r):
        R = rot(r['q']); ap_ = r['app']; ok = ~np.isnan(ap_[:, 0])
        Ra = Rot.from_euler('ZYX', np.column_stack([ap_[:, 0] + DECL, -ap_[:, 1], ap_[:, 2]])[ok], degrees=True)
        a_ = np.full(len(ap_), np.nan); a_[ok] = np.degrees((R[ok].inv() * Ra).magnitude())
        return a_


    NOGATE = ['PHYSICAL', 'IAE', 'VIBRATION']
    S = dict(seed_app_yaw=True)
    variants = {
        'as live': {},
        'seed': dict(S),
        'seed + mag R 1.0': dict(S, mag_noise_var=1.0),
        'seed + mag R 3.0': dict(S, mag_noise_var=3.0),
        'seed + gate off + mag R 3.0': dict(S, mag_noise_var=3.0, adaptive_methods=NOGATE),
        'seed + gate off + mag R 10': dict(S, mag_noise_var=10.0, adaptive_methods=NOGATE),
        'seed + gate off + mag R 30': dict(S, mag_noise_var=30.0, adaptive_methods=NOGATE),
        'seed + psiMag + gate off + mag R 1.0': dict(S, use_mag_bias_state=True, mag_noise_var=1.0, adaptive_methods=NOGATE),
        'seed + psiMag + gate off + mag R 3.0': dict(S, use_mag_bias_state=True, mag_noise_var=3.0, adaptive_methods=NOGATE),
        'no magnetometer (gyro + accel)': dict(S, use_mag=False),
    }
    # samples where the phone is at rest or moving slowly (the app's orientation lags during fast
    # shaking, which is a timing difference, not an attitude error)
    w_ = np.degrees(np.linalg.norm(gy_all := np.array([frd(v[1:4]) for typ, _, v in rows if typ == 'G']), axis=1))
    tg_all = np.linspace(0, t[-1], len(w_))
    slow = np.interp(t, tg_all, np.convolve(w_, np.ones(50) / 50, mode='same')) < 20.0
    res = {}
    print()
    print(f"absolute total angle vs the app, phone at rest / slow motion ({slow.mean() * 100:.0f}% of the run)")
    print(f"{'variant':38s} {'median':>7s} {'p90':>6s} {'>10 deg':>8s} | first 150 s: {'median':>6s} {'max':>6s} | max|bgZ|")
    for name, kw in variants.items():
        r = replay(**kw); res[name] = r
        a_ = abs_vs_app(r)
        e1 = a_[slow]; e2 = a_[slow & (t < 150)]
        print(f"{name:38s} {np.nanmedian(e1):7.2f} {np.nanpercentile(e1, 90):6.2f} {np.nanmean(e1 > 10) * 100:7.1f}% |"
              f"              {np.nanmedian(e2):6.2f} {np.nanmax(e2):6.2f} | {np.abs(r['bg'][:, 2]).max():.3f} deg/s")

    # ---------------------------------------------------------------- resting poses: heading per source
    print("\nResting flat poses (>= 3 s still, |pitch|,|roll| < 10 deg): heading at each, and the turn since the previous pose")
    print(f"{'t (s)':>12s} | {'filter':>8s} {'gyro only':>10s} {'app':>8s} | turn: {'filter':>8s} {'gyro only':>10s} {'app':>8s} | bgZ")
    yf = ypr[:, 0]; yg = vs_app(res['no magnetometer (gyro + accel)'])[0][:, 0]
    ya = app[:, 0]
    rate = np.r_[0, np.abs(np.diff(np.unwrap(np.radians(yg)))) / np.maximum(np.diff(t), 1e-3)]
    still = (np.degrees(rate) < 1.0) & (np.abs(ypr[:, 1]) < 10) & (np.abs(ypr[:, 2]) < 10)
    edges = np.flatnonzero(np.diff(np.r_[0, still.astype(int), 0]))
    prev = None
    wrap = lambda x: (x + 180) % 360 - 180
    for a, b in zip(edges[::2], edges[1::2]):
        b = min(b, len(t) - 1)
        if t[b] - t[a] < 3.0:
            continue
        s = slice(a + (b - a) // 2, b)                                    # second half of the rest
        cm = lambda y: np.degrees(np.arctan2(np.nanmean(np.sin(np.radians(y[s]))), np.nanmean(np.cos(np.radians(y[s])))))
        cur = (cm(yf), cm(yg), cm(ya))
        turn = ('', '', '') if prev is None else tuple(f"{wrap(c - p):+8.1f}" for c, p in zip(cur, prev))
        print(f"{t[a]:5.0f}-{t[b]:5.0f} | {cur[0]:8.1f} {cur[1]:10.1f} {cur[2]:8.1f} | turn: {turn[0]:>8s} {turn[1]:>10s} {turn[2]:>8s} | {base['bg'][b, 2]:+.3f}")
        prev = cur
print('saved to', out)
