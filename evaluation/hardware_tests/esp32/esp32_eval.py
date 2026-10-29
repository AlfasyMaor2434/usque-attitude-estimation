"""
esp32_eval.py -- numbers and figures for the ESP32 hardware tests.

    python esp32_eval.py       -> results.json, figures/*.png

Input: the 3D-viewer logs in data/ (5 Oct 2026 tests + the 2 Oct raw-magnetometer log).
There is no reference orientation: accuracy = size of known turns and return-to-start error.
"""
import json
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, FIG = os.path.join(HERE, 'data'), os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)
un = lambda a: np.degrees(np.unwrap(np.radians(a)))
wrap = lambda x: (np.asarray(x) + 180.0) % 360.0 - 180.0


def load(name):
    """-> t (s from first data line), dict of arrays (NaN where a field is missing), marks, other text"""
    rows, marks, text, keys = [], [], [], set()
    for ln in open(os.path.join(DATA, name), encoding='utf-8', errors='ignore'):
        m = re.search(r"\+([\d.]+)s\] (RAW|MARK)\s+(.*)", ln)
        if not m:
            continue
        if m.group(2) == 'MARK':
            marks.append((float(m.group(1)), m.group(3).strip())); continue
        if not m.group(3).startswith('roll_deg:'):
            text.append(m.group(3)); continue
        try:
            d = {k: float(v) for k, v in (kv.split(':') for kv in m.group(3).strip().split(','))}
        except ValueError:
            continue
        d['_t'] = float(m.group(1)); rows.append(d); keys |= set(d)
    T = np.array([r['_t'] for r in rows])
    D = {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys}
    marks = [(mt - T[0], s) for mt, s in marks if not s.startswith(('---', '==='))]
    return T - T[0], D, marks, '\n'.join(text)


def plateaus(t, y, rate_lim=2.0, min_s=1.5):
    m = np.abs(np.gradient(y, t)) < rate_lim
    e = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0])); out = []
    for a, b in zip(e[::2], e[1::2]):
        b = min(b, len(t) - 1)
        if t[b] - t[a] >= min_s:
            out.append((float(t[a]), float(t[b]), float(np.median(y[a:b]))))
    return out


def rest_mask(t, D):
    return (np.abs(np.gradient(D['pitch_deg'], t)) + np.abs(np.gradient(un(D['roll_deg']), t))
            + np.abs(np.gradient(un(D['yaw_deg']), t))) < 1.5


def longest(mask, t):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    a, b = max(zip(e[::2], e[1::2]), key=lambda ab: t[min(ab[1], len(t) - 1)] - t[ab[0]])
    return a, min(b, len(t) - 1)


def quality(D):
    g = lambda k: D[k] if k in D else np.full(1, np.nan)
    return dict(imu_hz=float(np.nanmedian(g('imuHz'))), imu_hz_min=float(np.nanmin(g('imuHz')[5:] if len(g('imuHz')) > 5 else g('imuHz'))),
                bad=float(g('badSmp')[-1] - g('badSmp')[0]), rd_err=float(g('rdErr')[-1] - g('rdErr')[0]),
                cfg_fix=float(g('cfgFix')[-1] - g('cfgFix')[0]), acc_fault=float(np.nanmean(g('accFault'))))


def orient_fig(t, D, fname, title, marks=(), extra=None, tmax=None):
    """heading / pitch / roll (continuous), filter 3-sigma band; optional 4th panel"""
    m = np.ones(len(t), bool) if tmax is None else t <= tmax
    n = 4 if extra else 3
    fig, axs = plt.subplots(n, 1, figsize=(10.5, 2.2 * n + 0.6), sharex=True)
    held = (D['yawHeld'] > 0) if 'yawHeld' in D else np.zeros(len(t), bool)
    for ax, y, s, lab in ((axs[0], un(D['yaw_deg']), D['sigYaw_deg'], 'heading (deg)'), (axs[1], D['pitch_deg'], D['sigPitch_deg'], 'pitch (deg)'),
                          (axs[2], un(D['roll_deg']), D['sigRoll_deg'], 'roll (deg)')):
        ax.plot(t[m], y[m], color='tab:blue', lw=1.3, label='USQUE estimate')
        e = np.flatnonzero(np.diff(np.r_[0, (held & m).astype(int), 0])); lab2 = 'near +-90 deg pitch (heading held)'
        for a, b in zip(e[::2], e[1::2]):
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color='grey', alpha=0.2, lw=0, label=lab2); lab2 = None
        for mt, _ in marks:
            if tmax is None or mt <= tmax:
                ax.axvline(mt, color='k', ls=':', lw=0.8)
        ax.set_ylabel(lab); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8, ncol=2, loc='best')
    if extra:
        extra(axs[3], m)
        axs[3].grid(alpha=0.3)
    axs[-1].set_xlabel('time (s)')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=160); plt.close(fig)


def bias_panel(t, D):
    def f(ax, m):
        for k, lab in (('gyroBiasX_dps', 'x'), ('gyroBiasY_dps', 'y'), ('gyroBiasZ_dps', 'z')):
            ax.plot(t[m], D[k][m], lw=1.2, label=lab)
        ax.set_ylabel('gyro bias estimate (deg/s)'); ax.legend(fontsize=8, ncol=3)
    return f


R = {}

# ============================================================== before / after: data quality
t0, O, _, _ = load('old1_static.txt'); t1, N, _, txt1 = load('t1_static_seeded.txt')
an = lambda D: np.sqrt(D['accX'] ** 2 + D['accY'] ** 2 + D['accZ'] ** 2)
gz = lambda D: np.column_stack([D['gyrX_dps'], D['gyrY_dps'], D['gyrZ_dps']])
early, late = t0 < 40, (t0 > 100) & (np.abs(gz(O)).max(1) < 10)
i_acc = int(np.argmax(an(O) > 12))
t2o, O2, _, _ = load('old2_static.txt')
R['quality_old'] = dict(dur=float(t0[-1]), gyro_std_early=float(gz(O)[early].std(0).max()), gyro_std_late=float(gz(O)[late].std(0).max()),
                        spikes=int((np.abs(gz(O)).max(1) > 10).sum()), t_acc_step=float(t0[i_acc]), acc_before=float(np.median(an(O)[t0 < 40])),
                        acc_after=float(np.median(an(O)[t0 > t0[i_acc] + 2])), pitch_start=float(O['pitch_deg'][0]), pitch_end=float(O['pitch_deg'][-1]),
                        yaw_end=float(O['yaw_deg'][-1]), yaw_jump=float(np.abs(np.diff(O['yaw_deg'])).max()),
                        old2_dur=float(t2o[-1]), old2_yaw_end=float(un(O2['yaw_deg'])[-1]))
R['quality_new'] = dict(gyro_std=float(gz(N).std(0).max()), acc=float(np.median(an(N))), acc_std=float(an(N).std()), **quality(N))
fig, axs = plt.subplots(2, 2, figsize=(11, 5.6), sharex='col')
for j, (t, D, ttl) in enumerate(((t0, O, 'before the fix (static, old read path)'), (t1, N, 'after the fix (static)'))):
    axs[0, j].plot(t, D['gyrZ_dps'], lw=0.7, color='tab:orange'); axs[0, j].set_ylim(-12, 12); axs[0, j].set_title(ttl, fontsize=10)
    axs[0, j].set_ylabel('raw gyro z (deg/s)'); axs[0, j].grid(alpha=0.3)
    axs[1, j].plot(t, an(D), lw=0.9, color='tab:green'); axs[1, j].axhline(9.81, color='k', ls='--', lw=0.8); axs[1, j].set_ylim(8, 18)
    axs[1, j].set_ylabel('accelerometer |a| (m/s^2)'); axs[1, j].set_xlabel('time (s)'); axs[1, j].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'quality_before_after.png'), dpi=160); plt.close(fig)

# ============================================================== magnetometer
tm, M, _, _ = load('mag_raw_oct2.txt'); mn = np.sqrt(M['magRawX_uT'] ** 2 + M['magRawY_uT'] ** 2 + M['magRawZ_uT'] ** 2); mm = tm < 40
_, _, _, txt7 = load('t7_erratic2.txt')
g = lambda pat, s=txt7: (lambda m: m.groups() if m else None)(re.search(pat, s))
dd = {}
a = g(r"A\. WIA2=0x(\w+).*?CNTL2=0x(\w+)"); dd['wia2'], dd['cntl2'] = a
dd['mst'] = g(r"B\. I2C_MST_STATUS=0x(\w+)")[0]
for key, lab in (('auto', 'C. AUTOMATIC path'), ('direct', 'D. DIRECT path')):
    q = re.search(lab + r": (\d+) samples \| DRDY set (\d+), DOR set (\d+), HOFL \(overflow\) set (\d+).*?std \(max\) ([\d.]+) uT", txt7)
    dd[key] = dict(n=int(q.group(1)), hofl=int(q.group(4)), std=float(q.group(5)))
dd['selftest'] = [int(x) for x in g(r"E\. SELF-TEST ST1=0x\w+ X=(-?\d+) Y=(-?\d+) Z=(-?\d+) counts")]
dd['raw_median'], dd['raw_p5'], dd['raw_p95'] = float(np.median(mn[mm])), float(np.percentile(mn[mm], 5)), float(np.percentile(mn[mm], 95))
R['mag'] = dd
fig, ax = plt.subplots(figsize=(10.5, 3.0))
ax.semilogy(tm[mm], mn[mm], '.', ms=3, color='tab:blue', label='ESP32 magnetometer reading (board still)')
ax.axhspan(25, 65, color='tab:green', alpha=0.25, label="Earth's field range (25-65 uT)")
ax.set_xlabel('time (s)'); ax.set_ylabel('field strength (uT, log)'); ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'mag_raw.png'), dpi=160); plt.close(fig)

# ============================================================== 1 static, seeded
t, D = t1, N
seed = [float(x) for x in re.search(r"still_gyro_dps:(\S+?),(\S+?),(\S+?),", txt1).groups()]
true = gz(D).mean(0)
yaw = un(D['yaw_deg'])
R['static'] = dict(dur=float(t[-1]), roll_std=float(D['roll_deg'].std()), pitch_std=float(D['pitch_deg'].std()),
                   yaw_drift=float(yaw[-1] - yaw[0]), yaw_rate=float(np.polyfit(t, yaw, 1)[0]), seed=seed, true=true.tolist(),
                   seed_err_z=float(seed[2] - true[2]), bias_end=[float(D[k][-1]) for k in ('gyroBiasX_dps', 'gyroBiasY_dps', 'gyroBiasZ_dps')],
                   **quality(D))
fig, axs = plt.subplots(2, 1, figsize=(10.5, 5.0), sharex=True)
axs[0].plot(t, yaw, color='tab:blue', label='heading'); axs[0].set_ylabel('heading (deg)'); axs[0].grid(alpha=0.3); axs[0].legend(fontsize=8)
axs[1].plot(t, D['roll_deg'] - D['roll_deg'][0], label='roll - start'); axs[1].plot(t, D['pitch_deg'] - D['pitch_deg'][0], label='pitch - start')
axs[1].set_ylim(-0.5, 0.5); axs[1].set_ylabel('change (deg)'); axs[1].set_xlabel('time (s)'); axs[1].grid(alpha=0.3); axs[1].legend(fontsize=8, ncol=2)
fig.suptitle('Test 1: board at rest, bias seeded', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(FIG, 't1_static.png'), dpi=160); plt.close(fig)

# ============================================================== 2 static, no seed
t, D, _, txt = load('t2_static_noseed.txt')
seed2 = [float(x) for x in re.search(r"still_gyro_dps:(\S+?),(\S+?),(\S+?),", txt).groups()]
raw = gz(D); true2 = raw.mean(0); bias = np.column_stack([D['gyroBiasX_dps'], D['gyroBiasY_dps'], D['gyroBiasZ_dps']])
sig = np.column_stack([D['sigBiasX_dps'], D['sigBiasY_dps'], D['sigBiasZ_dps']])
conv = []
for i in range(3):
    ok = np.abs(bias[:, i] - true2[i]) < 0.05
    last_bad = np.flatnonzero(~ok)
    conv.append(float(t[last_bad[-1] + 1]) if len(last_bad) and last_bad[-1] + 1 < len(t) else (0.0 if ok.all() else float('nan')))
yaw = un(D['yaw_deg'])
R['noseed'] = dict(dur=float(t[-1]), true=true2.tolist(), bias_end=bias[-1].tolist(), sig_start=sig[0].tolist(), sig_end=sig[-1].tolist(),
                   conv_s=conv, bias_10s=bias[np.searchsorted(t, 10)].tolist(), bias_20s=bias[np.searchsorted(t, 20)].tolist(),
                   yaw_rate=float(np.polyfit(t, yaw, 1)[0]), yaw_drift=float(yaw[-1] - yaw[0]),
                   window_value=seed2, window_err=[float(a - b) for a, b in zip(seed2, true2)],
                   corrected_end=(raw[t > t[-1] - 30] - bias[t > t[-1] - 30]).mean(0).tolist(), **quality(D))
fig, axs = plt.subplots(3, 2, figsize=(12, 7.2), sharex=True)
for i, axn in enumerate('xyz'):
    ax = axs[i, 0]
    ax.fill_between(t, bias[:, i] - 3 * sig[:, i], bias[:, i] + 3 * sig[:, i], color='tab:blue', alpha=0.15, lw=0, label='filter 3-sigma')
    ax.plot(t, bias[:, i], color='tab:blue', lw=1.4, label='bias estimate')
    ax.axhline(true2[i], color='tab:red', ls='--', lw=1.0, label='true bias (mean raw gyro at rest)')
    ax.axhline(0, color='k', lw=0.5); ax.set_ylabel(f'gyro bias {axn} (deg/s)'); ax.grid(alpha=0.3)
    ax = axs[i, 1]
    ax.plot(t, raw[:, i], color='tab:orange', lw=0.8, label='raw gyro'); ax.plot(t, raw[:, i] - bias[:, i], color='tab:green', lw=0.8, label='raw gyro - bias estimate')
    ax.axhline(0, color='k', lw=0.5); ax.set_ylim(-1.6, 1.0); ax.set_ylabel(f'rate {axn} (deg/s)'); ax.grid(alpha=0.3)
axs[0, 0].legend(fontsize=8); axs[0, 1].legend(fontsize=8)
axs[0, 0].set_title('bias estimate, started from zero', fontsize=10); axs[0, 1].set_title('gyro rate before / after removing the bias', fontsize=10)
axs[2, 0].set_xlabel('time (s)'); axs[2, 1].set_xlabel('time (s)')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 't2_bias_convergence.png'), dpi=160); plt.close(fig)

# ============================================================== 3 yaw turns
t, D, marks, _ = load('t3_yaw.txt'); yaw = un(D['yaw_deg']); pl = plateaus(t, yaw)
t_slide = next(mt for mt, s in marks if 'shifting' in s)
first_step = next(p[2] for p in pl if p[2] > 45) - 0.0
full = next(p for p in pl if p[1] - p[0] > 20)                      # long rest after the first four turns
back = [p for p in pl if p[0] > full[1] + 30 and p[0] < t_slide]
sl = t > t_slide
rest = rest_mask(t, D); a, b = longest(rest & (t < t_slide), t)
R['yaw'] = dict(dur=float(t[-1]), start=float(yaw[0]), first_step=float(next(p[2] for p in pl if p[2] > 45) - next(p[2] for p in pl if p[0] > 3)),
                full_circle=float(full[2]), full_err=float(wrap(full[2] - yaw[0])), back=float(back[-1][2]) if back else float('nan'),
                back_err=float(back[-1][2] - yaw[0]) if back else float('nan'), end=float(yaw[-1]), end_err=float(yaw[-1] - yaw[0]),
                slide_min=float(yaw[sl].min() - np.median(yaw[sl])), slide_max=float(yaw[sl].max() - np.median(yaw[sl])),
                rest_rate=float(np.polyfit(t[a:b], yaw[a:b], 1)[0]), roll_change=float(D['roll_deg'][-1] - D['roll_deg'][0]),
                pitch_change=float(D['pitch_deg'][-1] - D['pitch_deg'][0]), t_slide=float(t_slide), **quality(D))
orient_fig(t, D, 't3_yaw.png', 'Test 3: 90 deg yaw turns (then sliding the board without turning it)', marks)

# ============================================================== 4 pitch turns
t, D, marks, _ = load('t4_pitch.txt'); yaw = un(D['yaw_deg'])
flat = plateaus(t, D['pitch_deg'])
R['pitch'] = dict(dur=float(t[-1]), max_pitch=float(np.abs(D['pitch_deg']).max()), pitch_start=float(D['pitch_deg'][0]), pitch_end=float(D['pitch_deg'][-1]),
                  flat_pitches=[p[2] for p in flat], roll_change=float(D['roll_deg'][-1] - D['roll_deg'][0]), yaw_change=float(yaw[-1] - yaw[0]),
                  held_frac=float(np.mean(D['yawHeld'] > 0)), **quality(D))
orient_fig(t, D, 't4_pitch.png', 'Test 4: pitch turns up to vertical', marks)

# ============================================================== 5 roll turns
t, D, marks, _ = load('t5_roll.txt'); yaw = un(D['yaw_deg']); roll = un(D['roll_deg'])
R['roll'] = dict(dur=float(t[-1]), total=float(roll[-1] - roll[0]), closure=float(wrap(roll[-1] - roll[0])), pitch_change=float(D['pitch_deg'][-1] - D['pitch_deg'][0]),
                 yaw_change=float(yaw[-1] - yaw[0]), **quality(D))
orient_fig(t, D, 't5_roll.png', 'Test 5: roll turns, one full circle', marks)

# ============================================================== 6 erratic (two runs)
for key, name in (('erratic1', 't6_erratic1.txt'), ('erratic2', 't7_erratic2.txt')):
    t, D, marks, _ = load(name)
    if key == 'erratic2':                                           # first lines belong to the previous run (before the reset)
        k0 = int(np.argmax(np.abs(np.diff(D['yaw_deg'])) > 5) + 1) if np.abs(D['yaw_deg'][0]) > 5 else 0
        t, D = t[k0:] - t[k0], {k: v[k0:] for k, v in D.items()}
        marks = [(mt - (t[0] if False else 0), s) for mt, s in marks]
    yaw = un(D['yaw_deg']); rates = np.abs(gz(D))
    mov = rates.max(1) > 20; i0, i1 = np.flatnonzero(mov)[0], np.flatnonzero(mov)[-1]
    b = lambda i: [float(D[k][i]) for k in ('gyroBiasX_dps', 'gyroBiasY_dps', 'gyroBiasZ_dps')]
    after = t > t[i1] + 3
    R[key] = dict(dur=float(t[-1]), move_dur=float(t[i1] - t[i0]), max_rate=float(rates.max()), sat_prints=int((rates > 450).any(1).sum()),
                  roll_err=float(wrap(D['roll_deg'][-1] - D['roll_deg'][0])), pitch_err=float(D['pitch_deg'][-1] - D['pitch_deg'][0]),
                  yaw_end_wrapped=float(wrap(D['yaw_deg'][-1] - D['yaw_deg'][0])), bias_before=b(max(i0 - 5, 0)), bias_after=b(-1),
                  post_rate=float(np.polyfit(t[after], yaw[after], 1)[0]) if after.sum() > 20 else float('nan'), **quality(D))
    if key == 'erratic2':
        sd = [float(x) for x in re.search(r"still_gyro_dps:(\S+?),(\S+?),(\S+?),", txt7).groups()]
        R[key]['seed'] = sd
        R[key]['true_rest'] = gz(D)[:max(i0 - 5, 1)].mean(0).tolist()
        R[key]['rest_before_s'] = float(t[i0])
        R[key]['rest_rate_before'] = float(np.polyfit(t[:max(i0 - 5, 2)], yaw[:max(i0 - 5, 2)], 1)[0])
        def extra(ax, m, t=t, D=D):
            for k, lab in (('gyroBiasX_dps', 'bias x'), ('gyroBiasY_dps', 'bias y'), ('gyroBiasZ_dps', 'bias z')):
                ax.plot(t[m], D[k][m], lw=1.2, label=lab)
            ax.set_ylabel('gyro bias estimate (deg/s)'); ax.legend(fontsize=8, ncol=3, loc='upper left')
            ax2 = ax.twinx(); ax2.plot(t[m], (D['rdErr'] - D['rdErr'][0])[m], color='k', lw=1.0, ls='--'); ax2.set_ylabel('bus read errors (count, dashed)')
        Dw = dict(D); Dw['yaw_deg'] = D['yaw_deg']
        fig_t, fig_D = t, D
        # wrapped heading is easier to read here than a continuous one (many turns)
        fig, axs = plt.subplots(4, 1, figsize=(10.5, 9.4), sharex=True)
        yy = D['yaw_deg'].astype(float).copy(); yy[np.r_[False, np.abs(np.diff(yy)) > 180]] = np.nan
        rr = D['roll_deg'].astype(float).copy(); rr[np.r_[False, np.abs(np.diff(rr)) > 180]] = np.nan
        for ax, y, lab in ((axs[0], yy, 'heading (deg)'), (axs[1], D['pitch_deg'], 'pitch (deg)'), (axs[2], rr, 'roll (deg)')):
            ax.plot(t, y, color='tab:blue', lw=1.1); ax.set_ylabel(lab); ax.grid(alpha=0.3)
        axs[0].axhline(D['yaw_deg'][0], color='tab:red', ls='--', lw=0.9, label='start heading'); axs[0].legend(fontsize=8)
        extra(axs[3], np.ones(len(t), bool)); axs[3].grid(alpha=0.3); axs[3].set_xlabel('time (s)')
        fig.suptitle('Test 6: erratic movement, board returned to its start pose at the end', fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, 't6_erratic.png'), dpi=160); plt.close(fig)

# ============================================================== continuous session (tests 3-6): rest drift
sess = []
for lab, name in (('yaw', 't3_yaw.txt'), ('pitch', 't4_pitch.txt'), ('roll', 't5_roll.txt'), ('erratic1', 't6_erratic1.txt')):
    t, D, _, _ = load(name); yaw = un(D['yaw_deg']); a, b = longest(rest_mask(t, D), t)
    sess.append(dict(test=lab, rest_s=float(t[b] - t[a]), rate=float(np.polyfit(t[a:b], yaw[a:b], 1)[0]), raw_z=float(D['gyrZ_dps'][a:b].mean()),
                     bias_z=float(D['gyroBiasZ_dps'][a:b].mean()),
                     tilt_roll=float(np.mean(wrap(D['roll_deg'] - D['accRoll_deg'])[a:b])), tilt_pitch=float(np.mean((D['pitch_deg'] - D['accPitch_deg'])[a:b]))))
R['session'] = sess

json.dump(R, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
print(json.dumps(R, indent=1)[:7000])
