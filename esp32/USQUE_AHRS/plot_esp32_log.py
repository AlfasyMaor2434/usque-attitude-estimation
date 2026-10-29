"""
plot_esp32_log.py <usque_ahrs_log_*.txt> [out_dir] [t_end_s]

Graphs from a 3D-viewer log of USQUE_AHRS.ino: orientation (continuous angles, the firmware's
gimbal-safe roll/pitch/yaw), the filter's own 1-sigma, gyro-bias estimates, with the viewer's MARK
notes as labelled vertical lines. Prints a per-segment summary (segments = between marks).
"""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'esp32_tests')
T_END = float(sys.argv[3]) if len(sys.argv) > 3 else None
os.makedirs(out, exist_ok=True)
tag = re.sub(r'^usque_ahrs_log_|\.txt$', '', os.path.basename(log))[:19]

keys = ['roll_deg', 'pitch_deg', 'yaw_deg', 'sigRoll_deg', 'sigPitch_deg', 'sigYaw_deg',
        'gyroBiasX_dps', 'gyroBiasY_dps', 'gyroBiasZ_dps', 'accelInfl']
opt = ['yawHeld', 'magOn']
# newer firmware: raw gyro (mean per print interval), bias 1-sigma, accelerometer tilt
ext = ['gyrX_dps', 'gyrY_dps', 'gyrZ_dps', 'sigBiasX_dps', 'sigBiasY_dps', 'sigBiasZ_dps', 'biasSeeded',
       'accRoll_deg', 'accPitch_deg', 'accLpf', 'imuHz', 'badSmp', 'rdErr', 'cfgFix', 'accFault']
rows, marks, still_ref = [], [], None
for ln in open(log, encoding='utf-8', errors='ignore'):
    m = re.search(r"\+([\d.]+)s\] (RAW|MARK)\s+(.*)", ln)
    if not m:
        continue
    if m.group(2) == 'MARK':
        marks.append((float(m.group(1)), m.group(3).strip())); continue
    if m.group(3).startswith('still_gyro_dps:'):
        v = m.group(3).split(':', 1)[1].split(',')
        still_ref = [float(v[0]), float(v[1]), float(v[2])]
        continue
    if not m.group(3).startswith('roll_deg:'):
        continue
    try:
        d = dict(kv.split(':') for kv in m.group(3).strip().split(','))
        rows.append([float(m.group(1))] + [float(d[k]) for k in keys] + [float(d.get(k, 0)) for k in opt]
                    + [float(d.get(k, 'nan')) for k in ext])
    except Exception:
        pass
A = np.array(rows); t0 = A[0, 0]; t = A[:, 0] - t0
marks = [(mt - t0, s) for mt, s in marks if mt >= t0 and not s.startswith(('===', '---'))]
if T_END:
    A, t = A[t <= T_END], t[t <= T_END]; marks = [m for m in marks if m[0] <= T_END]
roll, pitch, yaw, sr, sp, sy, bx, by, bz, ai, held, magon = A[:, 1:13].T
gyr, sigb, seeded, acc_roll, acc_pitch = A[:, 13:16], A[:, 16:19], A[:, 19], A[:, 20], A[:, 21]
acc_lpf = A[:, 22]
# accelerometer low-pass state of this run: 'on', 'off', or '' for older logs without the field
LPF_TXT = '' if not np.isfinite(acc_lpf).any() else ('accel filter ON' if np.nanmax(acc_lpf) > 0 else 'accel filter OFF')
have_ext = np.isfinite(gyr).all()
unw = lambda a: np.degrees(np.unwrap(np.radians(a)))
rollc, yawc = unw(roll), unw(yaw)

fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
for ax, y, s, lab in ((axs[0], yawc, sy, 'heading (deg, continuous)'), (axs[1], pitch, sp, 'pitch (deg)'),
                      (axs[2], rollc, sr, 'roll (deg, continuous)')):
    ax.fill_between(t, y - 3 * s, y + 3 * s, color='tab:blue', alpha=0.15, lw=0, label="filter's own 3-sigma")
    ax.plot(t, y, color='tab:blue', lw=1.3, label='USQUE estimate (ESP32, ICM-20948)')
    ax.set_ylabel(lab); ax.grid(alpha=0.3)
e = np.flatnonzero(np.diff(np.r_[0, (held > 0).astype(int), 0]))
for ax in axs[:3]:
    lab = 'near pitch +-90: heading held, roll carries the rotation'
    for a, b in zip(e[::2], e[1::2]):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color='grey', alpha=0.2, lw=0, label=lab); lab = None
axs[0].legend(fontsize=8, ncol=3, loc='best')
axs[3].plot(t, bx, label='x'); axs[3].plot(t, by, label='y'); axs[3].plot(t, bz, label='z')
axs[3].set_ylabel('gyro bias estimate (deg/s)'); axs[3].set_xlabel('time (s)'); axs[3].grid(alpha=0.3)
axs[3].legend(fontsize=8, ncol=3)
for i, (mt, s) in enumerate(marks):
    for ax in axs:
        ax.axvline(mt, color='k', ls=':', lw=0.8)
    axs[0].annotate(str(i + 1), (mt, 1.0), xycoords=('data', 'axes fraction'), xytext=(2, 3),
                    textcoords='offset points', fontsize=9)
note = chr(10).join(__import__('textwrap').wrap('   '.join(f'{i + 1}: {s}' for i, (_, s) in enumerate(marks)), 150))
fig.suptitle(f'ESP32 test {tag}: USQUE estimate' + ('' if magon.any() else ' (gyro + accelerometer, heading relative to start)')
             + (f' -- {LPF_TXT}' if LPF_TXT else '')
             + ('\n' + note if note else ''), fontsize=10)
fig.tight_layout()
name = f'esp32_{tag}' + (f'_0-{T_END:.0f}s' if T_END else '') + '_orientation.png'
fig.savefig(os.path.join(out, name), dpi=160); plt.close(fig)

# ---------------------------------------------------------------- gyro bias figures (newer firmware)
if have_ext:
    bias = np.column_stack([bx, by, bz])
    corr = gyr - bias                                  # what the filter integrates
    unseeded = not seeded.any()
    fig, axs = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for i, axn in enumerate('xyz'):
        ax = axs[i, 0]
        ax.fill_between(t, bias[:, i] - 3 * sigb[:, i], bias[:, i] + 3 * sigb[:, i], color='tab:blue', alpha=0.15, lw=0,
                        label="filter's own 3-sigma")
        ax.plot(t, bias[:, i], color='tab:blue', lw=1.4, label='bias estimate')
        if still_ref:
            ax.axhline(still_ref[i], color='tab:red', ls='--', lw=1.0, label='measured in the boot still window')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel(f'gyro bias {axn} (deg/s)'); ax.grid(alpha=0.3)
        ax = axs[i, 1]
        ax.plot(t, gyr[:, i], color='tab:orange', lw=0.8, label='raw gyro')
        ax.plot(t, corr[:, i], color='tab:green', lw=0.8, label='raw gyro - bias estimate')
        ax.axhline(0, color='k', lw=0.5)
        lim = max(1.5, 1.5 * np.nanmax(np.abs(bias[:, i])) + 0.5)
        ax.set_ylim(-lim, lim); ax.set_ylabel(f'rate {axn} (deg/s)'); ax.grid(alpha=0.3)
        for mt, _ in marks:
            axs[i, 0].axvline(mt, color='k', ls=':', lw=0.8); axs[i, 1].axvline(mt, color='k', ls=':', lw=0.8)
    axs[0, 0].legend(fontsize=8); axs[0, 1].legend(fontsize=8)
    axs[0, 0].set_title('bias estimate' + (' (started from ZERO, no boot seed)' if unseeded else ' (seeded from the boot still window)'))
    axs[0, 1].set_title('gyro rate before and after removing the bias (zoomed around zero)')
    axs[2, 0].set_xlabel('time (s)'); axs[2, 1].set_xlabel('time (s)')
    fig.suptitle(f'ESP32 test {tag}: gyro bias estimation')
    fig.tight_layout()
    bname = f'esp32_{tag}' + (f'_0-{T_END:.0f}s' if T_END else '') + '_gyro_bias.png'
    fig.savefig(os.path.join(out, bname), dpi=160); plt.close(fig)
    rest = (np.linalg.norm(corr, axis=1) < 1.0)
    print("gyro at rest (|corrected rate| < 1 deg/s): raw mean", np.round(np.nanmean(gyr[rest], 0), 3),
          "-> corrected mean", np.round(np.nanmean(corr[rest], 0), 3), "deg/s;",
          "final bias", np.round(bias[-1], 3), "+-", np.round(sigb[-1], 3),
          "| still-window value", still_ref, "| seeded" if not unseeded else "| NOT seeded")
    print('saved', os.path.join(out, bname))

print(f"{len(t)} lines, {t[-1]:.0f} s, magnetometer {'ON' if magon.any() else 'OFF'}" + (f", {LPF_TXT}" if LPF_TXT else ''))
if np.isfinite(A[:, 23]).any():
    # data-quality counters of the firmware (cumulative, except imuHz and accFault)
    print(f"  data quality: IMU rate {np.nanmedian(A[:, 23]):.0f} Hz (min {np.nanmin(A[5:, 23]):.0f}), rejected samples {A[-1, 24]:.0f}, "
          f"bus read errors {A[-1, 25]:.0f}, sensor re-configurations {A[-1, 26]:.0f}, "
          f"accelerometer ignored {100 * np.nanmean(A[:, 27]):.1f}% of the time")
    for ln_ in open(log, encoding='utf-8', errors='ignore'):
        if 'IMU CONFIG CHANGED' in ln_ or 'ACCELEROMETER FAULT' in ln_:
            print('   ', ln_.strip()[:200])
if np.isfinite(acc_pitch).any():
    # filter attitude minus the tilt the accelerometer alone implies, while the board is at rest
    rest = np.abs(np.gradient(pitch, t)) + np.abs(np.gradient(np.degrees(np.unwrap(np.radians(roll))), t)) < 1.0
    dr = (roll - acc_roll + 180) % 360 - 180
    print(f"  at rest: filter roll - accelerometer roll: mean {np.nanmean(dr[rest]):+.2f} deg (std {np.nanstd(dr[rest]):.2f}); "
          f"filter pitch - accelerometer pitch: mean {np.nanmean((pitch - acc_pitch)[rest]):+.2f} deg (std {np.nanstd((pitch - acc_pitch)[rest]):.2f})")
bounds = [0.0] + [m[0] for m in marks] + [t[-1]]
labels = ['(start)'] + [m[1] for m in marks]
for a, b, lab in zip(bounds[:-1], bounds[1:], labels):
    m = (t >= a) & (t < b)
    if m.sum() < 5: continue
    print(f"  {a:6.1f}-{b:6.1f} s  {lab[:48]:48s} roll {roll[m][0]:7.1f}->{roll[m][-1]:7.1f}  pitch {pitch[m][0]:6.1f}->{pitch[m][-1]:6.1f}  "
          f"heading {yawc[m][0]:7.1f}->{yawc[m][-1]:7.1f}  bias z {bz[m][0]:+.2f}->{bz[m][-1]:+.2f}")
print('saved', os.path.join(out, name))
