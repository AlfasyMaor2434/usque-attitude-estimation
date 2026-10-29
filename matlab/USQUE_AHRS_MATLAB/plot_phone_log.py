"""
plot_phone_log.py <usque_phone_log_*.txt> [out_dir] [t_end_s]

t_end_s (optional) crops every graph at that time, e.g. to leave out a free-handling phase.

Graphs from a live_phone_usque log (5 Hz print lines):
  1. orientation over time, filter (from the logged quaternion) vs the phone app's own orientation
  2. estimated gyro z bias over time
  3. magnetometer health: |mag|, inclination, gate

Euler angles are computed here from the quaternion (ZYX: yaw, pitch, roll). They are fine for
plotting except near pitch = +-90 deg, where yaw and roll are not separately defined (shaded).
The fourth panel of figure 1 is the total angle between the filter's and the app's orientation,
computed from the rotation matrices -- one number that has no such singularity.

App convention (Android, as logged): azimuth clockwise from magnetic north, pitch positive
top-edge DOWN, roll same sign as the filter's -> app as FRD/NED Euler = (yaw=az, pitch=-p, roll=r).
"""
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

log = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(log))
os.makedirs(out, exist_ok=True)
tag = os.path.splitext(os.path.basename(log))[0].replace('usque_phone_log_', '')
T_END = float(sys.argv[3]) if len(sys.argv) > 3 else None
# The app reports azimuth from MAGNETIC north, the filter yaw from TRUE north. Adding the site
# declination to the app azimuth puts both on true north, so the comparison is absolute (no
# offset fitted to the data).
DECLINATION_DEG = 5.006

pat = re.compile(r"sigRPY:\s*(\S+)\s+(\S+)\s+(\S+).*?bgZ:\s*(\S+) dps.*?magInfl:\s*(\S+)\s+gate:(\d).*?"
                 r"\|mag\|:\s*(\S+)\s+ref:\s*(\S+)\s+dev:\s*(\S+)%\s+incl:\s*(\S+).*?"
                 r"app az:\s*(\S+) p:\s*(\S+) r:\s*(\S+)\s+\|\s+q:\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")
text = open(log, encoding='utf-8', errors='ignore').read()
A = np.array([[float(x) for x in m.groups()] for m in pat.finditer(text)])
PRINT_HZ = 5.0
t = np.arange(len(A)) / PRINT_HZ
if T_END is not None:
    A, t = A[t <= T_END], t[t <= T_END]
    tag += f'_0-{T_END:.0f}s'
sr, sp, sy, bgz, mi, gate, mag, ref, dev, incl, az, ap, ar = A.T[:13]
az = az + DECLINATION_DEG                                        # app azimuth, true north
q = A[:, 13:17]                                                  # w x y z, body->nav
m_seed = re.search(r"gyro\s+FRD avg \(bias seed\):\s*(\S+)\s+(\S+)\s+(\S+)", text)
seed_z = np.degrees(float(m_seed.group(3))) if m_seed else np.nan
m_inc = re.search(r"mag inclination: WMM (\S+) deg, measured at boot (\S+) deg", text)

Rf = Rot.from_quat(np.column_stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]]))
ypr = Rf.as_euler('ZYX', degrees=True)                           # yaw, pitch, roll
ok = ~np.isnan(az)
Ra = Rot.from_euler('ZYX', np.column_stack([az, -ap, ar])[ok], degrees=True)
ang = np.full(len(t), np.nan)
ang[ok] = np.degrees((Rf[ok].inv() * Ra).magnitude())
gl = np.abs(ypr[:, 1]) > 75
dyaw = ((ypr[:, 0] - az + 180) % 360) - 180


def continuous(angle, mask_bad):
    """Angle without the +-180 wrap: each sample is placed on the 360-deg branch closest to the
    previous valid one (so a turn past 180 keeps climbing instead of jumping to -180). Samples
    in mask_bad (near pitch +-90, where yaw/roll are undefined) are left out and bridged."""
    out = np.full(len(angle), np.nan)
    last = None
    for i, a in enumerate(angle):
        if mask_bad[i] or np.isnan(a):
            continue
        out[i] = a if last is None else last + ((a - last + 180) % 360 - 180)
        last = out[i]
    return out


yaw_c = continuous(ypr[:, 0], gl)
roll_c = continuous(ypr[:, 2], gl)
# app curves on the same branch as the filter's (difference wrapped to +-180)
app_yaw_c = yaw_c + ((az - ypr[:, 0] + 180) % 360 - 180)
app_roll_c = roll_c + ((ar - ypr[:, 2] + 180) % 360 - 180)


def shade(ax, mask, color, label=None):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color=color, alpha=0.18, lw=0, label=label)
        label = None


def wrap_break(y):
    y = y.astype(float).copy()
    y[np.r_[False, np.abs(np.diff(y)) > 180]] = np.nan           # no vertical line at +-180 wraps
    return y


# ---------------------------------------------------------------- 1) orientation
fig, axs = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
for ax, est, app, lab in ((axs[0], yaw_c, app_yaw_c, 'heading (deg, continuous)'),
                          (axs[1], ypr[:, 1], -ap, 'pitch (deg)'),
                          (axs[2], roll_c, app_roll_c, 'roll (deg, continuous)')):
    ax.plot(t, est, color='tab:blue', lw=1.4, label='USQUE estimate')
    ax.plot(t, app, color='tab:orange', lw=1.1, ls='--', label='phone app (MATLAB Mobile) orientation')
    shade(ax, gl, 'grey', '|pitch| > 75 deg: heading and roll not defined')
    ax.set_ylabel(lab); ax.grid(alpha=0.3)
axs[0].legend(fontsize=8, ncol=3, loc='best')
axs[3].plot(t, ang, color='tab:red', lw=1.0, label='total angle between the two orientations (absolute)')
shade(axs[3], gate > 0, 'tab:purple', 'magnetometer rejected (gate)')
axs[3].set_ylabel('difference (deg)'); axs[3].set_xlabel('time (s)'); axs[3].grid(alpha=0.3)
axs[3].set_ylim(0, min(60, np.nanpercentile(ang, 99.5) * 1.2)); axs[3].legend(fontsize=8, ncol=2, loc='upper left')
fig.suptitle(f'Phone test {tag}: USQUE estimate vs phone app orientation (both from true north)')
fig.tight_layout(); fig.savefig(os.path.join(out, f'phone_{tag}_1_orientation.png'), dpi=160); plt.close(fig)

# ---------------------------------------------------------------- 2) bias
fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axs[0].plot(t, bgz, color='tab:blue', lw=1.3, label='estimated gyro bias z')
if np.isfinite(seed_z):
    axs[0].axhline(seed_z, color='tab:red', ls=':', label=f'boot seed {seed_z:+.3f} deg/s')
axs[0].axhline(0, color='k', lw=0.6)
shade(axs[0], gate > 0, 'tab:purple', 'magnetometer rejected (gate)')
axs[0].set_ylabel('bgZ (deg/s)'); axs[0].grid(alpha=0.3); axs[0].legend(fontsize=8)
axs[1].plot(t, wrap_break(dyaw), color='tab:green', lw=1.0, label='filter yaw - app heading (absolute)')
axs[1].plot(t, sy, color='grey', lw=0.8, label='filter yaw 1-sigma'); axs[1].plot(t, -sy, color='grey', lw=0.8)
shade(axs[1], gl, 'grey', '|pitch| > 75 deg')
axs[1].set_ylim(-30, 30); axs[1].set_ylabel('deg'); axs[1].set_xlabel('time (s)'); axs[1].grid(alpha=0.3)
axs[1].legend(fontsize=8, ncol=3)
fig.suptitle(f'Phone test {tag}: gyro bias estimate and heading difference')
fig.tight_layout(); fig.savefig(os.path.join(out, f'phone_{tag}_2_bias.png'), dpi=140); plt.close(fig)

# ---------------------------------------------------------------- 3) magnetometer health
fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axs[0].plot(t, mag, color='tab:blue', lw=1.0, label='|mag| (calibrated)')
axs[0].axhline(ref[0], color='k', ls=':', label=f'boot reference {ref[0]:.1f} uT')
axs[0].fill_between(t, ref[0] * 0.85, ref[0] * 1.15, color='k', alpha=0.06, label='gate limit +-15%')
axs[0].set_ylabel('uT'); axs[0].grid(alpha=0.3); axs[0].legend(fontsize=8, ncol=3)
axs[1].plot(t, incl, color='tab:blue', lw=1.0, label='measured inclination')
if m_inc:
    wmm, boot = float(m_inc.group(1)), float(m_inc.group(2))
    axs[1].axhline(wmm, color='tab:green', ls='--', label=f'WMM (expected) {wmm:.1f} deg')
    axs[1].axhline(boot, color='k', ls=':', label=f'boot reference {boot:.1f} deg')
    axs[1].fill_between(t, boot - 10, boot + 10, color='k', alpha=0.06, label='gate limit +-10 deg')
shade(axs[1], gate > 0, 'tab:purple', 'magnetometer rejected (gate)')
axs[1].set_ylabel('deg (positive = field points down)'); axs[1].set_xlabel('time (s)'); axs[1].grid(alpha=0.3)
axs[1].legend(fontsize=8, ncol=3)
fig.suptitle(f'Phone test {tag}: magnetometer field strength and inclination')
fig.tight_layout(); fig.savefig(os.path.join(out, f'phone_{tag}_3_mag.png'), dpi=140); plt.close(fig)

calm = ~gl & (np.r_[0, np.abs(np.diff(ang))] < 2.0)
print(f"{len(t)} lines, {t[-1]:.0f} s | gate active {np.mean(gate > 0) * 100:.0f}% | absolute total-angle difference "
      f"(outside the pitch band): median {np.nanmedian(ang[~gl]):.1f}, p90 {np.nanpercentile(ang[~gl], 90):.1f} deg, "
      f"above 10 deg {np.nanmean(ang[~gl] > 10) * 100:.0f}% of the time | bgZ start {bgz[0]:+.3f} end {bgz[-1]:+.3f}")
print('saved to', out)
