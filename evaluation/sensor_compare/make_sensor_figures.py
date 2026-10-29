"""Figures for run_sensor_compare.py (reads sensor_compare.csv + sensor_compare_arrays.npz)."""
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)
meta = json.load(open(os.path.join(HERE, 'sensor_compare_meta.json')))
FLIGHTS, VARIANTS = meta['flights'], meta['variants']
A = np.load(os.path.join(HERE, 'sensor_compare_arrays.npz'))
rows = list(csv.DictReader(open(os.path.join(HERE, 'sensor_compare.csv'))))
POOL = {r['variant']: r for r in rows if r['name'] == 'pooled'}

LABEL = {'px4': 'PX4 IMU', 'px4_mag': 'PX4 IMU + PX4 mag', 'lsm': 'LSM IMU (raw)',
         'lsm_mag': 'LSM IMU + LSM mag (raw)', 'lsm_sc': 'LSM IMU, gyro scale-corrected',
         'lsm_sc_mag': 'LSM IMU scale-corr. + LSM mag'}
COLOR = {'px4': 'tab:blue', 'px4_mag': 'tab:cyan', 'lsm': 'tab:red', 'lsm_mag': 'tab:orange',
         'lsm_sc': 'tab:purple', 'lsm_sc_mag': 'tab:pink'}


def arr(v, n, k):
    return A[f"{v}__{n}__{k}"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)


# 1) pooled error CDF
fig, ax = plt.subplots(figsize=(8, 5))
for v in VARIANTS:
    e = np.sort(np.concatenate([arr(v, n, 'err') for n in FLIGHTS]))
    ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=COLOR[v], label=LABEL[v])
for x in (1, 3):
    ax.axvline(x, color='k', ls=':', lw=1)
ax.set_xscale('log'); ax.set_xlim(0.05, 90)
ax.set_xlabel('total attitude error vs mocap (deg, log)'); ax.set_ylabel('fraction of samples')
ax.set_title('Attitude error CDF, indoor_1-3 pooled: PX4 vs LSM9DS1 sensors')
ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=8, loc='lower right')
save(fig, 'sc_01_error_cdf.png')

# 2) error vs time per flight (no-mag variants, then mag variants)
for tag, vs in (('nomag', ['px4', 'lsm', 'lsm_sc']), ('mag', ['px4_mag', 'lsm_mag', 'lsm_sc_mag'])):
    fig, axs = plt.subplots(len(FLIGHTS), 1, figsize=(10, 8), sharex=False)
    for ax, n in zip(axs, FLIGHTS):
        for v in vs:
            ax.plot(arr(v, n, 't_cmp'), arr(v, n, 'err'), color=COLOR[v], lw=0.8, label=LABEL[v])
        ax.set_yscale('log'); ax.set_ylim(0.05, 180); ax.grid(True, which='both', alpha=0.3)
        ax.axhline(1, color='k', ls=':', lw=0.8); ax.axhline(3, color='k', ls=':', lw=0.8)
        ax.set_ylabel(f'{n}\nerror (deg)')
    axs[0].legend(fontsize=8, loc='upper left', ncol=3)
    axs[-1].set_xlabel('time since start (s)')
    fig.suptitle(f"Attitude error vs time ({'no magnetometer' if tag == 'nomag' else 'with magnetometer'})")
    save(fig, f'sc_02_error_time_{tag}.png')

# 3) pooled metric bars
metrics = [('err_lt1', '< 1 deg (%)', 100), ('err_lt3', '< 3 deg (%)', 100)]
fig, axs = plt.subplots(1, 3, figsize=(13, 4.5))
x = np.arange(len(VARIANTS))
for ax, (k, lab, s) in zip(axs[:2], metrics):
    vals = [float(POOL[v][k]) * s for v in VARIANTS]
    ax.bar(x, vals, color=[COLOR[v] for v in VARIANTS])
    for xi, val in zip(x, vals):
        ax.text(xi, val + 1, f'{val:.0f}', ha='center', fontsize=8)
    ax.set_ylim(0, 110); ax.set_title(lab)
ax = axs[2]
w = 0.27
for j, (k, lab) in enumerate((('rmse_roll', 'roll'), ('rmse_pitch', 'pitch'), ('rmse_yaw', 'yaw'))):
    ax.bar(x + (j - 1) * w, [float(POOL[v][k]) for v in VARIANTS], w, label=lab)
ax.set_yscale('log'); ax.set_title('RMSE per axis (deg, log)'); ax.legend(fontsize=8)
for ax in axs:
    ax.set_xticks(x); ax.set_xticklabels([LABEL[v] for v in VARIANTS], rotation=35, ha='right', fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
fig.suptitle('indoor_1-3 pooled, same USQUE final config')
save(fig, 'sc_03_metric_bars.png')

# 4) gyro bias estimate (z) per flight
fig, axs = plt.subplots(1, len(FLIGHTS), figsize=(13, 4))
for ax, n in zip(axs, FLIGHTS):
    for v in ('px4', 'lsm', 'lsm_sc'):
        ax.plot(arr(v, n, 't_bg'), arr(v, n, 'bg')[:, 2], color=COLOR[v], label=LABEL[v])
    ax.set_title(n); ax.set_xlabel('t (s)'); ax.grid(True, alpha=0.3)
axs[0].set_ylabel('estimated gyro bias z (deg/s)'); axs[0].legend(fontsize=8)
save(fig, 'sc_04_gyro_bias_z.png')
print('figures ->', FIG)
