"""
run_lpf_study.py -- does the accelerometer low-pass filter help, and does a REAL-TIME (causal) filter
give the same result as the offline zero-phase one the report uses?

Indoor flights 1-3 (motion-capture ground truth, accelerometer-only updates), final USQUE config,
only the accelerometer pre-filter and (where noted) the accelerometer noise variance change.
Writes lpf_study.csv (one row per config and flight, plus pooled rows) and lpf_study.png.
"""
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FLIGHTS = ['indoor_1', 'indoor_2', 'indoor_3']

Z, CA, BX, NO = (dict(accel_lpf_kind='zero_phase'), dict(accel_lpf_kind='causal'),
                 dict(accel_lpf_kind='boxcar'), dict(lpf_mode='none'))
CONFIGS = {
    # name: (group, overrides)
    'zero-phase 15 Hz (report)':     ('filter type', dict(Z)),
    'causal 15 Hz (real time)':      ('filter type', dict(CA)),
    'average between updates':       ('filter type', dict(BX)),
    'no filter':                     ('filter type', dict(NO)),
    'no filter, accel noise 1e-2':   ('no filter, re-tuned', dict(NO, accel_noise_var=1e-2)),
    'no filter, accel noise 5e-2':   ('no filter, re-tuned', dict(NO, accel_noise_var=5e-2)),
    'no filter, accel noise 2e-1':   ('no filter, re-tuned', dict(NO, accel_noise_var=2e-1)),
    'average, accel noise 1e-2':     ('no filter, re-tuned', dict(BX, accel_noise_var=1e-2)),
    'causal 15 Hz, accel noise 1e-2': ('no filter, re-tuned', dict(CA, accel_noise_var=1e-2)),
    'zero-phase 5 Hz':               ('cutoff', dict(Z, accel_cutoff_hz=5.0)),
    'zero-phase 10 Hz':              ('cutoff', dict(Z, accel_cutoff_hz=10.0)),
    'zero-phase 30 Hz':              ('cutoff', dict(Z, accel_cutoff_hz=30.0)),
    'causal 5 Hz':                   ('cutoff', dict(CA, accel_cutoff_hz=5.0)),
    'causal 10 Hz':                  ('cutoff', dict(CA, accel_cutoff_hz=10.0)),
    'causal 30 Hz':                  ('cutoff', dict(CA, accel_cutoff_hz=30.0)),
}
FIELDS = ['config', 'group', 'name', 'lt1', 'lt3', 'p50', 'p90', 'rmse_roll', 'rmse_pitch', 'rmse_yaw', 'anees', 'anis']


def task(args):
    cfg, name = args
    import usque_eval as ue
    o = ue.run_one(name, 'indoor', CONFIGS[cfg][1], keep_arrays=True)
    a = o['arrays']
    return dict(config=cfg, group=CONFIGS[cfg][0], name=name, lt1=o['err_lt1'], lt3=o['err_lt3'], p50=o['err_p50'],
                p90=o['err_p90'], rmse_roll=o['att_rmse_rpy'][0], rmse_pitch=o['att_rmse_rpy'][1],
                rmse_yaw=o['att_rmse_rpy'][2], anees=o['anees'], anis=o['anis']), \
        (np.asarray(a['err']), np.asarray(a['axis_err']), np.asarray(a['nees']), np.asarray(a['nis']))


if __name__ == '__main__':
    t0 = time.perf_counter()
    jobs = sorted(((c, n) for c in CONFIGS for n in FLIGHTS), key=lambda j: j[1])   # longest flight first
    with Pool(11) as p:
        res = p.map(task, jobs, chunksize=1)
    rows = [r for r, _ in res]
    pooled = []
    for c in CONFIGS:
        rr = [(r, a) for r, a in res if r['config'] == c]
        e = np.concatenate([a[0] for _, a in rr]); ax = np.concatenate([a[1] for _, a in rr])
        pooled.append(dict(config=c, group=CONFIGS[c][0], name='pooled', lt1=np.mean(e < 1), lt3=np.mean(e < 3),
                           p50=np.median(e), p90=np.percentile(e, 90),
                           rmse_roll=np.sqrt(np.nanmean(ax[:, 0] ** 2)), rmse_pitch=np.sqrt(np.nanmean(ax[:, 1] ** 2)),
                           rmse_yaw=np.sqrt(np.nanmean(ax[:, 2] ** 2)),
                           anees=np.mean(np.concatenate([a[2] for _, a in rr])), anis=np.mean(np.concatenate([a[3] for _, a in rr]))))
    with open(os.path.join(HERE, 'lpf_study.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows); w.writerows(pooled)
    print(f"{len(jobs)} runs in {time.perf_counter() - t0:.0f} s\n")
    print(f"{'config':34s} {'<1 deg':>7s} {'<3 deg':>7s} {'P90':>6s} {'RMSE r/p/y':>18s} {'ANEES':>6s} {'ANIS':>6s}   (pooled; ideal ANEES 3, ANIS 2)")
    for r in pooled:
        print(f"{r['config']:34s} {100 * r['lt1']:6.1f}% {100 * r['lt3']:6.1f}% {r['p90']:6.2f} "
              f"{r['rmse_roll']:5.2f}/{r['rmse_pitch']:5.2f}/{r['rmse_yaw']:5.2f} {r['anees']:6.2f} {r['anis']:6.2f}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.5))
    names = [r['config'] for r in pooled]; y = np.arange(len(names))[::-1]
    col = {'filter type': 'tab:blue', 'no filter, re-tuned': 'tab:orange', 'cutoff': 'tab:green'}
    cs = [col[r['group']] for r in pooled]
    for ax, key, lab, ideal, sc in ((axs[0], 'lt1', 'estimates under 1° (%)', None, 100), (axs[1], 'anees', 'ANEES (ideal 3)', 3, 1),
                                    (axs[2], 'anis', 'ANIS (ideal 2)', 2, 1)):
        ax.barh(y, [sc * r[key] for r in pooled], color=cs)
        if ideal:
            ax.axvline(ideal, color='k', ls='--', lw=1)
            ax.set_xscale('log')
        ax.set_yticks(y); ax.set_yticklabels(names if ax is axs[0] else [], fontsize=8); ax.set_xlabel(lab); ax.grid(alpha=0.3, axis='x')
    fig.suptitle('Accelerometer pre-filter study, indoor flights 1-3 pooled')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'lpf_study.png'), dpi=150)
