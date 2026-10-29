"""
tune_mars.py -- parameter sweep for yaw vs dual-RTK heading on mars 1-11 (excl. 4).

  python tune_mars.py <stage>        runs the configs of STAGES[stage] on all 10 flights (Pool 11),
                                     appends one row per (config, flight) to mars_tuning_log.csv and
                                     prints a ranking by pooled fraction < 1 deg.

Config dicts are usque_eval overrides (filter params + fusion_rate_hz / per_flight_mag_calib /
rtk_seed_avg_s) plus '_no_mag': True to run without the magnetometer.
"""
import csv
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FLIGHTS = [f'mars_{i}' for i in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11)]
LOG = os.path.join(HERE, 'mars_tuning_log.csv')
FIELDS = ['stage', 'cfg', 'name', 'lt1', 'lt3', 'p50', 'p90', 'rmse', 'mean_signed', 'anees', 'anis',
          'in3sig', 'psi', 'n', 'overrides']


def _grid(base, **axes):
    keys = list(axes)
    out = {}
    for vals in itertools.product(*[axes[k] for k in keys]):
        ov = dict(base)
        ov.update(dict(zip(keys, vals)))
        out['|'.join(f'{k}={v}' for k, v in zip(keys, vals))] = ov
    return out


SEED = {'rtk_seed_avg_s': 10.0}
STAGES = {
    # M1: how much the mag is trusted vs how fast psiMag may move vs bias adaptivity
    'M1': dict(
        **{'final': {}, 'final+seed': dict(SEED), 'no_mag+seed': dict(SEED, _no_mag=True)},
        **_grid(SEED,
                psi=['60/30', '300/10', '1000/5', 'off'],
                mag_noise_var=[0.05, 0.5, 2.0],
                bias_rw_var=[1e-10, 1e-9]),
    ),
    # M2: the gyro z bias really moves 0.01-0.07 deg/s within a flight (bias_check.py) -> a bias random
    # walk of ~1e-8 rad^2/s^3; let the mag track it (psi slow or off, mag trusted or not)
    'M2': _grid(SEED,
                psi=['60/30', '300/10', 'off'],
                mag_noise_var=[0.05, 0.5],
                bias_rw_var=[1e-8, 3e-8, 1e-7]),
    # M3: everything that is NOT the magnetometer, one knob at a time around BASE3 (set from the
    # M1/M2 winner): gyro noise, bias prior and seed window, accel weight / PHYS gate, fusion rate,
    # which adaptive branches run, gyro low-pass
    'M3': None,   # built by build_m3()
    # M4: combine the M3 changes that helped
    'M4': {
        'P0b1e-8+accel0.01': dict(SEED, _p0_bias=1e-8, accel_noise_var=0.01),
        'P0b1e-8+yaw0.5': dict(SEED, _p0_bias=1e-8, _p0_yaw_deg=0.5),
        'P0b1e-8+accel0.01+yaw0.5': dict(SEED, _p0_bias=1e-8, accel_noise_var=0.01, _p0_yaw_deg=0.5),
        'P0b1e-8+no_mag': dict(SEED, _p0_bias=1e-8, _no_mag=True),
        'P0b1e-8+psi300/10+magR2': dict(SEED, _p0_bias=1e-8, psi='300/10', mag_noise_var=2.0),
    },
}

BASE3 = dict(SEED)   # filled in after M2


def build_m3():
    b = dict(BASE3)
    one = lambda **kw: dict(b, **kw)
    c = {'base': dict(b)}
    for v in (1e-6, 1e-5, 1e-4, 3e-4):
        c[f'gyro_noise_var={v}'] = one(gyro_noise_var=v)
    for v in (1e-8, 1e-7, 1e-5):
        c[f'P0_bias={v}'] = one(_p0_bias=v)
    for v in (1.0, 10.0, 30.0):
        c[f'bias_seed_window_s={v}'] = one(_bias_window_s=v)
    for v in (5e-4, 1e-2, 5e-2):
        c[f'accel_noise_var={v}'] = one(accel_noise_var=v)
    for v in (1.0, 5.0):
        c[f'phys_dir_thresh_deg={v}'] = one(phys_dir_thresh_deg=v)
    for v in (10.0, 100.0, 200.0):
        c[f'fusion_rate_hz={v}'] = one(fusion_rate_hz=v)
    c['adaptive=none'] = one(adaptive_methods=[])
    c['adaptive=no_IAE'] = one(adaptive_methods=['PHYSICAL', 'VIBRATION', 'MAG_GATE'])
    c['adaptive=no_VIB'] = one(adaptive_methods=['PHYSICAL', 'IAE', 'MAG_GATE'])
    c['lpf=both'] = one(_lpf_mode='both')
    c['lpf=none'] = one(_lpf_mode='none')
    c['P0_yaw_sigma=0.5deg'] = one(_p0_yaw_deg=0.5)
    return c


def expand(ov):
    ov = dict(ov)
    psi = ov.pop('psi', None)
    if psi == 'off':
        ov['use_mag_bias_state'] = False
    elif psi is not None:
        tau, sig = psi.split('/')
        ov['use_mag_bias_state'] = True
        ov['mag_bias_tau_s'] = float(tau)
        ov['mag_bias_sigma_ss_deg'] = float(sig)
    return ov


def task(args):
    stage, cfg, ov, name = args
    import usque_eval as ue
    ov = expand(ov)
    no_mag = ov.pop('_no_mag', False)
    mu = ue.mu
    keep = (mu.P0_BIAS_STATIC, mu.LPF_MODE, dict(mu.INIT_SIGMA_DEG['yaw']), ue.dl.load_static_gyro_bias)
    orig_bias = ue.dl.load_static_gyro_bias
    if '_p0_bias' in ov:
        mu.P0_BIAS_STATIC = float(ov['_p0_bias'])
    if '_lpf_mode' in ov:
        mu.LPF_MODE = ov['_lpf_mode']
    if '_p0_yaw_deg' in ov:
        mu.INIT_SIGMA_DEG['yaw']['rtk_yaw'] = float(ov['_p0_yaw_deg'])
    if '_bias_window_s' in ov:
        win = float(ov['_bias_window_s'])
        ue.dl.load_static_gyro_bias = lambda d, **k: orig_bias(d, max_window_s=win, **k)
    ov_run = {k: v for k, v in ov.items() if not k.startswith('_')}
    ue.dl.SITE_PROFILE['mars']['mag'] = not no_mag
    try:
        o = ue.run_one(name, 'mars', ov_run, keep_arrays=True)
    finally:
        ue.dl.SITE_PROFILE['mars']['mag'] = True
        mu.P0_BIAS_STATIC, mu.LPF_MODE = keep[0], keep[1]
        mu.INIT_SIGMA_DEG['yaw'].update(keep[2])
        ue.dl.load_static_gyro_bias = keep[3]
    e = np.asarray(o['arrays']['err'])
    return dict(stage=stage, cfg=cfg, name=name, lt1=o['err_lt1'], lt3=o['err_lt3'], p50=o['err_p50'],
                p90=o['err_p90'], rmse=o['err_rmse'], mean_signed=o['err_mean_signed'], anees=o['anees'],
                anis=o['anis'], in3sig=o['in3sig'], psi=o.get('psi_mag_deg'), n=len(e),
                overrides=json.dumps(ov if not no_mag else dict(ov, _no_mag=True), default=str)), e


def summarize(rows):
    by = {}
    for r in rows:
        by.setdefault(r['cfg'], []).append(r)
    out = []
    for cfg, rs in by.items():
        n = np.array([float(r['n']) for r in rs])
        lt1 = np.array([float(r['lt1']) for r in rs]); lt3 = np.array([float(r['lt3']) for r in rs])
        out.append(dict(cfg=cfg, pooled_lt1=float(np.sum(lt1 * n) / n.sum()), pooled_lt3=float(np.sum(lt3 * n) / n.sum()),
                        mean_lt1=float(lt1.mean()), min_lt1=float(lt1.min()),
                        anees=float(np.mean([float(r['anees']) for r in rs])),
                        anis=float(np.mean([float(r['anis']) for r in rs])), n_fl=len(rs)))
    return sorted(out, key=lambda d: -d['pooled_lt1'])


if __name__ == '__main__':
    stage = sys.argv[1]
    cfgs = build_m3() if stage == 'M3' else STAGES[stage]
    jobs = [(stage, c, ov, n) for c, ov in cfgs.items() for n in FLIGHTS]
    t0 = time.perf_counter()
    new = not os.path.exists(LOG)
    rows = []
    with Pool(11) as p, open(LOG, 'a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r, _ in p.imap_unordered(task, jobs, chunksize=1):
            w.writerow(r); fh.flush(); rows.append(r)
    print(f"stage {stage}: {len(jobs)} runs in {time.perf_counter() - t0:.0f} s")
    for s in summarize(rows):
        print(f"{s['pooled_lt1']:.3f} <1 pooled | {s['pooled_lt3']:.3f} <3 | mean-of-flights <1 {s['mean_lt1']:.3f} "
              f"(worst {s['min_lt1']:.2f}) | ANEES {s['anees']:.2f} ANIS {s['anis']:.2f} | {s['cfg']}")
