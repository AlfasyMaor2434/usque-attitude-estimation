"""Diagnose yaw-vs-RTK error on mars 1-11 (excl. 4) with the final config and a few one-knob variants."""
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

FLIGHTS = [f'mars_{i}' for i in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11)]
VARIANTS = {
    'final': {},
    'no_mag': {'_no_mag': True},
    'psi_off': {'use_mag_bias_state': False},
    'no_pf_calib': {'per_flight_mag_calib': False},
    'mag_r_0.5': {'mag_noise_var': 0.5},
}


def task(args):
    name, var = args
    import usque_eval as ue
    ov = dict(VARIANTS[var])
    no_mag = ov.pop('_no_mag', False)
    ue.dl.SITE_PROFILE['mars']['mag'] = not no_mag
    try:
        o = ue.run_one(name, 'mars', ov, keep_arrays=True)
    finally:
        ue.dl.SITE_PROFILE['mars']['mag'] = True
    a = o['arrays']
    t, e = np.asarray(a['t_err']), np.asarray(a['err_signed'])
    slope = np.polyfit(t, e, 1)[0] * 60.0          # deg/min linear drift of the signed error
    return dict(variant=var, name=name, lt1=o['err_lt1'], lt3=o['err_lt3'], p50=o['err_p50'],
                rmse=o['err_rmse'], mean_signed=float(np.mean(e)), drift_deg_per_min=float(slope),
                std_detr=float(np.std(e - np.polyval(np.polyfit(t, e, 1), t))),
                dur_s=float(t[-1] - t[0]), anees=o['anees'], anis=o['anis'], psi=o.get('psi_mag_deg'),
                sig_ref=o.get('sigma_ref_deg'))


if __name__ == '__main__':
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    jobs = [(n, v) for v in VARIANTS for n in FLIGHTS]
    with Pool(n_proc) as p:
        res = p.map(task, jobs, chunksize=1)
    out = os.path.join(HERE, 'diag_mars.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader(); w.writerows(res)
    for v in VARIANTS:
        rs = [r for r in res if r['variant'] == v]
        print(f"\n== {v}: mean <1 {np.mean([r['lt1'] for r in rs]):.2f}  <3 {np.mean([r['lt3'] for r in rs]):.2f}")
        for r in rs:
            print(f"  {r['name']:8s} <1 {r['lt1']:.2f} <3 {r['lt3']:.2f} p50 {r['p50']:5.2f} rmse {r['rmse']:6.2f} "
                  f"mean {r['mean_signed']:+6.2f} drift {r['drift_deg_per_min']:+6.2f} deg/min std_detr {r['std_detr']:5.2f} "
                  f"dur {r['dur_s']:4.0f}s psi {r['psi'] if r['psi'] is None else round(r['psi'], 1)} sig_ref {r['sig_ref']:.2f}")
