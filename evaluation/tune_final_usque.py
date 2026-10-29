"""
tune_final_usque.py -- parallel parameter sweep for the final USQUE results.

    python tune_final_usque.py <stage>      (stage names: see STAGES below)

Every (config, flight) pair is one task; results are appended to
tuning_log.csv as they finish, so a sweep can be inspected while running and
nothing is lost if it is interrupted. Selection happens afterwards
(select_config() / the final-run script), not here.
"""
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LOG = os.path.join(HERE, 'tuning_log.csv')

INDOOR = [('indoor_1', 'indoor'), ('indoor_2', 'indoor'), ('indoor_3', 'indoor')]
MARS = [('mars_5', 'mars'), ('mars_6', 'mars'), ('mars_7', 'mars')]


def grid(**axes):
    keys = list(axes)
    out = [{}]
    for k in keys:
        out = [dict(c, **{k: v}) for c in out for v in axes[k]]
    return out


STAGES = {
    # indoor: attitude/accel noise (mag unused indoors)
    'A': (INDOOR, grid(gyro_noise_var=[3e-5, 1e-5, 3e-6],
                       accel_noise_var=[1e-2, 3e-3, 1e-3, 3e-4])),
    # indoor refinement between the accuracy-safe 1e-2 and the NIS-friendly 3e-3
    'A2': (INDOOR, grid(gyro_noise_var=[1e-5, 3e-6],
                        accel_noise_var=[7e-3, 5e-3])),
    # mars: mag noise x psiMag on/off at the indoor-selected gyro noise, both accel candidates
    'B': (MARS, grid(gyro_noise_var=[1e-5, 3e-6],
                     accel_noise_var=[1e-2, 5e-3],
                     mag_noise_var=[0.05, 0.02, 0.01, 0.005],
                     use_mag_bias_state=[True, False])),
    # mars: psiMag dynamics (never swept before) x gyro noise, psiMag on, best mag/accel from B
    'C': (MARS, grid(accel_noise_var=[1e-2], mag_noise_var=[0.05], use_mag_bias_state=[True],
                     gyro_noise_var=[3e-6, 3e-5],
                     mag_bias_tau_s=[60.0, 120.0, 300.0],
                     mag_bias_sigma_ss_deg=[15.0, 30.0])),
    # update (fusion) rate: fewer, less-correlated accel/mag updates. Best settings so far.
    'D_in': (INDOOR, grid(gyro_noise_var=[3e-6], accel_noise_var=[1e-2],
                          fusion_rate_hz=[10.0, 15.0, 30.0])),
    'D_mars': (MARS, grid(gyro_noise_var=[3e-6], accel_noise_var=[1e-2], mag_noise_var=[0.05],
                          use_mag_bias_state=[True], fusion_rate_hz=[10.0, 15.0, 30.0])),
    # final mars check: intermediate gyro noise with the stage-C psiMag dynamics
    'E_mars': (MARS, grid(gyro_noise_var=[1e-5], accel_noise_var=[1e-2], mag_noise_var=[0.05],
                          use_mag_bias_state=[True], mag_bias_tau_s=[60.0], mag_bias_sigma_ss_deg=[30.0])),
    # ideal-NIS attempt: small base accel R + more aggressive accel inflation during manoeuvres
    'E_nis': (INDOOR, [dict(gyro_noise_var=3e-6, accel_noise_var=a, phys_dir_thresh_deg=2.0,
                            max_phys_inflation=20.0, max_vib_inflation=10.0, **extra)
                       for a in (1e-3, 2e-3)
                       for extra in ({}, {'phys_mag_test': True, 'phys_deviation_thresh': 0.03})]),
    # both-ideal attempt: aggressive accel inflation + small accel R, more gyro Q to bring ANEES down
    'F_in': (INDOOR, grid(phys_dir_thresh_deg=[2.0], max_phys_inflation=[20.0], max_vib_inflation=[10.0],
                          accel_noise_var=[1.5e-3, 2e-3], gyro_noise_var=[1e-5, 3e-5, 1e-4])),
    'F_mars': (MARS, grid(phys_dir_thresh_deg=[2.0], max_phys_inflation=[20.0], max_vib_inflation=[10.0],
                          accel_noise_var=[1.5e-3, 2e-3], gyro_noise_var=[1e-5, 3e-5],
                          mag_noise_var=[0.05], use_mag_bias_state=[True],
                          mag_bias_tau_s=[60.0], mag_bias_sigma_ss_deg=[30.0])),
}


def _task(args):
    stage, cfg, (name, site) = args
    import usque_eval as ue
    t0 = time.perf_counter()
    try:
        o = ue.run_one(name, site, cfg)
        o['error'] = ''
    except Exception as e:   # noqa: BLE001
        o = dict(name=name, site=site, error=repr(e))
    o['stage'] = stage
    o['cfg'] = json.dumps(cfg, sort_keys=True)
    o['seconds'] = round(time.perf_counter() - t0, 1)
    return o


FIELDS = ['stage', 'cfg', 'name', 'site', 'err_lt1', 'err_lt3', 'err_p50', 'err_p90', 'err_rmse',
          'anees', 'anis', 'anis_acc', 'anis_mag', 'yaw_nees_raw', 'att_qae_lt1', 'att_qae_lt3',
          'att_qae_p90', 'att_anees', 'psi_mag_deg', 'err_mean_signed', 'in3sig', 'seconds', 'error']


def run_stage(stage, workers=11):
    flights, cfgs = STAGES[stage]
    tasks = [(stage, c, f) for c in cfgs for f in flights]
    new = not os.path.exists(LOG)
    print(f"stage {stage}: {len(cfgs)} configs x {len(flights)} flights = {len(tasks)} tasks", flush=True)
    t0 = time.perf_counter()
    with open(LOG, 'a', newline='') as fh, Pool(workers) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        if new:
            w.writeheader()
        for k, o in enumerate(pool.imap_unordered(_task, tasks), 1):
            w.writerow(o)
            fh.flush()
            print(f"  [{k}/{len(tasks)} {time.perf_counter() - t0:6.0f}s] {o['name']} {o['cfg']} "
                  f"err<1 {o.get('err_lt1', float('nan')):.2f} <3 {o.get('err_lt3', float('nan')):.2f} "
                  f"ANEES {o.get('anees', float('nan')):.2f} ANIS {o.get('anis', float('nan')):.2f} {o['error']}",
                  flush=True)


if __name__ == '__main__':
    run_stage(sys.argv[1])
