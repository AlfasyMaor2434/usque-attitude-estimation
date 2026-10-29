"""
run_extra_tests.py -- T7 ablation, T8 held-out flights, T9 init source,
all starting from the selected final config (final_config.json).

    python run_extra_tests.py

Writes extra_tests.csv and figures/fig_ablation.png, fig_heldout.png,
fig_init_source.png.
"""
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_final_usque as rf   # noqa: E402  (plot style + helpers)
import matplotlib.pyplot as plt  # noqa: E402

INDOOR, MARS = rf.INDOOR, rf.MARS
HELDOUT = ['mars_1', 'mars_2', 'mars_3', 'mars_8', 'mars_9', 'mars_10', 'mars_11', 'mars_12', 'mars_13']
ALL_ADAPT = ['PHYSICAL', 'IAE', 'VIBRATION', 'MAG_GATE']


def _task(args):
    test, variant, name, site, ov = args
    import usque_eval as ue
    try:
        o = ue.run_one(name, site, ov)
        o['error'] = ''
    except Exception as e:   # noqa: BLE001
        o = dict(name=name, site=site, error=repr(e))
    o.update(test=test, variant=variant)
    return o


def build_tasks(cfg):
    tasks = []
    base_in, base_ma = cfg['indoor'], cfg['mars']
    adapt_in = base_in.get('adaptive_methods', ALL_ADAPT)
    adapt_ma = base_ma.get('adaptive_methods', ALL_ADAPT)

    # T7 ablation (indoor: accel-side branches only -- mag is off indoors)
    variants = [('final', {}, {})]
    for br in ('PHYSICAL', 'IAE', 'VIBRATION'):
        variants.append((f'no {br}', {'adaptive_methods': [a for a in adapt_in if a != br]},
                         {'adaptive_methods': [a for a in adapt_ma if a != br]}))
    variants.append(('no MAG_GATE', None, {'adaptive_methods': [a for a in adapt_ma if a != 'MAG_GATE']}))
    variants.append(('no adaptive R', {'adaptive_methods': []}, {'adaptive_methods': []}))
    psi = bool(base_ma.get('use_mag_bias_state', True))
    variants.append((f"psiMag {'off' if psi else 'on'}", None, {'use_mag_bias_state': not psi}))
    variants.append(('no per-flight mag calib', None, {'per_flight_mag_calib': False}))
    for label, d_in, d_ma in variants:
        if d_in is not None:
            tasks += [('T7', label, n, 'indoor', dict(base_in, **d_in)) for n in INDOOR]
        tasks += [('T7', label, n, 'mars', dict(base_ma, **d_ma)) for n in MARS]

    # T8 held-out mars flights (not used in this tuning)
    tasks += [('T8', 'final', n, 'mars', dict(base_ma)) for n in HELDOUT]

    # T9 init source on mars 5-7 (rtk_yaw is the 'final' T7 row)
    tasks += [('T9', 'mag_triad', n, 'mars', dict(base_ma, init_mode='mag_triad')) for n in MARS]
    return tasks


FIELDS = ['test', 'variant', 'name', 'site', 'err_lt1', 'err_lt3', 'err_p50', 'err_p90', 'err_rmse',
          'anees', 'anis', 'anis_acc', 'anis_mag', 'in3sig', 'psi_mag_deg', 'error']


def grouped_bar(ax, labels, series, names, ylabel, ttl, ref=None, reflabel=None):
    x = np.arange(len(labels))
    w = 0.8 / len(names)
    for i, nm in enumerate(names):
        ax.bar(x + (i - (len(names) - 1) / 2) * w, series[i], w * 0.9, color=rf.C[i % 3], label=nm)
    if ref is not None:
        ax.axhline(ref, color=rf.INK, lw=1.2, ls='--', label=reflabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(ttl, fontsize=9.5)
    ax.legend(fontsize=7.5, ncol=2)


def main():
    cfg = rf.load_config()
    tasks = build_tasks(cfg)
    print(f"{len(tasks)} runs", flush=True)
    with Pool(11) as pool:
        rows = []
        for k, o in enumerate(pool.imap_unordered(_task, tasks), 1):
            rows.append(o)
            print(f"  [{k}/{len(tasks)}] {o['test']} {o['variant']:24s} {o['name']:9s} "
                  f"err<1 {o.get('err_lt1', np.nan):.2f} ANEES {o.get('anees', np.nan):.2f} {o['error']}",
                  flush=True)
    with open(os.path.join(HERE, 'extra_tests.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        for o in sorted(rows, key=lambda r: (r['test'], r['variant'], r['name'])):
            w.writerow(o)

    ok = [r for r in rows if not r['error']]
    get = lambda test, var, nm, key: next((r[key] for r in ok if r['test'] == test and r['variant'] == var
                                           and r['name'] == nm), np.nan)

    # ---- T7 figure: per variant, pooled err<1 and ANEES/ideal per site
    variants = []
    for r in ok:
        if r['test'] == 'T7' and r['variant'] not in variants:
            variants.append(r['variant'])
    variants.sort(key=lambda v: (v != 'final', v))
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    for col, (names, site, ideal) in enumerate(((INDOOR, 'indoor', 3.0), (MARS, 'mars', 1.0))):
        vs = [v for v in variants if any(r['test'] == 'T7' and r['variant'] == v and r['site'] == site for r in ok)]
        grouped_bar(axs[0, col], vs, [[get('T7', v, n, 'err_lt1') for v in vs] for n in names], names,
                    'fraction < 1°', f"{site.capitalize()} — accuracy per ablation", 0.5, '50% target')
        grouped_bar(axs[1, col], vs, [[get('T7', v, n, 'anees') / ideal for v in vs] for n in names], names,
                    'ANEES / ideal', f"{site.capitalize()} — consistency per ablation", 1.0, 'ideal')
        axs[1, col].set_yscale('log')
    fig.tight_layout()
    rf.save(fig, 'fig_ablation.png')

    # ---- T8 figure: held-out mars flights
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    x = np.arange(len(HELDOUT) + 3)
    names = HELDOUT + MARS
    vals1 = [get('T8', 'final', n, 'err_lt1') for n in HELDOUT] + [get('T7', 'final', n, 'err_lt1') for n in MARS]
    vals3 = [get('T8', 'final', n, 'err_lt3') for n in HELDOUT] + [get('T7', 'final', n, 'err_lt3') for n in MARS]
    cols = [rf.MUTED] * len(HELDOUT) + [rf.C[0]] * 3
    axs[0].bar(x, vals1, 0.6, color=cols)
    axs[0].axhline(0.5, color=rf.INK, ls='--', lw=1.2, label='50% target')
    axs[0].set_title('|yaw error| < 1° vs dual-RTK (grey: held-out, blue: tuned-on)', fontsize=9.5)
    axs[1].bar(x, vals3, 0.6, color=cols)
    axs[1].axhline(0.9, color=rf.INK, ls='--', lw=1.2, label='90% target')
    axs[1].set_title('|yaw error| < 3° vs dual-RTK (grey: held-out, blue: tuned-on)', fontsize=9.5)
    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha='right')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    fig.tight_layout()
    rf.save(fig, 'fig_heldout.png')

    # ---- T9 figure: init source
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, lab in ((axs[0], 'err_rmse', 'yaw RMSE vs dual-RTK (deg)'), (axs[1], 'err_lt1', 'fraction < 1°')):
        a = [get('T7', 'final', n, key) for n in MARS]
        b = [get('T9', 'mag_triad', n, key) for n in MARS]
        grouped_bar(ax, MARS, [a, b], ['rtk_yaw (final)', 'mag_triad'], lab, lab)
    fig.suptitle('Mars — initial heading from dual-RTK vs from the magnetometer (TRIAD)', y=1.02)
    fig.tight_layout()
    rf.save(fig, 'fig_init_source.png')


if __name__ == '__main__':
    main()
