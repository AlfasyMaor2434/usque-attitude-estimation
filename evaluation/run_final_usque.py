"""
run_final_usque.py -- final USQUE run on indoor_1..3 and mars_5..7 with the
selected parameters (final_config.json), producing the figures, summary.csv
and params_used.json used by the results report.

    python run_final_usque.py
"""
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

# light-surface categorical order (fixed: flight 1/5 -> blue, 2/6 -> orange, 3/7 -> aqua)
C = ['#2a78d6', '#eb6834', '#1baf7a']
INK, INK2, MUTED, SURF, BAND = '#0b0b0b', '#52514e', '#898781', '#fcfcfb', '#d9d8d3'
plt.rcParams.update({
    'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': INK, 'axes.titlecolor': INK, 'axes.grid': True, 'grid.color': '#e6e5e0',
    'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'font.size': 9.5, 'axes.titlesize': 10.5, 'legend.frameon': False, 'lines.linewidth': 1.4,
})

INDOOR = ['indoor_1', 'indoor_2', 'indoor_3']
MARS = ['mars_5', 'mars_6', 'mars_7']


def load_config():
    with open(os.path.join(HERE, 'final_config.json')) as fh:
        return json.load(fh)


def _task(args):
    name, site, ov = args
    import usque_eval as ue
    return ue.run_one(name, site, ov, keep_arrays=True)


def movavg(t, x, win_s):
    """Centered time-window mean (x sampled at times t)."""
    out = np.empty_like(x, dtype=float)
    lo = np.searchsorted(t, t - win_s / 2)
    hi = np.searchsorted(t, t + win_s / 2)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    out[:] = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    return out


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- figures
def fig_indoor_error_time(res):
    fig, axs = plt.subplots(3, 3, figsize=(11, 7.2), sharex='row')
    for i, nm in enumerate(INDOOR):
        a = res[nm]['arrays']
        for k, lab in enumerate(('roll', 'pitch', 'yaw')):
            ax = axs[i, k]
            s3 = 3 * a['sig_att'][:, k]
            ax.fill_between(a['t_cmp'], -s3, s3, color=BAND, lw=0, label='±3σ (filter)')
            ax.plot(a['t_cmp'], a['axis_err'][:, k], color=C[i], lw=1.0, label='error')
            ax.set_ylim(-3, 3)
            if i == 0:
                ax.set_title(f"{lab} error")
            if k == 0:
                ax.set_ylabel(f"{nm}\nerror (deg)")
            if i == 2:
                ax.set_xlabel('time (s)')
    axs[0, 0].legend(loc='upper right', fontsize=8)
    fig.suptitle('Indoor flights — attitude error vs Optitrack mocap, with the filter\'s ±3σ bound', y=1.0)
    fig.tight_layout()
    save(fig, 'fig_indoor_error_time.png')


def fig_mars_yaw_error_time(res):
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=False)
    for i, nm in enumerate(MARS):
        a, r = res[nm]['arrays'], res[nm]
        s3 = 3 * np.sqrt(a['sig_err'] ** 2 + r['sigma_ref_deg'] ** 2)
        ax = axs[i]
        ax.fill_between(a['t_err'], -s3, s3, color=BAND, lw=0, label='±3σ (filter + RTK reference)')
        ax.plot(a['t_err'], a['err_signed'], color=C[i], lw=1.0, label='yaw error vs dual-RTK')
        for y in (-1, 1):
            ax.axhline(y, color=MUTED, lw=0.8, ls='--')
        for y in (-3, 3):
            ax.axhline(y, color=MUTED, lw=0.8, ls=':')
        ax.set_ylabel(f"{nm}\nyaw error (deg)")
        # show the whole error AND where the (wide, cautious) 3-sigma bound sits
        lim = max(4.0, float(np.percentile(np.abs(a['err_signed']), 99.5)) * 1.1,
                  float(np.median(s3)) * 1.15)
        ax.set_ylim(-lim, lim)
    axs[0].legend(loc='upper right', fontsize=8, ncol=2)
    axs[-1].set_xlabel('time (s)')
    fig.suptitle('Mars flights — yaw error vs the dual-RTK heading (dashed ±1°, dotted ±3°)', y=1.0)
    fig.tight_layout()
    save(fig, 'fig_mars_yaw_error_time.png')


def fig_cdf(res):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, names, ttl, xl in ((axs[0], INDOOR, 'Indoor — total attitude error vs mocap', 'attitude error (deg)'),
                               (axs[1], MARS, 'Mars — |yaw error| vs dual-RTK heading', '|yaw error| (deg)')):
        pooled = []
        for i, nm in enumerate(names):
            e = np.sort(res[nm]['arrays']['err'])
            pooled.append(e)
            ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=C[i], label=nm)
        p = np.sort(np.concatenate(pooled))
        ax.plot(p, np.arange(1, len(p) + 1) / len(p), color=INK, lw=2.0, label='pooled')
        for x, y in ((1, 0.5), (3, 0.9)):
            ax.axvline(x, color=MUTED, lw=0.8, ls='--')
            ax.axhline(y, color=MUTED, lw=0.8, ls='--')
        ax.plot([1, 3], [0.5, 0.9], 's', color=INK2, ms=7, mfc='none', label='targets (50%<1°, 90%<3°)')
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 1.0)
        ax.set_title(ttl)
        ax.set_xlabel(xl)
        ax.set_ylabel('fraction of estimates')
        ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    save(fig, 'fig_error_cdf.png')


def fig_consistency_bars(res):
    names = INDOOR + MARS
    anees = [res[n]['anees'] for n in names]
    anis = [res[n]['anis'] for n in names]
    ideal_nees = [3, 3, 3, 1, 1, 1]
    ideal_nis = [2, 2, 2, 4, 4, 4]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(names))
    for ax, vals, ideal, ttl in ((axs[0], anees, ideal_nees, 'ANEES (indoor: 3-dof attitude; mars: 1-dof yaw vs RTK)'),
                                 (axs[1], anis, ideal_nis, 'ANIS (indoor: accel, 2 dof; mars: accel+mag, 4 dof)')):
        cols = [C[i % 3] for i in range(len(names))]
        ax.bar(x, vals, 0.6, color=cols)
        for xi, idl in zip(x, ideal):
            ax.plot([xi - 0.4, xi + 0.4], [idl, idl], color=INK, lw=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=0)
        ax.set_title(ttl, fontsize=9.5)
        ax.plot([], [], color=INK, lw=2, label='ideal value')
        ax.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    save(fig, 'fig_consistency_anees_anis.png')


def fig_nees_nis_time(res):
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.4))
    for col, (names, dn, dm) in enumerate(((INDOOR, 3, 2), (MARS, 1, 4))):
        for row, (key, tkey, dof, lab) in enumerate((('nees', 't_err', dn, 'NEES'), ('nis', 't_nis', dm, 'NIS'))):
            ax = axs[row, col]
            lo, hi = chi2.ppf(0.025, dof), chi2.ppf(0.975, dof)
            ax.axhspan(lo, hi, color=BAND, lw=0, label=f'single-sample 95% χ²({dof})')
            for i, nm in enumerate(names):
                a = res[nm]['arrays']
                t, v = np.asarray(a[tkey]), np.asarray(a[key], float)
                ax.plot(t, movavg(t, v, 10.0), color=C[i], label=nm)
            ax.axhline(dof, color=INK, lw=1.4, label=f'ideal = {dof}')
            ax.set_yscale('log')
            ax.set_title(f"{'Indoor' if col == 0 else 'Mars'} — {lab} (10 s moving average)")
            ax.set_xlabel('time (s)')
            ax.legend(loc='lower right', fontsize=7.5, ncol=2)
    fig.tight_layout()
    save(fig, 'fig_nees_nis_time.png')


def fig_mars_full_attitude(res):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(3)
    w = 0.26
    for i, nm in enumerate(MARS):
        v = res[nm]['att_rmse_rpy']
        ax.bar(x + (i - 1) * w, v, w * 0.92, color=C[i], label=nm)
    ax.set_xticks(x)
    ax.set_xticklabels(['roll', 'pitch', 'yaw'])
    ax.set_ylabel('RMSE (deg)')
    ax.set_title('Mars — error vs the RTK (Wahba) attitude reference (reference-limited: its own\n'
                 'roll/pitch error is ~7–9°; yaw vs dual-RTK heading is the trustworthy figure)', fontsize=9.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'fig_mars_attitude_vs_rtk_attitude.png')


def fig_euler_examples(res):
    fig, axs = plt.subplots(3, 2, figsize=(11, 6.6), sharex='col')
    for col, nm in enumerate(('indoor_1', 'mars_5')):
        a = res[nm]['arrays']
        for k, lab in enumerate(('roll', 'pitch', 'yaw')):
            ax = axs[k, col]
            gt, est = a['eul_gt'][:, k], a['eul_est'][:, k]
            if k == 2:   # unwrap yaw so a +-180 deg crossing isn't drawn as a vertical jump
                gt = np.degrees(np.unwrap(np.radians(gt)))
                est = gt + ((est - gt + 180.0) % 360.0 - 180.0)
            ax.plot(a['t_cmp'], gt, color=INK, lw=1.8, label='reference')
            ax.plot(a['t_cmp'], est, color=C[0], lw=1.0, label='USQUE')
            ax.set_ylabel(f'{lab} (deg)')
            if k == 0:
                ref = 'Optitrack mocap' if nm.startswith('indoor') else 'RTK attitude (Wahba)'
                ax.set_title(f'{nm} — USQUE vs {ref}')
        axs[2, col].set_xlabel('time (s)')
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'fig_euler_examples.png')


def fig_containment(res):
    """T5: fraction of errors inside the filter's own 3-sigma bound (ideal ~0.997 for a
    Gaussian error; much lower = overconfident, 1.0 with a tiny ANEES = overly cautious)."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    names = INDOOR + MARS
    x = np.arange(len(names))
    vals = [res[n]['in3sig'] for n in names]
    ax.bar(x, vals, 0.6, color=[C[i % 3] for i in range(len(names))])
    ax.axhline(0.997, color=INK, lw=1.4, label='ideal 99.7% (Gaussian 3σ)')
    for xi, v in zip(x, vals):
        ax.text(xi, v, f"{100 * v:.1f}%", ha='center', va='bottom', fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(min(0.8, min(vals) - 0.05), 1.02)
    ax.set_ylabel('fraction inside ±3σ')
    ax.set_title('Errors inside the filter\'s own ±3σ bound (indoor: mean of roll/pitch/yaw; '
                 'mars: yaw incl. RTK noise)', fontsize=9.5)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    save(fig, 'fig_containment_3sigma.png')


def fig_bias(res):
    """T10: gyro-bias estimate over time, relative to its still-window seed."""
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.6), sharex=False)
    for row, names in enumerate((INDOOR, MARS)):
        for k, lab in enumerate(('x (roll)', 'y (pitch)', 'z (yaw)')):
            ax = axs[row, k]
            for i, nm in enumerate(names):
                a = res[nm]['arrays']
                d = a['bg'][:, k] - a['bg'][0, k]
                ax.plot(a['t_bg'], d, color=C[i], label=nm)
                ax.fill_between(a['t_bg'], d - a['sig_bg'][:, k], d + a['sig_bg'][:, k],
                                color=C[i], alpha=0.12, lw=0)
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_title(f"{'Indoor' if row == 0 else 'Mars'} — gyro bias {lab}")
            if k == 0:
                ax.set_ylabel('change from seed (deg/s)')
            if row == 1:
                ax.set_xlabel('time (s)')
            ax.legend(fontsize=7.5)
    fig.suptitle('Gyro-bias estimate vs its still-window seed (shaded ±1σ)', y=1.0)
    fig.tight_layout()
    save(fig, 'fig_gyro_bias_convergence.png')


def fig_tuning_tradeoff(chosen):
    path = os.path.join(HERE, 'tuning_log.csv')
    if not os.path.exists(path):
        return
    rows = [r for r in csv.DictReader(open(path)) if not r['error']]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, site, ideal in ((axs[0], 'indoor', 3.0), (axs[1], 'mars', 1.0)):
        by = {}
        for r in rows:
            if r['site'] == site:
                by.setdefault(r['cfg'], []).append(r)
        xs, ys, cs = [], [], []
        for cfg, rr in by.items():
            if len(rr) < 3:
                continue
            xs.append(np.mean([float(r['anees']) for r in rr]) / ideal)
            ys.append(np.mean([float(r['err_lt1']) for r in rr]))
            cs.append(cfg)
        ax.scatter(xs, ys, s=28, color=MUTED, label='swept configs')
        # the tuning log stores only the swept keys; a config is "the selected one" when every
        # key it sets matches the final config (most specific match wins)
        want = chosen.get(site, {})

        def matches(cfg_json):
            d = json.loads(cfg_json)
            return all(k in want and np.isclose(float(want[k]), float(v)) if not isinstance(v, bool)
                       else want.get(k) == v for k, v in d.items())
        cand = [(len(json.loads(c)), i) for i, c in enumerate(cs) if matches(c)]
        sel = [max(cand)[1]] if cand else []
        if sel:
            ax.scatter([xs[sel[0]]], [ys[sel[0]]], s=90, color=C[0], label='selected', zorder=3)
        ax.axvline(1, color=INK, lw=1.2, label='ideal ANEES')
        ax.axhline(0.5, color=MUTED, lw=0.8, ls='--', label='50% < 1° target')
        ax.set_xscale('log')
        ax.set_xlabel('ANEES / ideal')
        ax.set_ylabel('fraction of estimates < 1°')
        ax.set_title(f"{site.capitalize()} — accuracy vs consistency across the tuning sweep")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, 'fig_tuning_tradeoff.png')


def _full_task(args):
    name, site, ov = args
    import usque_eval as ue
    return ue.run_full(name, site, ov)


def main_usque_figures(cfg):
    """main_usque's own batch report + figures (RMSE bars, CDF, error bounds, approximate
    true error, bias convergence, covariance, trajectory) for each site, with the final
    parameters. Figures -> figures/main_usque/<site>/, printed report -> main_usque_report_<site>.txt."""
    import io
    import contextlib
    import re
    import main_usque as mu
    for site, names in (('indoor', INDOOR), ('mars', MARS)):
        with Pool(3) as pool:
            results = pool.map(_full_task, [(n, site, cfg[site]) for n in names])
        out_dir = os.path.join(FIG, 'main_usque', site)
        os.makedirs(out_dir, exist_ok=True)
        mu.FLIGHT_SITE, mu.RUN_MODE, mu.RESULTS_DIR = site, 1, out_dir
        mu.FUSION_RATE_HZ = float(cfg[site].get('fusion_rate_hz', 30.0))
        mode = results[0]['mode']
        plt.close('all')
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mu.report_and_plot(results, mode, do_plots=True)
        with open(os.path.join(HERE, f'main_usque_report_{site}.txt'), 'w', encoding='utf-8') as fh:
            fh.write(buf.getvalue())
        for k, num in enumerate(plt.get_fignums(), 1):
            fig = plt.figure(num)
            # prefer the first panel's own title; main_usque's suptitle is often just the run tag
            ttl = (fig.axes[0].get_title() if fig.axes and fig.axes[0].get_title() else
                   (fig._suptitle.get_text() if fig._suptitle else ''))
            slug = re.sub(r'[^A-Za-z0-9]+', '_', ttl.split('|')[0]).strip('_')[:60] or f'figure_{k}'
            fig.savefig(os.path.join(out_dir, f'{k:02d}_{slug}.png'), dpi=200, bbox_inches='tight')
        print(f"main_usque figures [{site}]: {len(plt.get_fignums())} saved to {out_dir}")
        plt.close('all')


# ---------------------------------------------------------------- main
def main():
    cfg = load_config()
    tasks = [(n, 'indoor', cfg['indoor']) for n in INDOOR] + [(n, 'mars', cfg['mars']) for n in MARS]
    with Pool(6) as pool:
        out = pool.map(_task, tasks)
    res = {o['name']: o for o in out}

    fig_indoor_error_time(res)
    fig_mars_yaw_error_time(res)
    fig_cdf(res)
    fig_consistency_bars(res)
    fig_nees_nis_time(res)
    fig_mars_full_attitude(res)
    fig_euler_examples(res)
    fig_containment(res)
    fig_bias(res)
    fig_tuning_tradeoff(cfg)

    keys = ['name', 'site', 'err_lt1', 'err_lt3', 'err_p50', 'err_p90', 'err_rmse', 'anees', 'anis',
            'anis_acc', 'anis_mag', 'yaw_nees_raw', 'sigma_ref_deg', 'in3sig', 'err_mean_signed',
            'att_qae_lt1', 'att_qae_lt3',
            'att_qae_p50', 'att_qae_p90', 'att_anees', 'psi_mag_deg']
    with open(os.path.join(HERE, 'summary.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(keys + ['att_rmse_roll', 'att_rmse_pitch', 'att_rmse_yaw'])
        for nm in INDOOR + MARS:
            r = res[nm]
            w.writerow([r.get(k, '') for k in keys] + list(r['att_rmse_rpy']))
        for site, names in (('indoor', INDOOR), ('mars', MARS)):
            e = np.concatenate([res[n]['arrays']['err'] for n in names])
            w.writerow([f'POOLED_{site}', site, np.mean(e < 1), np.mean(e < 3), np.median(e),
                        np.percentile(e, 90), np.sqrt(np.mean(e ** 2)),
                        np.mean([res[n]['anees'] for n in names]), np.mean([res[n]['anis'] for n in names]),
                        '', '', '', '', np.mean([res[n]['in3sig'] for n in names])])
    # T10 numbers: final bias change vs seed (deg/s)
    with open(os.path.join(HERE, 'bias_summary.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['name', 'seed_x', 'seed_y', 'seed_z', 'final_x', 'final_y', 'final_z',
                    'final_sig_x', 'final_sig_y', 'final_sig_z'])
        for nm in INDOOR + MARS:
            a = res[nm]['arrays']
            w.writerow([nm, *a['bg'][0], *a['bg'][-1], *a['sig_bg'][-1]])

    import usque_eval as ue
    used = {site: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in ue.make_params(cfg[site]).items()} for site in ('indoor', 'mars')}
    for site in used:
        used[site]['fusion_rate_hz'] = cfg[site].get('fusion_rate_hz', 30.0)
        used[site]['per_flight_mag_calib'] = cfg[site].get('per_flight_mag_calib', True)
    with open(os.path.join(HERE, 'params_used.json'), 'w') as fh:
        json.dump(dict(selected_overrides=cfg, full_params=used), fh, indent=2, default=float)

    for nm in INDOOR + MARS:
        r = res[nm]
        print(f"{nm:9s} err<1 {r['err_lt1']:.2f} <3 {r['err_lt3']:.2f} P50 {r['err_p50']:.2f} "
              f"P90 {r['err_p90']:.2f} ANEES {r['anees']:.2f} ANIS {r['anis']:.2f}")

    main_usque_figures(cfg)


if __name__ == '__main__':
    main()
