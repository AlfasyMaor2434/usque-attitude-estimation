"""
mag_bias_analysis.py
---------------------
Is the magnetometer error at the Mars/Klagenfurt sites RANDOM NOISE (something a
Kalman R can absorb) or a SYSTEMATIC BIAS (something R cannot fix, no matter how it
is tuned)? And does that bias explain which flights score well and which don't?

Method, per flight, independent of the USQUE filter itself:
  1. TRIAD attitude from the RAW accel+mag at each dual-RTK heading sample (using the
     site's mag_ref direction, i.e. the model reference, not a per-flight fit) gives a
     magnetometer-only yaw. Its residual against the (bias-free) RTK heading is the
     "mag innovation" a filter would see, without any filter dynamics smoothing it.
  2. Mean of that residual + a confidence interval built from an EFFECTIVE sample count
     (Sokal autocorrelation time -- consecutive residuals are highly correlated, so the
     naive N overstates precision). A mean significantly different from 0 is a bias.
  3. Ljung-Box test on the residual: random noise has no serial correlation; a slowly
     drifting bias does. This is the "not random" test.
  4. Welch PSD of the residual: white noise is flat; a bias/slow drift concentrates
     power at low frequency.
  5. Field magnitude coefficient-of-variation and inclination-vs-reference deviation,
     both from the raw body-frame mag+accel (no heading needed) -- a second, independent
     symptom of magnetic distortion. Regressed against the vibration/throttle proxy to
     check for a current-dependent (motor) component.
  6. Across flights: correlate the magnetic-anomaly metrics against each flight's Wahba
     GT's OWN roll/pitch error (vs the accelerometer tilt reference) -- since that GT is
     itself derived from the magnetometer (mag_gps_wahba), a real correlation is the
     causal link "distorted flight -> distorted GT -> flight looks bad in the report".

Run: python mag_bias_analysis.py
Output: printed tables + PNGs under results_usque/mag_diagnosis/.
"""
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))   # the repository root
from data_path import DATA_DIR  # noqa: E402
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import welch

import data_loader_USQUE as dl
import project_utils as q
import usque
import main_usque as mu
import metrics_usque as mt

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_usque", "mag_diagnosis")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_usque", "all_flights", "cache")

SITES = {
    'mars': [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19],
    'outdoor': [1],
    'transition': [1, 2, 3],
}
DECIM_HZ = 10.0     # residual series rate for the stat tests / PSD / ACF (RTK-heading-limited anyway)


# ------------------------------------------------------------------
# Per-flight raw-sensor analysis (filter-independent)
# ------------------------------------------------------------------
def triad_yaw_series(t_imu, accel, mag, mag_ref, t_query):
    """Raw-sensor TRIAD yaw (rad) at each t_query, using the site's mag_ref direction."""
    idx = np.clip(np.searchsorted(t_imu, t_query), 0, len(t_imu) - 1)
    yaws = np.empty(len(t_query))
    for k, i in enumerate(idx):
        qi = usque.USQUE._triad_to_quat(accel[i], mag[i], mag_ref)
        yaws[k] = q.to_euler(qi)[2]
    return yaws


def ljung_box(x, lags=10):
    """Manual Ljung-Box Q statistic and p-value (chi2, `lags` dof) -- tests whether the
    series has any serial correlation, i.e. is NOT white noise."""
    n = len(x)
    x = np.asarray(x, float) - np.mean(x)
    denom = np.dot(x, x)
    acf = np.array([np.dot(x[:-k], x[k:]) / denom for k in range(1, lags + 1)])
    Q = n * (n + 2) * np.sum([acf[k - 1] ** 2 / (n - k) for k in range(1, lags + 1)])
    p = float(1.0 - stats.chi2.cdf(Q, lags))
    return float(Q), p, acf


def field_anomaly(accel, mag, mag_ref):
    """Field-magnitude CV (%) and mean inclination deviation from the site reference (deg),
    both computed WITHOUT any heading -- a heading-independent symptom of distortion."""
    norm_m = np.linalg.norm(mag, axis=1)
    cv = float(np.std(norm_m) / max(np.mean(norm_m), 1e-12) * 100.0)
    down = -accel / np.maximum(np.linalg.norm(accel, axis=1, keepdims=True), 1e-9)
    m_unit = mag / np.maximum(norm_m[:, None], 1e-9)
    cos_incl = np.clip(np.einsum('ij,ij->i', m_unit, down), -1.0, 1.0)
    incl = np.degrees(np.arcsin(cos_incl))
    incl_ref = np.degrees(np.arcsin(np.clip(mag_ref[2], -1.0, 1.0)))
    return cv, float(np.mean(incl) - incl_ref), float(np.std(incl))


def analyze_flight(site, num):
    d = dl.flight_dirs(site, mu.BASE_DIR, include=[num])[0]
    t_imu, gyro, accel, mag = dl.load_sensor_data(d)
    t_hdg, yaw_hdg, _ = dl.load_rtk_heading(d)
    if t_hdg is None or len(t_hdg) < 30:
        return None
    mag_ref = dl.load_mag_reference(d)

    step = max(1, int(round(len(t_hdg) / max((t_hdg[-1] - t_hdg[0]) * DECIM_HZ, 1))))
    tq, yh = t_hdg[::step], yaw_hdg[::step]
    yaw_triad = triad_yaw_series(t_imu, accel, mag, mag_ref, tq)
    resid_deg = np.degrees((yaw_triad - yh + np.pi) % (2 * np.pi) - np.pi)

    n_eff = mt.effective_sample_size(resid_deg)
    mean_r = float(np.mean(resid_deg))
    se_r = float(np.std(resid_deg) / np.sqrt(max(n_eff, 1.0)))
    Q, p_lb, acf = ljung_box(resid_deg, lags=10)

    fs = 1.0 / np.median(np.diff(t_imu))
    accel_lp = mu.apply_low_pass_filter(accel, fs, 15.0)
    throttle = dl.load_motor_data(d, t_imu, accel=accel_lp, accel_raw=accel)
    cv, incl_dev, incl_std = field_anomaly(accel, mag, mag_ref)
    # per-sample |mag| vs throttle correlation (current-dependent hard iron)
    r_thr = float(np.corrcoef(np.linalg.norm(mag, axis=1), throttle)[0, 1]) if np.std(throttle) > 1e-6 else float('nan')

    return dict(site=site, num=num, t=tq - tq[0], resid_deg=resid_deg, mean=mean_r, se=se_r,
               n_eff=n_eff, n_raw=len(resid_deg), lb_Q=Q, lb_p=p_lb, acf=acf,
               field_cv=cv, incl_dev=incl_dev, incl_std=incl_std, mag_throttle_r=r_thr,
               fs_resid=1.0 / np.median(np.diff(tq)))


# ------------------------------------------------------------------
# Cross-flight: link the anomaly to the reported GT-own-error
# ------------------------------------------------------------------
def gt_own_error(site, num, mode='rtk_yaw'):
    """(roll_own, pitch_own, n_windows) -- the Wahba GT's own error vs accel-tilt, from
    the cached run (loose still-detector preset if it has enough windows, else default)."""
    path = os.path.join(CACHE, f"{site}_{num}_{mode}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        r = pickle.load(f)
    ls_ = (r.get('extras', {}).get('tilt_sens') or {}).get('loose') or {}
    dr, dp = list(ls_.get('gt_d_roll_vs_accel', [])), list(ls_.get('gt_d_pitch_vs_accel', []))
    if len(dr) < 2:
        a_ = r.get('ar') or {}
        dr, dp = list(a_.get('gt_d_roll_vs_accel', [])), list(a_.get('gt_d_pitch_vs_accel', []))
    if not dr:
        return None
    return (float(np.sqrt(np.mean(np.square(dr)))), float(np.sqrt(np.mean(np.square(dp)))), len(dr))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for site, ids in SITES.items():
        for n in ids:
            try:
                a = analyze_flight(site, n)
            except Exception as e:                                    # noqa: BLE001
                print(f"  !! {site}_{n}: {e}")
                continue
            if a is None:
                continue
            a['gt_own'] = gt_own_error(site, n)
            rows.append(a)

    print(f"\n{'flight':<16} {'mean bias':>10} {'+-SE':>7} {'n_eff/n':>10} "
          f"{'LB-Q(10)':>9} {'p(white)':>9} {'|mag| CV%':>10} {'incl.dev':>9} {'r(|m|,thr)':>10}")
    for a in rows:
        sig = "***" if abs(a['mean']) > 2 * a['se'] else ""
        print(f"{a['site']}_{a['num']:<11} {a['mean']:9.2f}{sig:<3} {a['se']:6.2f}  "
              f"{a['n_eff']:5.0f}/{a['n_raw']:<4} {a['lb_Q']:9.1f} {a['lb_p']:9.4f} "
              f"{a['field_cv']:10.2f} {a['incl_dev']:9.2f} {a['mag_throttle_r']:10.2f}")

    n_sig = sum(1 for a in rows if abs(a['mean']) > 2 * a['se'])
    n_white = sum(1 for a in rows if a['lb_p'] > 0.05)
    print(f"\n{n_sig}/{len(rows)} flights: mean mag-heading bias significant at 2 SE "
          f"(*** marks them above)")
    print(f"{len(rows) - n_white}/{len(rows)} flights: Ljung-Box rejects whiteness at p<0.05 "
          f"(residual is NOT random noise)")

    # ---- cross-flight correlation: anomaly vs GT-own-error ----
    have_gt = [a for a in rows if a['gt_own'] is not None]
    if len(have_gt) >= 4:
        gt_max = np.array([max(a['gt_own'][0], a['gt_own'][1]) for a in have_gt])
        incl_abs = np.array([abs(a['incl_dev']) for a in have_gt])
        bias_abs = np.array([abs(a['mean']) for a in have_gt])
        cv = np.array([a['field_cv'] for a in have_gt])
        names = [f"{a['site']}_{a['num']}" for a in have_gt]

        print(f"\n--- CROSS-FLIGHT: magnetic anomaly vs Wahba GT's OWN roll/pitch error (n={len(have_gt)}) ---")
        for label, x in (("|inclination deviation| (deg)", incl_abs),
                         ("|mean mag-heading bias| (deg)", bias_abs),
                         ("field-magnitude CV (%)", cv)):
            r_, p_ = stats.pearsonr(x, gt_max)
            print(f"  corr({label:<32}, GT-own-error) = r={r_:+.2f}  p={p_:.3f}")

        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        for a_, (label, x) in zip(ax, (("|inclination deviation| (deg)", incl_abs),
                                       ("|mean mag-heading bias| (deg)", bias_abs),
                                       ("field-magnitude CV (%)", cv))):
            r_, p_ = stats.pearsonr(x, gt_max)
            a_.scatter(x, gt_max, s=50)
            for xi, yi, nm in zip(x, gt_max, names):
                a_.annotate(nm.replace("_sensors", ""), (xi, yi), fontsize=7, xytext=(3, 3),
                           textcoords="offset points")
            k, b = np.polyfit(x, gt_max, 1)
            xs = np.linspace(x.min(), x.max(), 20)
            a_.plot(xs, k * xs + b, "r--", lw=1)
            a_.set_xlabel(label)
            a_.set_ylabel("max(GT own roll/pitch error) [deg]")
            a_.set_title(f"r={r_:+.2f}, p={p_:.3f}")
            a_.grid(True, alpha=0.4)
        fig.suptitle("Is the flight's magnetic anomaly why its Wahba GT (and hence its\n"
                     "reported roll/pitch accuracy) is bad? (mars + outdoor + transition)")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "anomaly_vs_gt_error.png"), dpi=110)

    # ---- per-flight bias bar chart with 2-SE error bars ----
    fig, ax = plt.subplots(figsize=(13, 5))
    names = [f"{a['site']}_{a['num']}" for a in rows]
    means = [a['mean'] for a in rows]
    ses = [2 * a['se'] for a in rows]
    colors = ['crimson' if abs(m) > 2 * a['se'] else 'steelblue' for m, a in zip(means, rows)]
    ax.bar(range(len(rows)), means, yerr=ses, color=colors, capsize=3)
    ax.axhline(0, color='k', lw=1)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=60, ha='right', fontsize=8)
    ax.set_ylabel("mean mag-heading bias vs RTK (deg)")
    ax.set_title("Raw-sensor TRIAD yaw residual: mean +/- 2 SE (effective-N).\n"
                 "Red = significantly non-zero -> a BIAS, not noise, at that flight/site.")
    ax.grid(axis='y', ls='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "per_flight_bias.png"), dpi=110)

    # ---- PSD + ACF for a few representative flights ----
    reps = [a for a in rows if f"{a['site']}_{a['num']}" in
            ('mars_5', 'mars_2', 'mars_8', 'mars_16')]
    if reps:
        fig, ax = plt.subplots(2, len(reps), figsize=(4.2 * len(reps), 7))
        if len(reps) == 1:
            ax = ax.reshape(2, 1)
        for k, a in enumerate(reps):
            x = a['resid_deg'] - np.mean(a['resid_deg'])
            nper = min(len(x), 64)
            if nper >= 8:
                f, Pxx = welch(x, fs=a['fs_resid'], nperseg=nper)
                ax[0, k].semilogy(f, Pxx)
            ax[0, k].set_title(f"{a['site']}_{a['num']}  PSD")
            ax[0, k].set_xlabel("Hz")
            ax[1, k].stem(range(1, len(a['acf']) + 1), a['acf'])
            ax[1, k].axhline(0, color='k', lw=0.8)
            ax[1, k].set_title("ACF (10 lags)")
            ax[1, k].set_xlabel("lag")
        fig.suptitle("Mag-heading-residual spectrum / autocorrelation: flat PSD + ACF~0 would mean "
                     "white noise; these don't.")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "psd_acf_examples.png"), dpi=110)

    print(f"\nFigures saved to {RESULTS}")
    return rows


if __name__ == "__main__":
    main()
