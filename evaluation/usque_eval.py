"""
usque_eval.py -- shared evaluation core for the final USQUE results
(indoor_1..3 vs Optitrack mocap, mars_5..7 vs dual-RTK heading / RTK attitude).

Runs main_usque.run_flight() with an explicit parameter set and turns the
result into the metrics the report uses. main_usque.py / usque.py are not
modified; run-level switches that live as module globals in main_usque
(FUSION_RATE_HZ, PER_FLIGHT_MAG_CALIB) are set per call.

Metric definitions (one place, used by both the tuner and the final run):
  indoor : error = total attitude error angle vs mocap (qAE, boresight-
           corrected as in main_usque), ANEES dof 3 (ideal 3), ANIS dof 2
           (accel only, ideal 2).
  mars   : error = |yaw - dual-RTK heading| (the trustworthy reference),
           yaw NEES dof 1 (ideal 1) computed against P_yaw + sigma_ref^2
           (the RTK heading's own noise), ANIS dof 4 (accel 2 + mag 2).
           Full attitude vs the RTK (Wahba) attitude is also returned,
           labelled reference-limited: that reference's own roll/pitch error
           is ~7-9 deg (HANDOFF section 10/11).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import main_usque as mu            # noqa: E402
import data_loader_USQUE as dl     # noqa: E402
import project_utils as q          # noqa: E402

FLIGHTS = [('indoor_1', 'indoor'), ('indoor_2', 'indoor'), ('indoor_3', 'indoor'),
           ('mars_5', 'mars'), ('mars_6', 'mars'), ('mars_7', 'mars')]
IDEAL = dict(nees_indoor=3.0, nis_indoor=2.0, nees_mars=1.0, nis_mars=4.0)


def dataset_dir(name):
    return os.path.join(mu.BASE_DIR, f"{name}_sensors")


def make_params(overrides=None):
    p = dict(mu.FILTER_PARAMS)
    p['adaptive_methods'] = list(p['adaptive_methods'])
    for k, v in (overrides or {}).items():
        if k in ('fusion_rate_hz', 'per_flight_mag_calib', 'init_mode', 'rtk_seed_avg_s',
                 'lpf_mode', 'accel_lpf_kind', 'accel_cutoff_hz'):
            continue
        p[k] = v
    return p


def run_full(name, site, overrides=None):
    """Full main_usque.run_flight result dict (with the extras main_usque's own
    report_and_plot needs), for regenerating main_usque's figures."""
    ov = dict(overrides or {})
    mu.FUSION_RATE_HZ = float(ov.get('fusion_rate_hz', 30.0))
    mu.PER_FLIGHT_MAG_CALIB = bool(ov.get('per_flight_mag_calib', True))
    mu.RTK_SEED_AVG_S = float(ov.get('rtk_seed_avg_s', 0.0))
    mu.LPF_MODE = ov.get('lpf_mode', 'accel_only')
    mu.ACCEL_LPF_KIND = ov.get('accel_lpf_kind', 'zero_phase')
    mu.ACCEL_CUTOFF_HZ = float(ov.get('accel_cutoff_hz', 15.0))
    mode = 'mocap_yaw' if site == 'indoor' else ov.get('init_mode', 'rtk_yaw')
    return mu.run_flight(dataset_dir(name), site, mode, params=make_params(ov),
                         verbose=False, want_ar=True, want_extras=True)


def run_one(name, site, overrides=None, keep_arrays=False):
    ov = dict(overrides or {})
    mu.FUSION_RATE_HZ = float(ov.get('fusion_rate_hz', 30.0))
    mu.PER_FLIGHT_MAG_CALIB = bool(ov.get('per_flight_mag_calib', True))
    mu.RTK_SEED_AVG_S = float(ov.get('rtk_seed_avg_s', 0.0))
    mu.LPF_MODE = ov.get('lpf_mode', 'accel_only')
    mu.ACCEL_LPF_KIND = ov.get('accel_lpf_kind', 'zero_phase')
    mu.ACCEL_CUTOFF_HZ = float(ov.get('accel_cutoff_hz', 15.0))
    mode = 'mocap_yaw' if site == 'indoor' else ov.get('init_mode', 'rtk_yaw')
    r = mu.run_flight(dataset_dir(name), site, mode, params=make_params(ov),
                      verbose=False, want_ar=False, want_extras=False)
    out = dict(name=name, site=site,
               anis=float(np.mean(r['nis'])), dof_meas=int(r['dof_meas']),
               anis_acc=float(np.mean(r['nis_acc'])) if len(r['nis_acc']) else np.nan,
               anis_mag=float(np.mean(r['nis_mag'])) if len(r['nis_mag']) else np.nan)

    qae = np.asarray(r['qae'], float)
    ax = np.abs(np.asarray(r['axis_err'], float))
    out.update(att_qae_lt1=float(np.mean(qae < 1)), att_qae_lt3=float(np.mean(qae < 3)),
               att_qae_p50=float(np.median(qae)), att_qae_p90=float(np.percentile(qae, 90)),
               att_rmse_rpy=[float(np.sqrt(np.nanmean(ax[:, k] ** 2))) for k in range(3)],
               att_anees=float(np.mean(r['nees'])))

    sig_att = np.degrees(np.sqrt(np.maximum(np.diagonal(r['Pqq_c'], axis1=1, axis2=2), 0)))
    out['in3sig_rpy'] = [float(np.mean(ax[:, k] <= 3 * sig_att[:, k])) for k in range(3)]
    if site == 'indoor':
        err = qae
        err_signed = qae
        nees = np.asarray(r['nees'], float)
        t_err = r['t_cmp']
        sig_err = None
        out['in3sig'] = float(np.mean(out['in3sig_rpy']))
    else:
        # yaw vs dual-RTK heading over the same scored window as run_flight
        t_hdg, yaw_hdg, sig_hdg = dl.load_rtk_heading(dataset_dir(name))
        e0, e1 = r['t_cmp'][0], r['t_cmp'][-1]
        m = (t_hdg >= e0) & (t_hdg <= e1)
        t_hdg, yaw_hdg = t_hdg[m], yaw_hdg[m]
        sig_ref = np.asarray(sig_hdg)[m] if np.ndim(sig_hdg) else np.full(m.sum(), float(sig_hdg))
        idx = np.clip(np.searchsorted(r['t_imu'], t_hdg), 0, len(r['t_imu']) - 1)
        yaw_est = np.array([q.to_euler(r['q_est'][i])[2] for i in idx])
        e = (yaw_est - yaw_hdg + np.pi) % (2 * np.pi) - np.pi
        p_yaw = np.maximum(r['P_hist'][idx][:, 2, 2], 1e-12)
        nees = e ** 2 / (p_yaw + sig_ref ** 2)
        out['yaw_nees_raw'] = float(np.mean(e ** 2 / p_yaw))
        out['sigma_ref_deg'] = float(np.degrees(np.mean(sig_ref)))
        err = np.abs(np.degrees(e))
        err_signed = np.degrees(e)
        t_err = t_hdg
        sig_err = np.degrees(np.sqrt(p_yaw))
        out['in3sig'] = float(np.mean(err <= 3 * np.degrees(np.sqrt(p_yaw + sig_ref ** 2))))
        out['err_mean_signed'] = float(np.mean(err_signed))   # constant heading offset vs RTK

    out.update(err_lt1=float(np.mean(err < 1)), err_lt3=float(np.mean(err < 3)),
               err_p50=float(np.median(err)), err_p90=float(np.percentile(err, 90)),
               err_rmse=float(np.sqrt(np.mean(err ** 2))), anees=float(np.mean(nees)),
               psi_mag_deg=r.get('psi_mag_deg'))
    if r.get('boresight'):
        out['boresight_fit_rpy_deg'] = np.asarray(r['boresight']['boresight_rpy_deg'], float).tolist()
    if keep_arrays:
        out['arrays'] = dict(t_err=np.asarray(t_err) - r['t_imu'][0], err=err, err_signed=err_signed,
                             nees=nees,
                             sig_err=sig_err, t_nis=r['t_nis'] - r['t_imu'][0], nis=r['nis'],
                             qae=qae, axis_err=np.asarray(r['axis_err']),
                             t_cmp=r['t_cmp'] - r['t_imu'][0],
                             sig_att=sig_att,
                             eul_est=r['eul_est'], eul_gt=r['eul_gt'],
                             # gyro bias history (deg/s), decimated to ~20 Hz, plus its 1-sigma
                             t_bg=(r['t_imu'] - r['t_imu'][0])[::10],
                             bg=np.degrees(r['bg_hist'][::10]),
                             sig_bg=np.degrees(np.sqrt(np.maximum(
                                 np.diagonal(r['P_hist'][::10, 3:6, 3:6], axis1=1, axis2=2), 0))))
    return out


if __name__ == '__main__':
    import time
    for nm, st in (('indoor_1', 'indoor'), ('mars_5', 'mars')):
        t0 = time.perf_counter()
        o = run_one(nm, st)
        o.pop('arrays', None)
        print(f"{nm}: {time.perf_counter() - t0:.1f}s  {o}")
