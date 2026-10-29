import numpy as np

# ---------------------------------------------------------------------------
# Conventions (checked by USQUE.self_check(), which runs in the constructor)
# ---------------------------------------------------------------------------
#   world  : NED   (x North, y East, z Down)
#   body   : FRD   (x Forward, y Right, z Down)
#   q_nom  : Hamilton, scalar-first, BODY -> NAV rotation (R_nb)
#   accel  : SPECIFIC FORCE in the body frame. At rest and level it reads
#            (0, 0, -g): the accelerometer "feels" the reaction to gravity,
#            which points UP, i.e. -z in a z-down frame.
#   mag    : field vector in the body frame. mag_ref is the same field in NED,
#            (cos I, 0, +sin I) for inclination I, dip positive DOWN.
#
#   predicted accel   h_a = R_bn @ (0, 0, -1)  = -(third row of R_nb)
#   predicted mag     h_m = R_bn @ mag_ref
#
# History: h_a used to be +R_bn @ (0,0,1) (a permanent 180-deg innovation),
# _triad_to_quat returned R_bn instead of R_nb, and the
# mag_ref_inclination_deg branch had the vertical sign of mag_ref flipped.
# All three were invisible at zero attitude (identity), which is why they
# survived. self_check() now tests the model at non-trivial attitudes.
_PARAM_RENAMES = {
    'gyro_noise': 'gyro_noise_var',
    'accel_noise': 'accel_noise_var',
    'mag_noise': 'mag_noise_var',
    'bias_rw': 'bias_rw_var',
}


class USQUE:
    def __init__(self, params, initial_accel=None, initial_mag=None,
                 init_q=None, mag_ref=None):
        """
        params        dict; noise entries are VARIANCES (see *_var below).
        initial_accel specific force at rest, body frame [m/s^2] (or None)
        initial_mag   magnetometer sample, body frame (any units) or None
        init_q        optional initial body->nav quaternion. If given, TRIAD is
                      skipped (e.g. yaw seeded from RTK heading). Roll/pitch
                      must still be consistent with initial_accel.
        mag_ref       optional unit NED magnetic reference. If None it is
                      derived from mag_ref_inclination_deg (if set) or from
                      the initial attitude + initial_mag.
        """
        for old, new in _PARAM_RENAMES.items():
            if old in params:
                raise ValueError(
                    "USQUE parameter '%s' was renamed to '%s' (it is a "
                    "variance, not a sigma). Update the params dict." % (old, new))

        self.grav = 9.81
        # error state = [delta_theta(3), delta_bias_gyro(3), delta_psi_mag(1)].
        # delta_psi_mag is a scalar world-frame YAW offset applied to mag_ref (see
        # _rotate_yaw_ref / update()): it absorbs a magnetic heading BIAS (declination
        # error, per-flight hard-iron residual) that no R tuning can fix, because a
        # Kalman filter assumes zero-mean innovations and a biased mag reference gives
        # a systematically non-zero one. See mag_bias_analysis.py for the evidence.
        self.L = 7

        # --- Unscented Transform Params ---
        # Canonical Crassidis & Markley tuning: alpha=1, kappa=1 ->
        # lambda = alpha^2 (L+kappa) - L = 1, all weights positive
        # (Wm[0] = 1/7), sigma-point spread sqrt(L+lambda) = sqrt(7).
        alpha = params.get('alpha', 1.0)
        beta = params.get('beta', 2.0)
        kappa = params.get('kappa', 1.0)
        self.lambda_ = alpha ** 2 * (self.L + kappa) - self.L

        # L + lambda = alpha^2 (L + kappa) is the squared sigma-point spread.
        # With alpha=1e-3 it is 7e-6: the points sit at 0.26% of one sigma
        # with weights of +-1e5, so the "UT" degenerates into a finite
        # difference (an EKF in disguise) and loses precision. The old guard
        # (lambda <= -L) let that through; this one does not.
        if self.L + self.lambda_ < 0.5:
            raise ValueError(
                "Degenerate UT scaling: L + lambda = %g (alpha=%g, kappa=%g). "
                "Use alpha=1, kappa=1 (lambda=1)." % (self.L + self.lambda_, alpha, kappa))

        self.Wm = np.full(2 * self.L + 1, 1.0 / (2 * (self.L + self.lambda_)))
        self.Wc = np.full(2 * self.L + 1, 1.0 / (2 * (self.L + self.lambda_)))
        self.Wm[0] = self.lambda_ / (self.L + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.L + self.lambda_) + (1.0 - alpha ** 2 + beta)

        # Fading-memory covariance inflation per predict() step. Keep 1.0:
        # at 200 Hz, 1.0279 compounds to ~245x per second and makes every
        # ANEES number meaningless (any value is reachable with this knob).
        self.gamma = params.get('gamma', 1.0)

        # GRP Parameters (a=1, f=4 for standard USQUE parameters preventing singularities)
        self.a = 1.0
        self.f = 2.0 * (self.a + 1.0)
        # clip counters for _enforce_pd (how often P actually had to be repaired)
        self.pd_clip_count = 0
        self.pd_calls = 0

        # =====================================================================
        # ----- Nominal State -------------------------------------------------
        # =====================================================================
        self.q_nom = np.array([1.0, 0.0, 0.0, 0.0])  # global state quaternion
        self.bias_nom = np.zeros(3)  # global gyro bias

        # error state vector, always 0 after reset
        self.error_state = np.zeros(self.L)

        # =====================================================================
        # ---- P error covariance matrix 6x6 ----------------------------------
        # =====================================================================
        self.P = np.zeros((self.L, self.L))
        # P0_att: initial attitude VARIANCE [rad^2], scalar or per-axis
        # (roll, pitch, yaw) in the body frame. Keep it realistic: a large
        # P0 (e.g. 0.1 = 18 deg) puts the sigma points ~48 deg away, where
        # the unscented transform of the gravity model is badly non-Gaussian,
        # and the filter converts tilt manoeuvres into large spurious YAW
        # corrections (measured: ~60 deg of yaw error in 20 s indoors).
        # --- Magnetic heading BIAS state (delta_psi_mag) ----------------------
        # Modelled as a mean-reverting (Ornstein-Uhlenbeck) process: it decays toward 0
        # with time constant mag_bias_tau_s and has steady-state 1-sigma
        # mag_bias_sigma_ss_deg. Re-estimated fresh each run (psi_mag_nom starts at 0),
        # so a per-flight hard-iron/declination offset is learned within roughly one
        # time constant, whatever its sign or size on that particular flight.
        # use_mag_bias_state=False freezes it (P0/Q ~ 0) for A/B comparison.
        self.use_mag_bias_state = params.get('use_mag_bias_state', True)
        self.mag_bias_tau = params.get('mag_bias_tau_s', 300.0)
        self.mag_bias_sigma_ss = np.radians(params.get('mag_bias_sigma_ss_deg', 15.0))
        self.psi_mag_nom = 0.0     # nominal delta_psi_mag [rad]

        self.P[0:3, 0:3] = np.diag(np.ones(3) * np.asarray(params.get('P0_att', 1e-2), float))
        self.P[3:6, 3:6] = np.eye(3) * params.get('P0_bias', 1e-4)
        self.P[6, 6] = self.mag_bias_sigma_ss ** 2 if self.use_mag_bias_state else 1e-12

        # =====================================================================
        # ---- R noise matrix 6x6 ---------------------------------------------
        # =====================================================================
        # All four are VARIANCES:
        #   gyro_noise_var   sigma_g^2   [rad^2/s]      attitude Q = var * dt
        #   bias_rw_var      sigma_b^2   [rad^2/s^3]    bias Q     = var * dt
        #   accel_noise_var  direction-measurement variance (unit-vector, rad^2)
        #   mag_noise_var    direction-measurement variance (unit-vector, rad^2)
        self.gyro_noise_var = params.get('gyro_noise_var', 1e-2)
        self.bias_rw_var = params.get('bias_rw_var', 1e-5)

        self.R = np.eye(6)
        self.R[0:3, 0:3] *= params.get('accel_noise_var', 1e-1)
        self.R[3:6, 3:6] *= params.get('mag_noise_var', 1e-1)

        self.adaptive_methods = params.get('adaptive_methods', [])
        self.q_scale_factor = params.get('q_scale_factor', 5.0)

        # --- Algorithmic switches (all default to the CORRECTED behaviour;
        #     flip to False to reproduce the original results) -------------
        #
        # use_sigma0_reference: propagate the nominal quaternion as sigma
        #   point 0 (which is exactly what Crassidis & Markley's USQUE does)
        #   instead of a weighted quaternion average. Sigma point 0 is by
        #   construction the zero-error point, i.e. the nominal itself, so
        #   this is both cheaper and immune to the negative-Wm[0] hazard
        #   described above.
        self.use_sigma0_reference = params.get('use_sigma0_reference', True)
        #
        # use_corrected_Q: use sigma_g^2 * dt for the attitude process noise
        #   instead of the old (0.5*dt)^2 * sigma_g^2 (see predict()).
        self.use_corrected_Q = params.get('use_corrected_Q', True)
        #
        # use_Q_cross_term: include the attitude/bias cross-covariance term
        #   of the Van Loan discretisation. Improves gyro-bias observability.
        self.use_Q_cross_term = params.get('use_Q_cross_term', True)

        # --- Measurement oversampling compensation -----------------------
        # accel is low-passed to ~15 Hz but fused at the full 200 Hz IMU
        # rate, and mag is natively 80 Hz but interpolated to 200 Hz. Both
        # mean consecutive measurements are strongly CORRELATED while the
        # filter treats them as independent, so each physical measurement is
        # counted several times over and P shrinks far more than it should.
        # These factors inflate R to undo that double-counting. They are
        # only used by FUSION_MODE='inflate' in main; 'decimate' fixes the
        # same problem at the source by simply not fusing that often.
        self.accel_oversample = params.get('accel_oversample', 1.0)
        self.mag_oversample = params.get('mag_oversample', 1.0)
        self.R[0:3, 0:3] *= self.accel_oversample
        self.R[3:6, 3:6] *= self.mag_oversample

        # --- Consistency tuning (ANEES) ----------------------------------
        # On this dataset the attitude error is dominated by a SYSTEMATIC
        # bias, not by noise: the INSANE outdoor ground truth derives
        # orientation from the RTK-GNSS antenna baseline plus magnetometer
        # and never uses the accelerometer, whereas this filter gets its
        # tilt from gravity. Under sustained linear acceleration the two
        # disagree by roughly atan(a_horizontal / g) -- several degrees --
        # and no amount of 9-DOF tuning removes that.
        #
        # What tuning CAN fix is the filter's self-assessment. ANEES >> 3
        # means P is too small for the error actually being made. Raising R
        # alone barely moves it (steady-state P is set by the balance of Q
        # and R); raising Q alone is also not enough. Raising BOTH together
        # is what makes P honest, and it costs essentially nothing in
        # accuracy because the error is a bias the filter cannot remove
        # either way.
        #
        # These two multipliers exist so that balance can be swept without
        # disturbing the physically-meaningful noise values above.
        self.q_tune_scale = params.get('q_tune_scale', 1.0)
        self.r_tune_scale = params.get('r_tune_scale', 1.0)
        self.R *= self.r_tune_scale

        # PHYSICAL detector threshold, previously hardcoded at 0.05.
        # IMPORTANT CAVEAT: this detector triggers on the MAGNITUDE of the
        # specific force deviating from g, but a purely HORIZONTAL
        # acceleration barely changes that magnitude while rotating the
        # vector a lot: |g - a| / g = sqrt(1 + (a/g)^2), so a 0.12 g
        # sideways acceleration -- which tilts the gravity reference by
        # 6.8 deg -- shows up as under 1% of magnitude change and slips
        # straight past the 5% threshold. PHYSICAL therefore misses exactly
        # the disturbance that hurts roll/pitch most. The innovation-based
        # IAE test is the one that can actually see it.
        self.phys_deviation_thresh = params.get('phys_deviation_thresh', 0.05)
        # Direction test that closes the blind spot described above: the
        # angle between the measured specific force and the one the filter
        # PREDICTS from its own attitude. A horizontal acceleration of 0.12 g
        # is invisible to |a|-g but shows up as ~6.8 deg here. Caveat: the
        # filter's own tilt error also shows up in this angle, so keep the
        # threshold above the expected tilt error (None disables the test).
        self.phys_dir_thresh_deg = params.get('phys_dir_thresh_deg', 5.0)
        # Mag disturbance GATE ('MAG_GATE' in adaptive_methods): flags a TRANSIENT
        # departure from this flight's OWN baseline field (magnitude fraction and
        # inclination vs a raw, gyro/accel-independent-of-heading down estimate).
        # Complements delta_psi_mag, which absorbs a STATIC bias present from t=0 --
        # this catches a mid-flight event (e.g. passing near metal/motors/a building)
        # that a slow-varying bias state does not react to in time.
        self.mag_gate_mag_frac = params.get('mag_gate_mag_frac', 0.15)
        self.mag_gate_incl_rad = np.radians(params.get('mag_gate_incl_deg', 10.0))
        self.max_mag_gate_inflation = params.get('max_mag_gate_inflation', 50.0)
        self.mag_norm_ref = (float(np.linalg.norm(initial_mag)) if initial_mag is not None else None)
        self.mag_incl_ref = 0.0    # set below once self.mag_ref is final (depends on init_q/TRIAD)
        # |a| - g magnitude test on/off (blind to horizontal acceleration, and a pure vertical
        # acceleration does not change the direction the filter uses); see the ablation.
        self.phys_mag_test = params.get('phys_mag_test', False)
        # IAE per accel/mag channel (True) or pooled over all 6 dof (legacy)
        self.iae_per_channel = params.get('iae_per_channel', True)
        # Remove the (unobservable) rotation-about-gravity component from the
        # attitude gain of accel-only updates -- see update().
        self.accel_only_yaw_projection = params.get('accel_only_yaw_projection', True)
        self.accel_only_project_bias = params.get('accel_only_project_bias', True)
        # accel-only updates carry no mag information either: bundle delta_psi_mag with
        # the bias projection above (they are the same "world-frame yaw" argument).

        # --- Adaptive-R inflation caps -----------------------------------
        # NOTE: previously these three were computed in update() but two of
        # them (max_vib_inflation, max_phys_inflation) were defined here and
        # never actually used -- the caps were hardcoded instead (50.0/10.0/
        # 50.0), i.e. far above what these parameters implied. They are now
        # wired into update() below. max_vib_inflation_mag is new: the old
        # code had no separate/tunable cap for the mag side of VIBRATION at
        # all (it was hardcoded at 50.0, meaning mag was almost always
        # inflated close to its ceiling whenever throttle_pct was nonzero).
        self.max_vib_inflation = params.get('max_vib_inflation', 3.0)
        self.max_vib_inflation_mag = params.get('max_vib_inflation_mag', 8.0)
        self.max_phys_inflation = params.get('max_phys_inflation', 5.0)

        # --- Q/R coupling --------------------------------------------------
        # When the last measurement update heavily distrusted accel/mag
        # (adaptive R inflation kicked in), the filter is effectively
        # coasting on uncorrected gyro propagation. Without this, predict()
        # only grows Q with gyro rate, so P can under-represent the true
        # uncertainty building up during those stretches -- a likely
        # contributor to the very high ANEES observed. q_inflation_coupling
        # scales how strongly that happens; set to 0.0 to disable and
        # recover the old (uncoupled) behavior.
        self.q_inflation_coupling = params.get('q_inflation_coupling', 0.0)
        self._last_meas_inflation = 1.0
        self._last_accel_dir_err_deg = float('nan')
        # per-branch firing statistics of the adaptive-R logic (see adaptation_summary)
        self.adapt_stats = dict(n_updates=0, acc_infl_sum=0.0, mag_infl_sum=0.0,
                                acc_infl_any=0, mag_infl_any=0, branches={})

        # =====================================================================
        # ---- Initialising starting orientation ------------------------------
        # =====================================================================
        # Priority for the initial attitude:
        #   1. init_q            (caller-supplied, e.g. yaw seeded from RTK)
        #   2. TRIAD(accel, mag) (mag-referenced heading; uses mag_ref if given,
        #                         so a known declination is honoured)
        # Priority for the magnetic reference used by every later update:
        #   1. mag_ref                      (explicit unit NED vector)
        #   2. mag_ref_inclination_deg      (model dip, zero declination; the
        #                                    INSANE paper publishes it per site,
        #                                    e.g. 46.281 deg for Ramon Crater)
        #   3. R_nb(q0) @ mag_sample        (self-consistent with q0: the
        #                                    first mag residual is exactly 0,
        #                                    but any start-of-flight field
        #                                    disturbance is frozen in)
        # NED means dip is positive DOWN: vertical component = +sin(I).
        self.mag_ref_inclination_deg = params.get('mag_ref_inclination_deg', None)

        have_meas = initial_accel is not None and initial_mag is not None
        explicit_ref = None
        if mag_ref is not None:
            explicit_ref = np.asarray(mag_ref, float)
            explicit_ref = explicit_ref / np.linalg.norm(explicit_ref)
        elif self.mag_ref_inclination_deg is not None:
            inc = np.radians(self.mag_ref_inclination_deg)
            explicit_ref = np.array([np.cos(inc), 0.0, np.sin(inc)])

        self.init_source = 'identity'
        if init_q is not None:
            self.q_nom = np.asarray(init_q, float) / np.linalg.norm(init_q)
            self.init_source = 'init_q'
        elif have_meas:
            self.q_nom = self._triad_to_quat(initial_accel, initial_mag, explicit_ref)
            self.init_source = 'triad'

        if explicit_ref is not None:
            self.mag_ref = explicit_ref
        elif have_meas:
            self._init_mag_ref(initial_mag)
        else:
            self.mag_ref = np.array([1.0, 0.0, 0.0])

        # ---- permanent sanity check: the measurement model must agree with the
        # data convention at the initial attitude. Accel is exact by
        # construction of TRIAD (roll/pitch come from it), so a large residual
        # here means a sign/transpose bug, not noise. Mag is data dependent
        # (start-of-flight disturbance), so it is only reported.
        self.mag_incl_ref = float(np.arcsin(np.clip(self.mag_ref[2], -1.0, 1.0)))
        USQUE.self_check(46.281, 0.0)      # model/TRIAD convention, non-trivial attitudes
        USQUE.self_check(64.0, 3.0)        # a second site (Klagenfurt-like) incl. declination
        self.init_residual_accel = float('nan')
        self.init_residual_mag = float('nan')
        if initial_accel is not None:
            a0 = np.asarray(initial_accel, float)
            a0 = a0 / np.linalg.norm(a0)
            h_a0, h_m0 = self._h(self.q_nom)
            self.init_residual_accel = float(np.linalg.norm(h_a0 - a0))
            if initial_mag is not None:
                m0 = np.asarray(initial_mag, float)
                self.init_residual_mag = float(np.linalg.norm(h_m0 - m0 / np.linalg.norm(m0)))
            if self.init_residual_accel > 0.05:
                raise AssertionError(
                    "USQUE measurement model disagrees with the data at t=0: "
                    "|h_a(q0) - a_meas| = %.3f (expected ~0). Check the accel "
                    "sign (specific force, FRD) / q convention (body->nav)."
                    % self.init_residual_accel)

    # ------------------------------------------------------------------
    # Measurement model
    # ------------------------------------------------------------------
    @staticmethod
    def _h_from_q(q_, mag_ref):
        """Predicted unit accel and unit mag direction (body frame) for a
        body->nav quaternion q_ (wxyz). See the conventions block at the top."""
        qw, qx, qy, qz = q_
        # h_a = R_bn @ (0,0,-1) = -(third row of R_nb): specific force at rest
        h_a = -np.array([
            2.0 * (qx * qz - qw * qy),
            2.0 * (qw * qx + qy * qz),
            qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2
        ])
        N, E, D = mag_ref
        # h_m = R_bn @ mag_ref
        h_m = np.array([
            N * (qw ** 2 + qx ** 2 - qy ** 2 - qz ** 2) + 2 * E * (qx * qy + qw * qz) + 2 * D * (qx * qz - qw * qy),
            2 * N * (qx * qy - qw * qz) + E * (qw ** 2 - qx ** 2 + qy ** 2 - qz ** 2) + 2 * D * (qy * qz + qw * qx),
            2 * N * (qx * qz + qw * qy) + 2 * E * (qy * qz - qw * qx) + D * (qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2)
        ])
        return h_a, h_m

    def _h(self, q_):
        return self._h_from_q(q_, self.mag_ref)

    @staticmethod
    def _rotate_yaw_ref(ref, psi):
        """Rotate a NED reference vector about the DOWN axis by psi (rad). Used to apply
        delta_psi_mag: a world-frame heading offset on the magnetic reference itself,
        equivalent to saying 'the true north used for mag_ref is off by psi'."""
        c, s_ = np.cos(psi), np.sin(psi)
        N, E, D = ref
        return np.array([c * N - s_ * E, s_ * N + c * E, D])

    @staticmethod
    def self_check(inclination_deg=46.281, declination_deg=0.0, tol=1e-9):
        """Synthesise perfect measurements at several NON-TRIVIAL attitudes and
        verify that (1) both measurement residuals vanish and (2) TRIAD
        recovers the attitude. All three historical bugs gave exactly 0 at
        zero attitude, so identity is deliberately not sufficient here.
        Raises AssertionError on failure; returns the worst residual found."""
        from scipy.spatial.transform import Rotation as Rot
        g = 9.81
        inc, dec = np.radians(inclination_deg), np.radians(declination_deg)
        ref = np.array([np.cos(inc) * np.cos(dec), np.cos(inc) * np.sin(dec), np.sin(inc)])
        worst = 0.0
        for rpy in [(0, 0, 0), (20, -10, 35), (-45, 30, -80), (10, 60, 170), (-5, -50, -120)]:
            r, p, y = np.radians(rpy)
            # ZYX (yaw-pitch-roll) == extrinsic 'xyz' in scipy notation
            x_, y_, z_, w_ = Rot.from_euler('xyz', [r, p, y]).as_quat()
            q_true = np.array([w_, x_, y_, z_])
            R_nb = Rot.from_quat([x_, y_, z_, w_]).as_matrix()
            R_bn = R_nb.T
            a_meas = R_bn @ np.array([0.0, 0.0, -g])
            m_meas = R_bn @ ref
            h_a, h_m = USQUE._h_from_q(q_true, ref)
            ra = np.linalg.norm(h_a - a_meas / g)
            rm = np.linalg.norm(h_m - m_meas)
            q_t = USQUE._triad_to_quat(a_meas, m_meas, ref)
            terr = np.degrees(2.0 * np.arccos(min(1.0, abs(float(np.dot(q_t, q_true))))))
            worst = max(worst, ra, rm)
            assert ra < tol, "accel model residual %.3g at rpy=%s (sign/frame bug)" % (ra, rpy)
            assert rm < tol, "mag model residual %.3g at rpy=%s (mag_ref sign/frame bug)" % (rm, rpy)
            assert terr < 1e-4, "TRIAD attitude error %.3g deg at rpy=%s (transpose bug)" % (terr, rpy)
        return worst

    def _init_mag_ref(self, mag):
        """Self-consistent magnetic reference: the sample rotated into NED by
        the initial attitude. Its first residual is zero by construction."""
        m = np.asarray(mag, float)
        n = np.linalg.norm(m)
        m = m / n if n > 0 else m
        qw, qx, qy, qz = self.q_nom
        R_nb = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)],
        ])
        ref = R_nb @ m
        self.mag_ref = ref / np.linalg.norm(ref)

    @staticmethod
    def _quat_mult(p, q):
        pw, px, py, pz = p
        qw, qx, qy, qz = q
        return np.array([
            pw * qw - px * qx - py * qy - pz * qz,
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw
        ])

    def _error_vec_to_quat(self, p_grp):
        """GRP error vector -> error quaternion, for general (a, f):
            dq0 = (-a|p|^2 + f sqrt(f^2 + (1-a^2)|p|^2)) / (f^2 + |p|^2)
            dqv = (a + dq0) / f * p
        (a=1, f=4 reduces to (16-|p|^2)/(16+|p|^2) and 8p/(16+|p|^2).)
        Exact inverse of _quat_to_error_vec."""
        a, f = self.a, self.f
        p2 = float(np.dot(p_grp, p_grp))
        q0 = (-a * p2 + f * np.sqrt(f * f + (1.0 - a * a) * p2)) / (f * f + p2)
        q_err = np.zeros(4)
        q_err[0] = q0
        q_err[1:4] = (a + q0) / f * p_grp
        return q_err

    def _quat_to_error_vec(self, dq):
        if dq[0] < 0:
            dq = -dq
        denominator = self.a + dq[0]
        if denominator < 1e-8:
            return self.f * dq[1:4]
        p_grp = self.f * (dq[1:4] / denominator)
        return p_grp

    def _enforce_pd(self, matrix, min_eig=1e-9):
        """Symmetrize and clip negative/near-zero eigenvalues.

        The previous version only clipped the diagonal, which does not
        guarantee positive-definiteness when P has picked up small negative
        eigenvalues from roundoff (off-diagonal terms can still make the
        matrix indefinite even with a "clean" diagonal). That silently
        pushed sigma-point generation into the crude diagonal-only Cholesky
        fallback more often than necessary. Eigenvalue clipping is the
        standard robust fix and is cheap here since P is only 6x6.
        """
        matrix = (matrix + matrix.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(matrix)
        self.pd_calls += 1
        if eigvals[0] < min_eig:
            # P was actually repaired (not merely re-assembled). A high
            # fraction here means the covariance recursion is not PSD-safe
            # (P - K S K^T is less forgiving than the Joseph form).
            self.pd_clip_count += 1
        eigvals_clipped = np.clip(eigvals, min_eig, None)
        return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    @staticmethod
    def average_quaternions(quaternions, weights):
        M = np.zeros((4, 4))
        q_ref = quaternions[0]
        for i in range(len(weights)):
            q_i = quaternions[i].copy()
            if np.dot(q_i, q_ref) < 0:
                q_i = -q_i
            M += weights[i] * np.outer(q_i, q_i)
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        q_avg = eigenvectors[:, -1]
        return q_avg / np.linalg.norm(q_avg)

    @staticmethod
    def _triad_to_quat(accel, mag, mag_ref=None):
        """Body->nav quaternion from a specific-force sample and a mag sample.

        Two-vector TRIAD with gravity as the primary vector: (down, mag) are
        matched between the body frame and NED. `mag_ref` is the NED field
        direction; only its HORIZONTAL direction matters (inclination cancels),
        so passing a reference with the local declination gives a TRUE-north
        heading. None = zero declination (heading relative to magnetic north).
        """
        a = -np.array(accel, float)          # down direction in the body frame
        norm_a = np.linalg.norm(a)
        if norm_a == 0: return np.array([1.0, 0, 0, 0])
        a = a / norm_a

        m = np.array(mag, float)
        norm_m = np.linalg.norm(m)
        if norm_m == 0: return np.array([1.0, 0, 0, 0])
        m = m / norm_m

        # body triad (rows are body-frame components)
        b1 = a
        b2 = np.cross(b1, m)
        nb2 = np.linalg.norm(b2)
        if nb2 == 0: return np.array([1.0, 0, 0, 0])
        b2 = b2 / nb2
        b3 = np.cross(b1, b2)

        # nav triad from the same construction on (down, mag_ref)
        r = np.array([1.0, 0.0, 0.0]) if mag_ref is None else np.asarray(mag_ref, float)
        n1 = np.array([0.0, 0.0, 1.0])
        n2 = np.cross(n1, r)
        n2 = n2 / np.linalg.norm(n2)
        n3 = np.cross(n1, n2)

        # R_nb maps body -> nav:  R_nb b_i = n_i   =>   R_nb = sum n_i b_i^T.
        # (The old code column-stacked the NED axes expressed in the body
        # frame, which is R_bn, i.e. the TRANSPOSE of what the rest of the
        # filter expects.)
        R = np.outer(n1, b1) + np.outer(n2, b2) + np.outer(n3, b3)
        tr = np.trace(R)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw, qx, qy, qz = 0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw, qx, qy, qz = (R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw, qx, qy, qz = (R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw, qx, qy, qz = (R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S

        q = np.array([qw, qx, qy, qz])
        return q / np.linalg.norm(q)

    def _get_sigma_points(self, P):
        sigma_pts = np.zeros((2 * self.L + 1, self.L))
        P_safe = self._enforce_pd(P)
        try:
            U = np.linalg.cholesky((self.L + self.lambda_) * P_safe)
        except np.linalg.LinAlgError:
            U = np.diag(np.sqrt(np.clip(np.diag(P_safe), 1e-12, None))) * np.sqrt(self.L + self.lambda_)

        for i in range(self.L):
            sigma_pts[i + 1] = U[:, i]
            sigma_pts[self.L + i + 1] = -U[:, i]
        return sigma_pts

    def predict(self, gyro, dt):
        sigmas_err = self._get_sigma_points(self.P)

        q_propagated = np.zeros((2 * self.L + 1, 4))
        b_propagated = np.zeros((2 * self.L + 1, 3))
        # delta_psi_mag: OU decay toward 0 (dt/tau); disabled ==> decay=1 (frozen at 0).
        psi_decay = (np.exp(-dt / self.mag_bias_tau)
                    if (self.use_mag_bias_state and self.mag_bias_tau > 0) else 1.0)
        psi_propagated = np.zeros(2 * self.L + 1)

        for i in range(2 * self.L + 1):
            dp_i = sigmas_err[i, 0:3]
            db_i = sigmas_err[i, 3:6]
            psi_propagated[i] = (self.psi_mag_nom + sigmas_err[i, 6]) * psi_decay

            dq_i = self._error_vec_to_quat(dp_i)
            q_i = self._quat_mult(self.q_nom, dq_i)

            omega_i = gyro - (self.bias_nom + db_i)
            wx_i, wy_i, wz_i = omega_i

            Omega_i = np.array([
                [0, -wx_i, -wy_i, -wz_i],
                [wx_i, 0, wz_i, -wy_i],
                [wy_i, -wz_i, 0, wx_i],
                [wz_i, wy_i, -wx_i, 0]
            ])

            # Exact closed-form integration of qdot = 0.5*Omega(w)*q over one
            # step (constant-w assumption), instead of the previous
            # first-order Euler step (q_i + 0.5*Omega@q_i*dt). This is the
            # matrix exponential of (dt/2)*Omega(w), which reduces exactly
            # to cos(|w|dt/2)*I + sinc(...)*Omega(w) -- using the SAME
            # Omega_i already built above, so the frame/handedness
            # convention used elsewhere in this class is unaffected. Cuts
            # per-step integration error at higher angular rates or larger
            # dt; falls back to the old Euler-equivalent Taylor expansion at
            # very small angles for numerical stability.
            angle = np.linalg.norm(omega_i) * dt
            half_angle = angle / 2.0
            if angle > 1e-8:
                cos_term = np.cos(half_angle)
                sin_term = dt * np.sin(half_angle) / angle
            else:
                cos_term = 1.0 - (half_angle ** 2) / 2.0
                sin_term = dt * 0.5 * (1.0 - (angle ** 2) / 24.0)

            q_i_new = cos_term * q_i + sin_term * (Omega_i @ q_i)
            q_i_new /= np.linalg.norm(q_i_new)

            q_propagated[i] = q_i_new
            b_propagated[i] = self.bias_nom + db_i

        if self.use_sigma0_reference:
            # Sigma point 0 has zero error state, so q_propagated[0] IS the
            # propagated nominal. This is the reference used by USQUE as
            # published, and it avoids feeding a negative Wm[0] into the
            # eigenvector-based quaternion average.
            q_nom_new = q_propagated[0]
        else:
            q_nom_new = self.average_quaternions(q_propagated, self.Wm)
        b_nom_new = np.sum(self.Wm[:, None] * b_propagated, axis=0)
        psi_nom_new = float(np.sum(self.Wm * psi_propagated))

        sigmas_err_pred = np.zeros((2 * self.L + 1, self.L))
        q_nom_new_inv = np.array([q_nom_new[0], -q_nom_new[1], -q_nom_new[2], -q_nom_new[3]])

        for i in range(2 * self.L + 1):
            dq_new = self._quat_mult(q_nom_new_inv, q_propagated[i])
            dq_new_vec = self._quat_to_error_vec(dq_new)
            sigmas_err_pred[i, 0:3] = dq_new_vec
            sigmas_err_pred[i, 3:6] = b_propagated[i] - b_nom_new
            sigmas_err_pred[i, 6] = psi_propagated[i] - psi_nom_new

        err_pred_mean = np.sum(self.Wm[:, None] * sigmas_err_pred, axis=0)

        dp_mean = err_pred_mean[0:3]
        db_mean = err_pred_mean[3:6]
        dpsi_mean = err_pred_mean[6]

        dq_mean = self._error_vec_to_quat(dp_mean)
        self.q_nom = self._quat_mult(q_nom_new, dq_mean)
        self.q_nom /= np.linalg.norm(self.q_nom)
        self.bias_nom = b_nom_new + db_mean
        self.psi_mag_nom = psi_nom_new + dpsi_mean

        sigmas_err_pred -= err_pred_mean

        norm_gyro = np.linalg.norm(gyro)
        dynamic_Q_scale = 1.0 + self.q_scale_factor * norm_gyro

        # Couple attitude process noise to how much the last update
        # distrusted accel/mag (see __init__ / update()). sqrt() compresses
        # the (potentially large) R inflation factor into a more modest Q
        # multiplier, and the result is capped at 5x so a single strong
        # vibration/physical event doesn't blow P up unreasonably -- it
        # should make P track true uncertainty growth during "gyro-only"
        # stretches without making the filter overly conservative overall.
        meas_infl_factor = 1.0 + self.q_inflation_coupling * (
            np.sqrt(max(self._last_meas_inflation, 1.0)) - 1.0
        )
        meas_infl_factor = min(meas_infl_factor, 5.0)

        Q = np.zeros((self.L, self.L))
        # Exact OU discretisation: Var(w_k) = sigma_ss^2 * (1 - decay^2), consistent
        # with the decay already applied to the nominal and every sigma point above.
        Q[6, 6] = (self.mag_bias_sigma_ss ** 2 * (1.0 - psi_decay ** 2)
                  if self.use_mag_bias_state else 0.0)
        if self.use_corrected_Q:
            # The GRP error vector with a=1, f=2(a+1)=4 IS the small-angle
            # rotation vector in radians, so Q[0:3,0:3] must be in rad^2 and
            # the standard discrete attitude process noise from gyro white
            # noise is simply sigma_g^2 * dt.
            #
            # The original line used ((0.5*dt)**2) * sigma_g^2. At the PX4
            # IMU rate (200 Hz, dt = 0.005 s) that is 6.25e-6 * sigma_g^2
            # versus the correct 5e-3 * sigma_g^2 -- roughly 800x too small,
            # and the (0.5*dt)^2 form gets worse as dt shrinks because it
            # scales as dt^2 instead of dt. P therefore grew far too slowly,
            # the filter was badly overconfident, and the ONLY thing keeping
            # P from collapsing was the per-step multiplicative `gamma`
            # (1.0279 per step at 200 Hz compounds to ~245x per second).
            # That is an isotropic band-aid over a mis-scaled Q and is a
            # prime suspect for ANEES sitting ~7x above ideal.
            q_att = self.gyro_noise_var * self.q_tune_scale * dt * dynamic_Q_scale * meas_infl_factor
            # bias random walk also feeds attitude uncertainty over the step
            q_att += (self.bias_rw_var * dt ** 3) / 3.0
            Q[0:3, 0:3] = np.eye(3) * q_att
            Q[3:6, 3:6] = np.eye(3) * self.bias_rw_var * dt
            if self.use_Q_cross_term:
                # Van Loan attitude/bias cross-covariance. Always keeps Q
                # PSD (det = a*d + sigma_b^4*dt^4/12 > 0) and tells the
                # filter that attitude error and bias error grow together,
                # which is what makes the bias observable from attitude
                # corrections rather than only from their magnitude.
                cross = -np.eye(3) * (self.bias_rw_var * dt ** 2) / 2.0
                Q[0:3, 3:6] = cross
                Q[3:6, 0:3] = cross.T
        else:
            # --- original (mis-scaled) formulation, kept for A/B testing ---
            Q[0:3, 0:3] = np.eye(3) * ((0.5 * dt) ** 2) * (self.gyro_noise_var * dynamic_Q_scale) * meas_infl_factor
            Q[3:6, 3:6] = np.eye(3) * self.bias_rw_var * dt

        P_pred = np.zeros((self.L, self.L))
        for i in range(2 * self.L + 1):
            P_pred += self.Wc[i] * np.outer(sigmas_err_pred[i], sigmas_err_pred[i])

        P_pred += Q

        if self.gamma > 1.0:
            P_pred *= self.gamma

        self.error_state = np.zeros(self.L)
        self.P = self._enforce_pd(P_pred)

    def update(self, accel, mag=None, throttle_pct=0.0, fuse_mag=True):
        # fuse_mag=False lets the caller run an accel-only update on steps
        # where no NEW magnetometer sample has arrived, instead of re-fusing
        # an interpolated copy of a sample already used (see main).
        has_mag = (mag is not None) and fuse_mag
        meas_dim = 6 if has_mag else 3

        sigmas_err = self._get_sigma_points(self.P)
        sigmas_z = np.zeros((2 * self.L + 1, meas_dim))

        norm_a = np.linalg.norm(accel)
        a = accel / norm_a if norm_a > 0 else accel.copy()

        if has_mag:
            norm_m = np.linalg.norm(mag)
            m = mag / norm_m if norm_m > 0 else mag.copy()
            y = np.hstack([a, m])
        else:
            y = a

        for i in range(2 * self.L + 1):
            dp_i = sigmas_err[i, 0:3]
            dq_i = self._error_vec_to_quat(dp_i)
            q_i = self._quat_mult(self.q_nom, dq_i)

            if has_mag and self.use_mag_bias_state:
                mag_ref_i = self._rotate_yaw_ref(self.mag_ref, self.psi_mag_nom + sigmas_err[i, 6])
            else:
                mag_ref_i = self.mag_ref
            h_a, h_m = self._h_from_q(q_i, mag_ref_i)
            if has_mag:
                sigmas_z[i] = np.hstack((h_a, h_m))
            else:
                sigmas_z[i] = h_a

        z_pred = np.sum(self.Wm[:, None] * sigmas_z, axis=0)
        innovation = y - z_pred

        P_zz_no_R = np.zeros((meas_dim, meas_dim))
        P_xz = np.zeros((self.L, meas_dim))

        diff_z = sigmas_z - z_pred
        diff_x = sigmas_err

        for i in range(2 * self.L + 1):
            P_zz_no_R += self.Wc[i] * np.outer(diff_z[i], diff_z[i])
            P_xz += self.Wc[i] * np.outer(diff_x[i], diff_z[i])

        R_adaptive = self.R.copy() if has_mag else self.R[0:3, 0:3].copy()
        R_base = R_adaptive.copy()          # before any adaptive inflation

        accel_inflation_factor = 1.0
        mag_inflation_factor = 1.0
        st = self.adapt_stats

        try:
            P_zz_base_inv = np.linalg.inv(P_zz_no_R + R_adaptive)
        except np.linalg.LinAlgError:
            P_zz_base_inv = np.linalg.pinv(P_zz_no_R + R_adaptive)

        nis = innovation.T @ P_zz_base_inv @ innovation



        # ==========================================
        # ADAPTIVE MECHANISMS (Soft Rejection)
        # ==========================================

        if 'IAE' in self.adaptive_methods:
            S_base_ = P_zz_no_R + R_base
            if self.iae_per_channel:
                # Per-CHANNEL innovation test: the accel block and the mag block are each judged by
                # their own NIS and inflate only themselves, so a magnetic disturbance no longer
                # suppresses the accelerometer (a pooled 6-dof NIS inflated both by the same factor).
                # Each unit-vector block has 2 effective dof -> chi2(0.95, 2) = 5.99.
                blocks = [('IAE_ACC', slice(0, 3))] + ([('IAE_MAG', slice(3, 6))] if has_mag else [])
                for nm, sl in blocks:
                    Sb = S_base_[sl, sl]
                    nb = float(innovation[sl] @ np.linalg.solve(Sb, innovation[sl]))
                    if nb > 5.99:
                        sc = min(nb / 5.99, 15.0)
                        self._log_branch(nm, sc, 15.0)
                        if nm == 'IAE_ACC':
                            accel_inflation_factor = max(accel_inflation_factor, sc)
                        else:
                            mag_inflation_factor = max(mag_inflation_factor, sc)
            else:
                nis_threshold = 12.59 if has_mag else 7.81
                if nis > nis_threshold:
                    scale_factor = min(nis / nis_threshold, 15.0)
                    self._log_branch('IAE', scale_factor, 15.0)
                    accel_inflation_factor = max(accel_inflation_factor, scale_factor)
                    if has_mag:
                        mag_inflation_factor = max(mag_inflation_factor, scale_factor)

        if 'PHYSICAL' in self.adaptive_methods:
            if norm_a > 0:
                g_deviation = abs(norm_a - self.grav) / self.grav if self.phys_mag_test else 0.0
                # כאשר יש תאוצה קווית (מעל 5% סטייה מכוח המשיכה)
                if g_deviation > self.phys_deviation_thresh:
                    # ניפוח אקספוננציאלי אגרסיבי - המסנן יסמוך רק על הגירוסקופ
                    # ולא יהרוס את ה-Pitch!
                    # (was hardcoded cap of 50.0, ignoring self.max_phys_inflation)
                    phys_scale = min(1.0 + 100.0 * (g_deviation ** 2), self.max_phys_inflation)
                    self._log_branch('PHYS_MAG', phys_scale, self.max_phys_inflation)
                    accel_inflation_factor = max(accel_inflation_factor, phys_scale)

            # Direction test (see phys_dir_thresh_deg in __init__): angle
            # between the measured and the predicted specific force. Sigma
            # point 0 is the nominal, so sigmas_z[0][0:3] is h_a(q_nom).
            if self.phys_dir_thresh_deg is not None and norm_a > 0:
                cosang = float(np.clip(np.dot(a, sigmas_z[0][0:3]), -1.0, 1.0))
                ang_deg = np.degrees(np.arccos(cosang))
                self._last_accel_dir_err_deg = ang_deg
                if ang_deg > self.phys_dir_thresh_deg:
                    dir_scale = min((ang_deg / self.phys_dir_thresh_deg) ** 2,
                                    self.max_phys_inflation)
                    self._log_branch('PHYS_DIR', dir_scale, self.max_phys_inflation)
                    accel_inflation_factor = max(accel_inflation_factor, dir_scale)

        if 'MAG_GATE' in self.adaptive_methods and has_mag:
            if self.mag_norm_ref is None or self.mag_norm_ref <= 0:
                self.mag_norm_ref = norm_m       # lazy self-calibration from the first sample
            else:
                mag_frac_dev = abs(norm_m - self.mag_norm_ref) / self.mag_norm_ref
                # down direction from the filter's own nominal attitude (sigma point 0),
                # NOT from mag_ref -- independent of whatever delta_psi_mag currently is.
                down_est = -sigmas_z[0][0:3]
                incl_meas = float(np.arcsin(np.clip(np.dot(m, down_est), -1.0, 1.0)))
                incl_dev = abs(incl_meas - self.mag_incl_ref)
                if mag_frac_dev > self.mag_gate_mag_frac or incl_dev > self.mag_gate_incl_rad:
                    self._log_branch('MAG_GATE', self.max_mag_gate_inflation, self.max_mag_gate_inflation)
                    mag_inflation_factor = max(mag_inflation_factor, self.max_mag_gate_inflation)

        if 'VIBRATION' in self.adaptive_methods and throttle_pct > 0.0:
            # (was hardcoded caps of 10.0 / 50.0, ignoring self.max_vib_inflation
            # and leaving the mag side completely unconfigurable)
            vib_accel = min(1.0 + 10.0 * (throttle_pct ** 2), self.max_vib_inflation)
            self._log_branch('VIB_ACC', vib_accel, self.max_vib_inflation)
            accel_inflation_factor = max(accel_inflation_factor, vib_accel)
            if has_mag:
                vib_mag = min(1.0 + 50.0 * throttle_pct, self.max_vib_inflation_mag)
                self._log_branch('VIB_MAG', vib_mag, self.max_vib_inflation_mag)
                mag_inflation_factor = max(mag_inflation_factor, vib_mag)

        R_adaptive[0:3, 0:3] *= accel_inflation_factor
        if has_mag:
            R_adaptive[3:6, 3:6] *= mag_inflation_factor

        P_zz = P_zz_no_R + R_adaptive
        # S with the UN-inflated R: this is what the NIS/ANIS consistency check
        # must use (with the inflated R the statistic is deflated exactly when
        # the adaptive branches fire). P_zz (inflated) is used only for the gain.
        S_base = P_zz_no_R + R_base
        st['n_updates'] += 1
        st['acc_infl_sum'] += accel_inflation_factor
        st['mag_infl_sum'] += mag_inflation_factor if has_mag else 1.0
        if accel_inflation_factor > 1.0:
            st['acc_infl_any'] += 1
        if has_mag and mag_inflation_factor > 1.0:
            st['mag_infl_any'] += 1
        try:
            P_zz_inv = np.linalg.inv(P_zz)
        except np.linalg.LinAlgError:
            P_zz_inv = np.linalg.pinv(P_zz)

        K = P_xz @ P_zz_inv

        # ACCEL-ONLY updates carry NO heading information: a rotation about the
        # gravity direction leaves the predicted specific force unchanged (the
        # exact null direction of h_a). In the body-frame error state that
        # direction is d_b = R_bn(q) @ (0,0,1) = -h_a(q_nom). With a loose
        # yaw prior the gain nevertheless leaks: a linear-acceleration
        # disturbance or tilt transient is partly explained as a rotation about
        # the BODY z axis (which is observable through the tilt), and heading
        # drifts by ~2x the prior sigma (measured indoors: ~60 deg in 20 s).
        # Removing the d_b component of the attitude gain is the same
        # constraint Mahony/Madgwick impose by construction (cross-product
        # correction).
        if (not has_mag) and self.accel_only_yaw_projection:
            d_b = -sigmas_z[0][0:3]
            d_b = d_b / np.linalg.norm(d_b)
            K = K.copy()
            K[0:3] -= np.outer(d_b, d_b @ K[0:3])
            if self.accel_only_project_bias:
                # same argument for the gyro-bias component along d_b: it drives WORLD
                # yaw, which gravity cannot see, so a correction to it from an
                # accel-only update is spurious. delta_psi_mag is a SCALAR, not a
                # body-frame vector, so it has no "d_b component" to project out the
                # same way -- any accel-only movement of it comes only through a
                # genuine P[psi, attitude] correlation built up by earlier mag fusions,
                # which is ordinary (correct) correlated-state estimation, not leakage.
                K[3:6] -= np.outer(d_b, d_b @ K[3:6])
        delta_x = K @ innovation

        # The bias correction is valid for ANY measurement dimension: K is
        # L x meas_dim, so its bottom rows exist for accel-only updates too.
        # (Discarding it while still shrinking P[bg,bg] via P - K S K^T starved
        # the bias of corrections while the covariance claimed it had received them.)
        dp_update = delta_x[0:3]
        db_update = delta_x[3:6]
        dpsi_update = delta_x[6]

        dq_update = self._error_vec_to_quat(dp_update)

        self.q_nom = self._quat_mult(self.q_nom, dq_update)
        self.q_nom /= np.linalg.norm(self.q_nom)

        self.bias_nom += db_update
        self.psi_mag_nom += dpsi_update

        # Covariance update valid for ANY gain (reduces to P - K S K^T for the
        # optimal K; needed because the gain may have been projected above).
        P_update = self.P - K @ P_xz.T - P_xz @ K.T + K @ P_zz @ K.T
        self.P = self._enforce_pd(P_update)

        self.error_state = np.zeros(self.L)

        # Remember how much this update distrusted the measurements, so the
        # next predict() can grow Q accordingly (see __init__ / predict()).
        self._last_meas_inflation = max(accel_inflation_factor, mag_inflation_factor if has_mag else 1.0)

        return innovation, S_base, P_zz

    def _log_branch(self, name, scale, cap):
        b = self.adapt_stats['branches'].setdefault(name, dict(fired=0, sum=0.0, at_cap=0, cap=cap))
        b['fired'] += 1
        b['sum'] += scale
        if scale >= cap * 0.999:
            b['at_cap'] += 1

    def adaptation_summary(self):
        """Human-readable firing rate / mean inflation / at-cap fraction per adaptive
        branch. If a branch sits at its cap for most of the flight the measurement
        channel is effectively disabled, not adapted."""
        st = self.adapt_stats
        n = max(st['n_updates'], 1)
        lines = [f"adaptive-R over {st['n_updates']} updates: accel R inflated in "
                 f"{100 * st['acc_infl_any'] / n:.1f}% (mean x{st['acc_infl_sum'] / n:.2f}), "
                 f"mag R inflated in {100 * st['mag_infl_any'] / n:.1f}% (mean x{st['mag_infl_sum'] / n:.2f})"]
        for name, b in sorted(st['branches'].items()):
            lines.append(f"    {name:<9} fires {100 * b['fired'] / n:5.1f}%  mean scale x{b['sum'] / max(b['fired'], 1):.2f}"
                         f"  at cap (x{b['cap']:g}) {100 * b['at_cap'] / max(b['fired'], 1):5.1f}% of its firings")
        return "\n".join(lines)

    @property
    def state(self):
        return np.hstack((self.q_nom, self.bias_nom))

    @state.setter
    def state(self, value):
        self.q_nom = value[0:4]
        self.bias_nom = value[4:7]