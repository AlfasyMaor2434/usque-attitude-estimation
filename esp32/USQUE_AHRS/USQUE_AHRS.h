// ============================================================================
//  USQUE_AHRS.h  --  faithful embedded port of the PC project's usque.py
//  for ESP32 + ICM-20948 (IMU + magnetometer, NO GPS).
//
//  This REPLACES UKF_AHRS.h (kept, untouched, in arduino/UKF_AHRS/ for
//  reference). That file was a different, older design -- a linear ESKF
//  backbone (Phi = I + F*dt) with a statistically-linearised UT *refinement*
//  of the propagation Jacobian, small-angle exp-map error injection, an
//  accel-bias-Z state, separate sequential gravity/mag scalar-block updates,
//  and no adaptive-R logic at all. usque.py has since diverged from that
//  design in every one of those respects, and this file is a line-by-line
//  port of the CURRENT usque.py (see that file's own header comment for the
//  full history/citations), not an update of the old one. If your firmware
//  currently uses UKF_AHRS.h and it works, there is no forced need to
//  switch -- but it no longer matches what the PC evaluation reports.
//
//  What matches usque.py, and why it matters here:
//    * Error state (7):  [dtheta(0:3)  db_gyro(3:6)  dpsi_mag(6)]  -- GRP
//      (Gibbs/Rodrigues, a=1 f=4) for dtheta, linear for the rest. NO accel-
//      bias state (Python dropped it; accel is fused as a pure unit-vector
//      direction measurement, uncorrected).
//    * predict() is a TRUE unscented propagation: all 2*7+1=15 sigma points
//      are propagated through the SAME exact closed-form quaternion
//      integration as the nominal (each with its own gyro-bias
//      perturbation), then recombined by the unscented transform. There is
//      NO linear Phi/F backbone and no statistical-regression step -- this
//      is simpler than the old file's predict(), not more complex.
//    * update() is a SINGLE joint accel+mag unscented measurement update
//      (6-dim when both are available, 3-dim accel-only), not two separate
//      sequential 3-dim scalar-block updates -- the accel/mag cross-
//      covariance is used jointly, matching usque.py's update() exactly.
//    * dpsi_mag: a 7th, SCALAR error state absorbing a magnetic HEADING
//      BIAS (per-flight/per-site hard-iron residual or declination error),
//      modelled as a mean-reverting (Ornstein-Uhlenbeck) process. See
//      mag_bias_analysis.py in the PC project for the evidence this is a
//      real, non-random bias (10-40 deg, confirmed by a Ljung-Box test on
//      21 flights) that NO measurement-noise tuning can fix on its own.
//    * Adaptive soft-inflation of R (NOT a hard reject): IAE (per-channel
//      innovation chi-square test, accel and mag judged and inflated
//      SEPARATELY), PHYSICAL (angle between the measured and predicted
//      specific force), VIBRATION (accel high-frequency-residual energy,
//      relative to a CAUSAL rolling baseline -- an absolute/constant
//      threshold saturates permanently on a flying multirotor, which is
//      what usque.py's own history section found and fixed), and MAG_GATE
//      (field magnitude + inclination vs this run's own baseline).
//    * Accel-only updates (no fresh mag sample) project the unobservable
//      rotation-about-gravity component OUT of the attitude and gyro-bias
//      gain rows (NOT out of dpsi_mag, which is a scalar with no body-frame
//      "direction" to project against -- see update()'s comment).
//    * Positive-definiteness of P is enforced with a real eigenvalue clip
//      (Jacobi rotation, ~12 sweeps -- exact for a 7x7 to float precision),
//      not a diagonal-only floor.
//
//  What is DELIBERATELY different from usque.py (embedded constraints /
//  config choices that were already fixed in the PC project's final run):
//    * use_sigma0_reference, use_corrected_Q, use_Q_cross_term are always
//      TRUE, gamma is always 1.0, q_inflation_coupling is always 0.0, and
//      phys_mag_test is always FALSE -- these are exactly the values
//      main_usque.py's FILTER_PARAMS ships with, so hardcoding them removes
//      dead code paths instead of adding a behavioural difference.
//    * Per-flight hard-iron sphere-fit calibration (data_loader_USQUE.
//      per_flight_hard_iron_correct) has no batch-data embedded analogue --
//      see USQUE_AHRS_MagCalibration.ino, which now does the SAME
//      least-squares sphere fit (streaming, not batch) as a boot-time /
//      pre-flight calibration step instead.
//    * The 30 Hz measurement-update decimation and the causal (not
//      zero-phase) 15 Hz accel low-pass are handled in the .ino's loop, to
//      match main_usque.py's FUSION_MODE='decimate' @ 30 Hz and its offline
//      filtfilt() Butterworth as closely as a REAL-TIME filter can (a
//      causal IIR of the same order/cutoff has the same passband but is NOT
//      zero-phase -- it adds a few ms of group delay filtfilt does not).
//
//  Frame: NED world (x-North, y-East, z-Down), FRD body (x-Fwd, y-Right,
//  z-Down). Quaternion (w,x,y,z), Hamilton, BODY -> NAV (R_nb).
//
//  INPUT CONVENTION (unchanged from UKF_AHRS.h):
//    accel : specific force in FRD body, m/s^2. At rest & LEVEL this must
//            read approximately (0, 0, -9.81).
//    gyro  : body angular rate in FRD, rad/s.
//    mag   : magnetic field in FRD body, any consistent unit (normalised
//            internally). Apply hard-iron / soft-iron calibration first.
//
//  No dynamic allocation, no STL, single-precision.
// ============================================================================
#ifndef USQUE_AHRS_H
#define USQUE_AHRS_H

#include <math.h>
#include <Arduino.h>   // millis(), uint32_t -- used by VibBaseline

class USQUE_AHRS {
public:
  static const int L = 7;          // error-state dimension (fixed)
  static const int NSIGMA = 2 * L + 1; // 15 sigma points

  // Soft-inflation "gate" outcomes, for live diagnostics -- unlike
  // UKF_AHRS.h's MagGateStatus, MAG_GATE_TRIGGERED does NOT reject the
  // sample; it means the sample fired the branch and mag R was inflated
  // (see update()'s MAG_GATE block). getLastMagGateStatus() reports the
  // MOST RECENT call's outcome.
  enum MagGateStatus {
    MAG_GATE_OK                 = 0,  // sample used, gate did not trigger
    MAG_GATE_NOT_INITIALISED    = 1,  // begin() hasn't run yet
    MAG_GATE_ZERO_READING       = 2,  // |mag| ~ 0 (sensor not started / bad wiring)
    MAG_GATE_TRIGGERED          = 3,  // magnitude and/or inclination deviated -> mag R inflated (sample still used)
  };

  struct Params {
    // --- noise VARIANCES (matches usque.py's *_var naming exactly -- these
    // are variances, not sigmas; a stray sigma value here is 1e2-1e4x too
    // small and will look like a "dead" channel, see usque.py's history
    // comment on the *_noise -> *_noise_var rename). ---
    float gyro_noise_var  = 9.0e-5f;  // [rad^2/s]     attitude Q = var * dt
    float bias_rw_var     = 1.0e-10f; // [rad^2/s^3]   bias Q     = var * dt
    float accel_noise_var = 1.0e-2f;  // [rad^2]       accel unit-vector measurement variance
    float mag_noise_var   = 0.05f;    // [rad^2]       mag unit-vector measurement variance
                                       // (was 0.5 pre-dpsi_mag on the PC side; shrunk once the
                                       // bias state exists to absorb the systematic part -- see
                                       // mag_bias_analysis.py / HANDOFF.md section 11)

    // Scaled-UT shape params (usque.py: alpha=1, beta=2, kappa=1 ->
    // lambda=1, all-positive weights, spread sqrt(L+1)). Kept configurable;
    // do not set alpha this small it degenerates the UT into a finite
    // difference (see usque.py's own guard/comment).
    float alpha_ukf = 1.0f;
    float beta_ukf  = 2.0f;
    float kappa_ukf = 1.0f;

    // initial 1-sigma uncertainties. P0_att is PER-AXIS (roll,pitch,yaw) --
    // usque.py's comment: a loose ISOTROPIC prior (e.g. 18 deg on every
    // axis) puts the sigma points far into the non-Gaussian region of the
    // gravity model and lets the filter convert ordinary tilt manoeuvres
    // into large spurious YAW corrections (measured: ~60 deg in 20 s).
    // Roll/pitch come from a still-window accel average (~2 deg is
    // realistic); yaw depends on how begin()/setHeadingDeg() was seeded.
    float P0_att_roll_deg  = 2.0f;
    float P0_att_pitch_deg = 2.0f;
    float P0_att_yaw_deg   = 10.0f;   // pure-TRIAD boot; setHeadingDeg() re-tightens this, see its own comment
    float P0_bg      = 1.0e-6f;       // [rad/s]^2 -- tight: the boot still-window seed is accurate to a few 0.01 dps
    float P0_bg_no_static = 1.0e-4f;  // used instead if setGyroBias() was never called (no still window found)

    // false = usque.py behaviour (model inclination from mag_inclination_deg). true = begin()
    // takes the inclination and the |mag| baseline from its own still-window sample, keeping
    // mag_declination_deg (same as USQUE_AHRS.m). Indoors the field's dip is far from the WMM
    // value, and with the model value the MAG_GATE rejects every sample.
    bool mag_ref_from_boot = false;

    // --- Magnetic heading BIAS state (dpsi_mag) ---
    bool  use_mag_bias_state   = true;
    float mag_bias_tau_s       = 300.0f;  // OU time constant [s]
    float mag_bias_sigma_ss_deg = 15.0f;  // OU steady-state 1-sigma [deg]

    // --- Mag disturbance GATE (soft inflation, not a hard reject) ---
    float mag_gate_mag_frac = 0.15f;   // reject-worthy if | |m|-|m|_ref | / |m|_ref exceeds this
    float mag_gate_incl_deg = 10.0f;   // reject-worthy if inclination deviates from the reference by more than this
    float max_mag_gate_inflation = 50.0f;

    // --- PHYSICAL (accel-direction disturbance test) ---
    float phys_dir_thresh_deg = 5.0f;  // angle between measured and predicted specific force; <=0 disables
    float max_phys_inflation  = 5.0f;

    // --- VIBRATION (accel high-frequency residual, relative to a causal
    // rolling baseline -- see VibBaseline below) ---
    bool  use_vibration = true;
    float vib_rel_span  = 1.0f;        // excursion-above-baseline normaliser (1.0 = saturates at 2x the recent floor)
    float max_vib_inflation     = 3.0f;
    float max_vib_inflation_mag = 8.0f;
    float vib_hp_cutoff_hz = 15.0f;    // matches the PC project's accel low-pass cutoff (residual = raw - LPF)
    float vib_baseline_window_s = 60.0f;
    float vib_baseline_pct = 0.20f;    // 20th percentile, matches data_loader_USQUE.relative_proxy's default

    // --- IAE (per-channel innovation chi-square test) ---
    bool  use_iae = true;
    // chi2(0.95, 2 dof) = 5.99 threshold is a class constant (IAE_THRESH_ below), not per-instance
    float max_iae_inflation = 15.0f;

    // --- accel-only update: project out the unobservable rotation-about-
    // gravity component from the attitude AND gyro-bias gain rows (NOT
    // dpsi_mag -- see update()'s comment). ---
    bool accel_only_projection = true;

    // Local magnetic reference (NED unit vector). Set inclination/
    // declination for your site; both roll/pitch-only TRIAD (declination=0
    // effect on heading only) and full TRIAD use this. See setMagRef().
    float mag_inclination_deg = 46.281f;  // e.g. Ramon Crater/Negev default; SET FOR YOUR SITE
    float mag_declination_deg = 0.0f;

    float g = 9.81f;
  };

  USQUE_AHRS() { setParams(Params()); reset(); }

  void setParams(const Params& p) {
    P_ = p;
    nLambda_ = computeUkfWeights(P_.alpha_ukf, P_.beta_ukf, P_.kappa_ukf, Wm_, Wc_);
    setMagRefFromInclDecl(P_.mag_inclination_deg, P_.mag_declination_deg);
    vibBaseline_.reset();
  }

  // Explicit NED magnetic reference override (unit vector, N/E/D). Use this
  // instead of setParams()'s inclination/declination fields if you have a
  // better one (e.g. a WMM lookup done offline for your exact site/date).
  void setMagRef(const float ref[3]) {
    float n = sqrtf(ref[0]*ref[0] + ref[1]*ref[1] + ref[2]*ref[2]);
    if (n < 1e-9f) return;
    magRef_[0] = ref[0]/n; magRef_[1] = ref[1]/n; magRef_[2] = ref[2]/n;
    magInclRef_ = asinf(fmaxf(-1.f, fminf(1.f, magRef_[2])));
  }
  // Inclination = angle of the measured field below the horizontal given by the measured
  // gravity; declination from Params. Also fixes the |mag| baseline for the MAG_GATE.
  void calibrateMagRefFromBoot(const float accel[3], const float mag[3]) {
    float an = norm3(accel), mn = norm3(mag);
    if (an < 1e-9f || mn < 1e-9f) return;
    float dotv = -(accel[0]*mag[0] + accel[1]*mag[1] + accel[2]*mag[2]) / (an * mn);   // down = -accel/|accel|
    float incl = asinf(fmaxf(-1.f, fminf(1.f, dotv)));
    float dec = P_.mag_declination_deg * 0.01745329252f;
    float ref[3] = { cosf(incl)*cosf(dec), cosf(incl)*sinf(dec), sinf(incl) };
    setMagRef(ref);
    magNormRef_ = mn;
  }
  void setMagRefFromInclDecl(float inclDeg, float declDeg) {
    float inc = inclDeg * 0.01745329252f, dec = declDeg * 0.01745329252f;
    float ref[3] = { cosf(inc)*cosf(dec), cosf(inc)*sinf(dec), sinf(inc) };
    setMagRef(ref);
  }

  // ---- initialise from a static accel + mag sample via TRIAD against the
  // configured mag reference (setMagRef/setMagRefFromInclDecl). Equivalent
  // to main_usque.py's 'mag_triad' init mode. accel/mag: RAW (unnormalised)
  // body-frame readings, averaged over a still window (see the .ino). ----
  void begin(const float accel[3], const float mag[3]) {
    reset();    // already sets P0 with yaw sigma = P_.P0_att_yaw_deg
    if (P_.mag_ref_from_boot) calibrateMagRefFromBoot(accel, mag);
    triadToQuat(accel, mag, magRef_, q_);
    initialised_ = true;
  }

  // Equivalent to main_usque.py's 'rtk_yaw' init mode: TRIAD roll/pitch (as
  // begin() would compute), but the YAW is overridden ONCE from an external
  // heading you've verified some other way (phone compass, landmark, a
  // ground truth this filter has no other access to -- a magnetometer alone
  // can never establish an absolute bearing on its own, see the header
  // comment). Tightens the yaw prior accordingly (P0_att_yaw_deg would
  // otherwise assume the noisier pure-mag TRIAD case).
  void beginWithHeading(const float accel[3], const float mag[3], float heading_deg) {
    reset();
    float q0[4]; triadToQuat(accel, mag, magRef_, q0);
    float roll, pitch, yawUnused; quatToEuler(q0, roll, pitch, yawUnused);
    eulerToQuat(roll, pitch, heading_deg * 0.01745329252f, q_);
    for (int i = 0; i < 3; ++i) bg_[i] = 0.f;
    psiMag_ = 0.f;
    initP0(2.0f);     // externally-seeded yaw: same low uncertainty as roll/pitch
    initialised_ = true;
  }

  // Externally-verified heading RE-calibration, callable any time after
  // begin(). Same physical justification as beginWithHeading()'s yaw
  // override, applied mid-run instead of at boot: overrides yaw (keeping
  // the filter's current roll/pitch), and tightens the attitude-error
  // covariance block since this is a fresh, high-confidence anchor. dpsi_mag
  // is intentionally left as-is (it represents the CURRENT magnetometer's
  // own bias, still valid regardless of this correction).
  void setHeadingDeg(float heading_deg) {
    if (!initialised_) return;
    float roll, pitch, yawUnused; quatToEuler(q_, roll, pitch, yawUnused);
    eulerToQuat(roll, pitch, heading_deg * 0.01745329252f, q_);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        P_mat_[i][j] = (i == j) ? sqv(degToRad(2.0f)) : 0.f;
  }

  void reset() {
    q_[0] = 1.f; q_[1] = q_[2] = q_[3] = 0.f;
    for (int i = 0; i < 3; ++i) bg_[i] = 0.f;
    psiMag_ = 0.f;
    zero7x7(P_mat_);
    initP0(P_.P0_att_yaw_deg);
    magNormRef_ = -1.f;    // lazy-init on the first MAG_GATE-checked sample
    lastMagGateStatus_ = MAG_GATE_NOT_INITIALISED;
    lastAccelInfl_ = lastMagInfl_ = 1.f;
    vibBaseline_.reset();
    initialised_ = false;
  }

  bool initialised() const { return initialised_; }

  // Seed the nominal gyro bias directly, e.g. from a still-window average of
  // raw gyro readings at boot (true rate ~=0 while still, so the average IS
  // the bias, to noise/sqrt(N)). Call AFTER begin()/beginWithHeading() --
  // those call reset(), which zeroes bg_. Also tightens P0_bg to
  // Params::P0_bg (vs the looser P0_bg_no_static default), matching
  // main_usque.py's INIT_SIGMA choice.
  void setGyroBias(const float b[3]) {
    for (int i = 0; i < 3; ++i) bg_[i] = b[i];
    for (int i = 0; i < 3; ++i) P_mat_[3+i][3+i] = P_.P0_bg;
  }

  // ---------- PREDICT: call every IMU sample. gyro FRD rad/s, dt seconds.
  // Pure unscented propagation -- see the header comment. ----------
  void predict(const float gyro[3], float dt) {
    if (!(dt > 0.f) || dt > 0.2f) return;   // guard bad timestamps

    float chi[NSIGMA][L];
    getSigmaPoints(P_mat_, chi);

    float psiDecay = (P_.use_mag_bias_state && P_.mag_bias_tau_s > 0.f)
                        ? expf(-dt / P_.mag_bias_tau_s) : 1.f;

    float qProp[NSIGMA][4], bProp[NSIGMA][3], psiProp[NSIGMA];
    for (int s = 0; s < NSIGMA; ++s) {
      float dp[3] = { chi[s][0], chi[s][1], chi[s][2] };
      float dbg[3] = { chi[s][3], chi[s][4], chi[s][5] };
      float dq[4]; errorVecToQuat(dp, dq);
      float qi[4]; quatMul(q_, dq, qi); quatNorm(qi);
      float bgi[3] = { bg_[0]+dbg[0], bg_[1]+dbg[1], bg_[2]+dbg[2] };
      propagateQuat(qi, bgi, gyro, dt, qProp[s]);
      bProp[s][0] = bgi[0]; bProp[s][1] = bgi[1]; bProp[s][2] = bgi[2];
      psiProp[s] = (psiMag_ + chi[s][6]) * psiDecay;
    }

    // sigma point 0 has zero error state -> qProp[0] IS the propagated
    // nominal (use_sigma0_reference=True, always -- see header comment).
    float qNomNew[4] = { qProp[0][0], qProp[0][1], qProp[0][2], qProp[0][3] };
    float bNomNew[3] = { 0.f, 0.f, 0.f };
    float psiNomNew = 0.f;
    for (int s = 0; s < NSIGMA; ++s) {
      bNomNew[0] += Wm_[s]*bProp[s][0]; bNomNew[1] += Wm_[s]*bProp[s][1]; bNomNew[2] += Wm_[s]*bProp[s][2];
      psiNomNew += Wm_[s]*psiProp[s];
    }

    float qNomNewInv[4] = { qNomNew[0], -qNomNew[1], -qNomNew[2], -qNomNew[3] };
    float errPred[NSIGMA][L];
    for (int s = 0; s < NSIGMA; ++s) {
      float dqNew[4]; quatMul(qNomNewInv, qProp[s], dqNew); quatNorm(dqNew);
      float pv[3]; quatToErrorVec(dqNew, pv);
      errPred[s][0]=pv[0]; errPred[s][1]=pv[1]; errPred[s][2]=pv[2];
      errPred[s][3]=bProp[s][0]-bNomNew[0]; errPred[s][4]=bProp[s][1]-bNomNew[1]; errPred[s][5]=bProp[s][2]-bNomNew[2];
      errPred[s][6]=psiProp[s]-psiNomNew;
    }

    float errMean[L]; for (int k = 0; k < L; ++k) errMean[k] = 0.f;
    for (int s = 0; s < NSIGMA; ++s) for (int k = 0; k < L; ++k) errMean[k] += Wm_[s]*errPred[s][k];

    float dqMean[4]; errorVecToQuat(errMean, dqMean);
    quatMul(qNomNew, dqMean, q_); quatNorm(q_);
    for (int i = 0; i < 3; ++i) bg_[i] = bNomNew[i] + errMean[3+i];
    psiMag_ = psiNomNew + errMean[6];

    for (int s = 0; s < NSIGMA; ++s) for (int k = 0; k < L; ++k) errPred[s][k] -= errMean[k];

    // Q: attitude white noise + bias random walk + Van Loan cross term +
    // dpsi_mag's exact OU discretisation noise. Matches usque.py's
    // use_corrected_Q=True / use_Q_cross_term=True branch (always taken --
    // see header comment).
    float Q[L][L]; zero(&Q[0][0], L, L);
    float qAtt = P_.gyro_noise_var * dt;
    qAtt += (P_.bias_rw_var * dt*dt*dt) / 3.f;
    for (int i = 0; i < 3; ++i) { Q[i][i] = qAtt; Q[3+i][3+i] = P_.bias_rw_var * dt; }
    float cross = -(P_.bias_rw_var * dt*dt) / 2.f;
    for (int i = 0; i < 3; ++i) { Q[i][3+i] = cross; Q[3+i][i] = cross; }
    Q[6][6] = P_.use_mag_bias_state
                ? sqv(degToRad(P_.mag_bias_sigma_ss_deg)) * (1.f - psiDecay*psiDecay)
                : 0.f;

    float Ppred[L][L]; zero(&Ppred[0][0], L, L);
    for (int s = 0; s < NSIGMA; ++s)
      for (int i = 0; i < L; ++i)
        for (int j = 0; j < L; ++j)
          Ppred[i][j] += Wc_[s] * errPred[s][i] * errPred[s][j];
    for (int i = 0; i < L; ++i) for (int j = 0; j < L; ++j) Ppred[i][j] += Q[i][j];

    enforcePD(Ppred, P_mat_, L);
  }

  // ---------- UPDATE: joint accel(+mag) unscented measurement update.
  // Call at a DECIMATED rate (~30 Hz -- see the .ino), not every predict().
  // accel: specific force, FRD body. mag: FRD body field, or NULL for an
  // accel-only update (e.g. no fresh mag sample this call, or magnetometer
  // unavailable this site -- matches main_usque.py's fuse_mag=False path).
  // throttlePct: [0,1] vibration/disturbance proxy for the VIBRATION branch
  // (see VibBaseline below) -- pass 0 if you have none. ----------
  void update(const float accel[3], const float* mag, float throttlePct) {
    bool hasMag = (mag != nullptr);
    int measDim = hasMag ? 6 : 3;

    float chi[NSIGMA][L];
    getSigmaPoints(P_mat_, chi);

    float normA = norm3(accel);
    if (normA < 1e-9f) return;
    float aUnit[3] = { accel[0]/normA, accel[1]/normA, accel[2]/normA };

    float mUnit[3] = {0,0,0};
    float normM = 0.f;
    if (hasMag) {
      normM = norm3(mag);
      if (normM < 1e-9f) { hasMag = false; measDim = 3; lastMagGateStatus_ = MAG_GATE_ZERO_READING; }
      else { mUnit[0]=mag[0]/normM; mUnit[1]=mag[1]/normM; mUnit[2]=mag[2]/normM; }
    }
    if (!initialised_) return;

    float y[6] = { aUnit[0], aUnit[1], aUnit[2], mUnit[0], mUnit[1], mUnit[2] };

    float Z[NSIGMA][6];
    for (int s = 0; s < NSIGMA; ++s) {
      float dp[3] = { chi[s][0], chi[s][1], chi[s][2] };
      float dq[4]; errorVecToQuat(dp, dq);
      float qi[4]; quatMul(q_, dq, qi); quatNorm(qi);
      float ha[3];
      hAccel(qi, ha);
      Z[s][0]=ha[0]; Z[s][1]=ha[1]; Z[s][2]=ha[2];
      if (hasMag) {
        float magRefI[3];
        if (P_.use_mag_bias_state) rotateYawRef(magRef_, psiMag_ + chi[s][6], magRefI);
        else { magRefI[0]=magRef_[0]; magRefI[1]=magRef_[1]; magRefI[2]=magRef_[2]; }
        float hm[3]; hMag(qi, magRefI, hm);
        Z[s][3]=hm[0]; Z[s][4]=hm[1]; Z[s][5]=hm[2];
      }
    }

    float zbar[6] = {0,0,0,0,0,0};
    for (int s = 0; s < NSIGMA; ++s) for (int i = 0; i < measDim; ++i) zbar[i] += Wm_[s]*Z[s][i];
    float innov[6]; for (int i = 0; i < measDim; ++i) innov[i] = y[i] - zbar[i];

    float PzzNoR[6][6]; zero(&PzzNoR[0][0], 6, 6);
    float Pxz[L][6];    zero(&Pxz[0][0], L, 6);
    for (int s = 0; s < NSIGMA; ++s) {
      float dz[6]; for (int i = 0; i < measDim; ++i) dz[i] = Z[s][i] - zbar[i];
      for (int i = 0; i < measDim; ++i)
        for (int j = 0; j < measDim; ++j)
          PzzNoR[i][j] += Wc_[s]*dz[i]*dz[j];
      for (int i = 0; i < L; ++i)
        for (int j = 0; j < measDim; ++j)
          Pxz[i][j] += Wc_[s]*chi[s][i]*dz[j];
    }

    float Rbase[6][6]; zero(&Rbase[0][0], 6, 6);
    for (int i = 0; i < 3; ++i) Rbase[i][i] = P_.accel_noise_var;
    if (hasMag) for (int i = 3; i < 6; ++i) Rbase[i][i] = P_.mag_noise_var;

    float accelInfl = 1.f, magInfl = 1.f;
    lastMagGateStatus_ = hasMag ? MAG_GATE_OK : MAG_GATE_NOT_INITIALISED;  // NOT_INITIALISED reused as "n/a, accel-only" here

    // ---- IAE: per-channel innovation chi-square test ----
    if (P_.use_iae) {
      float SaccBlk[3][3];
      for (int i=0;i<3;++i) for (int j=0;j<3;++j) SaccBlk[i][j] = PzzNoR[i][j] + Rbase[i][j];
      float SaccInv[3][3];
      if (inv3(SaccBlk, SaccInv)) {
        float nb = quadForm3(innov, SaccInv);
        if (nb > IAE_THRESH_) accelInfl = fmaxf(accelInfl, fminf(nb/IAE_THRESH_, P_.max_iae_inflation));
      }
      if (hasMag) {
        float SmagBlk[3][3];
        for (int i=0;i<3;++i) for (int j=0;j<3;++j) SmagBlk[i][j] = PzzNoR[3+i][3+j] + Rbase[3+i][3+j];
        float SmagInv[3][3];
        if (inv3(SmagBlk, SmagInv)) {
          float innovMag[3] = { innov[3], innov[4], innov[5] };
          float nb = quadForm3(innovMag, SmagInv);
          if (nb > IAE_THRESH_) magInfl = fmaxf(magInfl, fminf(nb/IAE_THRESH_, P_.max_iae_inflation));
        }
      }
    }

    // ---- PHYSICAL: angle between measured and predicted specific force ----
    if (P_.phys_dir_thresh_deg > 0.f) {
      float cosang = clip1(aUnit[0]*Z[0][0] + aUnit[1]*Z[0][1] + aUnit[2]*Z[0][2]);
      float angDeg = radToDeg(acosf(cosang));
      if (angDeg > P_.phys_dir_thresh_deg) {
        float r = angDeg / P_.phys_dir_thresh_deg;
        accelInfl = fmaxf(accelInfl, fminf(r*r, P_.max_phys_inflation));
      }
    }

    // ---- MAG_GATE: field magnitude + inclination vs THIS run's own baseline ----
    if (hasMag) {
      if (magNormRef_ <= 0.f) {
        magNormRef_ = normM;   // lazy self-calibration from the first sample
      } else {
        float magFracDev = fabsf(normM - magNormRef_) / magNormRef_;
        float downEst[3] = { -Z[0][0], -Z[0][1], -Z[0][2] };
        float inclMeas = asinf(clip1(mUnit[0]*downEst[0] + mUnit[1]*downEst[1] + mUnit[2]*downEst[2]));
        float inclDev = fabsf(inclMeas - magInclRef_);
        if (magFracDev > P_.mag_gate_mag_frac || radToDeg(inclDev) > P_.mag_gate_incl_deg) {
          magInfl = fmaxf(magInfl, P_.max_mag_gate_inflation);
          lastMagGateStatus_ = MAG_GATE_TRIGGERED;
        }
      }
    }

    // ---- VIBRATION: accel high-frequency residual, relative to a causal
    // rolling baseline (see VibBaseline) ----
    if (P_.use_vibration && throttlePct > 0.f) {
      float vibAcc = fminf(1.f + 10.f*throttlePct*throttlePct, P_.max_vib_inflation);
      accelInfl = fmaxf(accelInfl, vibAcc);
      if (hasMag) {
        float vibMag = fminf(1.f + 50.f*throttlePct, P_.max_vib_inflation_mag);
        magInfl = fmaxf(magInfl, vibMag);
      }
    }

    lastAccelInfl_ = accelInfl; lastMagInfl_ = magInfl;

    float Radaptive[6][6]; copy(&Rbase[0][0], &Radaptive[0][0], 6, 6);
    for (int i = 0; i < 3; ++i) Radaptive[i][i] *= accelInfl;
    if (hasMag) for (int i = 3; i < 6; ++i) Radaptive[i][i] *= magInfl;

    float Pzz[6][6];
    for (int i=0;i<measDim;++i) for (int j=0;j<measDim;++j) Pzz[i][j] = PzzNoR[i][j] + Radaptive[i][j];

    float PzzInv[6][6];
    if (!invN(Pzz, PzzInv, measDim)) return;

    float K[L][6]; zero(&K[0][0], L, 6);
    for (int i = 0; i < L; ++i)
      for (int j = 0; j < measDim; ++j) {
        float s = 0.f;
        for (int k = 0; k < measDim; ++k) s += Pxz[i][k]*PzzInv[k][j];
        K[i][j] = s;
      }

    // ACCEL-ONLY: project the unobservable rotation-about-gravity component
    // out of the attitude AND gyro-bias gain rows -- see the header comment.
    // dpsi_mag (row 6) is a SCALAR, not a body-frame vector, so it has no
    // "d_b component" to project the same way; any accel-only movement of it
    // comes only via a genuine P[psi,attitude] correlation from earlier mag
    // fusions (ordinary correlated-state estimation, not leakage).
    if (!hasMag && P_.accel_only_projection) {
      float d_b[3] = { -Z[0][0], -Z[0][1], -Z[0][2] };
      float dn = norm3(d_b);
      if (dn > 1e-9f) {
        d_b[0]/=dn; d_b[1]/=dn; d_b[2]/=dn;
        for (int blk = 0; blk < 2; ++blk) {   // rows 0:3 (attitude), then 3:6 (bias)
          int r0 = blk*3;
          for (int j = 0; j < measDim; ++j) {
            float dj = 0.f;
            for (int i = 0; i < 3; ++i) dj += d_b[i]*K[r0+i][j];
            for (int i = 0; i < 3; ++i) K[r0+i][j] -= d_b[i]*dj;
          }
        }
      }
    }

    float dx[L]; for (int i = 0; i < L; ++i) { float s=0.f; for (int j=0;j<measDim;++j) s += K[i][j]*innov[j]; dx[i]=s; }

    float dq[4]; errorVecToQuat(dx, dq);
    quatMul(q_, dq, q_); quatNorm(q_);
    for (int i = 0; i < 3; ++i) bg_[i] += dx[3+i];
    psiMag_ += dx[6];

    // Covariance update, general form (valid for ANY gain -- needed because
    // K may have been projected above, so it is not necessarily optimal):
    //   P = P - K Pxz^T - Pxz K^T + K Pzz K^T
    float KPxzT[L][L]; zero(&KPxzT[0][0], L, L);
    for (int i=0;i<L;++i) for (int j=0;j<L;++j) { float s=0.f; for (int k=0;k<measDim;++k) s += K[i][k]*Pxz[j][k]; KPxzT[i][j]=s; }
    float KPzzKT[L][L]; zero(&KPzzKT[0][0], L, L);
    for (int i=0;i<L;++i)
      for (int j=0;j<L;++j) {
        float s = 0.f;
        for (int k=0;k<measDim;++k)
          for (int m=0;m<measDim;++m)
            s += K[i][k]*Pzz[k][m]*K[j][m];
        KPzzKT[i][j] = s;
      }
    float Pnew[L][L];
    for (int i=0;i<L;++i) for (int j=0;j<L;++j) Pnew[i][j] = P_mat_[i][j] - KPxzT[i][j] - KPxzT[j][i] + KPzzKT[i][j];
    enforcePD(Pnew, P_mat_, L);
  }

  // Call once per loop, before update(), with the RAW (unfiltered) and
  // low-pass-filtered accel for this sample -- feeds VibBaseline. Returns
  // the throttlePct value to pass to update(). No-op / returns 0 if
  // use_vibration is off.
  float updateVibrationProxy(const float accelRaw[3], const float accelLp[3]) {
    if (!P_.use_vibration) return 0.f;
    float dx=accelRaw[0]-accelLp[0], dy=accelRaw[1]-accelLp[1], dz=accelRaw[2]-accelLp[2];
    float energy = sqrtf(dx*dx+dy*dy+dz*dz);
    return vibBaseline_.push(energy, P_.vib_baseline_pct, P_.vib_rel_span);
  }

  // ---------- accessors ----------
  void getQuaternion(float q_out[4]) const { for (int i=0;i<4;++i) q_out[i]=q_[i]; }
  void getEulerDeg(float& roll, float& pitch, float& yaw) const {
    quatToEuler(q_, roll, pitch, yaw);
    roll = radToDeg(roll); pitch = radToDeg(pitch); yaw = radToDeg(yaw);
  }
  void getGyroBias(float b[3]) const { for (int i=0;i<3;++i) b[i]=bg_[i]; }
  // 1-sigma of the gyro-bias estimate [rad/s], from the filter's own covariance
  void getGyroBiasStd(float s[3]) const { for (int i=0;i<3;++i) s[i]=sqrtf(fmaxf(P_mat_[3+i][3+i], 0.f)); }
  float getPsiMagDeg() const { return radToDeg(psiMag_); }
  void getAttitudeStdDeg(float s[3]) const {
    for (int i = 0; i < 3; ++i) s[i] = radToDeg(sqrtf(fmaxf(P_mat_[i][i], 0.f)));
  }
  float getPsiMagStdDeg() const { return radToDeg(sqrtf(fmaxf(P_mat_[6][6], 0.f))); }
  int   getLastMagGateStatus() const { return lastMagGateStatus_; }
  float getLastAccelInflation() const { return lastAccelInfl_; }
  float getLastMagInflation() const { return lastMagInfl_; }

private:
  Params P_;
  float q_[4];
  float bg_[3];
  float psiMag_;
  float magRef_[3];     // unit vector, NAV frame
  float magInclRef_;    // asin(magRef_[2]) -- the MODEL's own inclination, not an empirical begin()-time sample
  float magNormRef_;    // this run's OWN baseline |mag|, lazily set from the first MAG_GATE-checked sample (<=0 = unset)
  float P_mat_[L][L];
  bool  initialised_;
  int   lastMagGateStatus_;
  float lastAccelInfl_, lastMagInfl_;

  float Wm_[NSIGMA], Wc_[NSIGMA];
  float nLambda_;
  static const float IAE_THRESH_;

  // ---- Vibration relative-baseline: excursion above a CAUSAL rolling
  // percentile of the recent accel high-frequency-residual energy. Matches
  // data_loader_USQUE.relative_proxy's PURPOSE (0 in ordinary flight, so
  // the VIBRATION branch actually adapts instead of being a constant
  // inflation -- an ABSOLUTE threshold saturates permanently on a flying
  // multirotor, see usque.py's history section). Fixed-size circular buffer
  // at ~1 Hz (60 samples ~= 60 s at the default window), sorted on demand
  // (60 floats, trivial on ESP32) -- a coarser but functionally equivalent
  // embedded analogue of the PC's 600-sample/10 Hz version.
  struct VibBaseline {
    static const int N = 64;
    float buf[N];
    int count, head;
    uint32_t lastPushMs;
    float cachedFloor;   // re-sorted only when a new sample is actually added (~1 Hz), not on every call
    void reset() { count = 0; head = 0; lastPushMs = 0; cachedFloor = 1e-3f; }
    // energy: this sample's high-frequency residual magnitude. Returns the
    // excursion-above-baseline proxy in [0,1]: 0 while energy is at/below
    // the recent floor, 1 once it reaches (1+span) x that floor. Called at
    // full IMU rate but only actually does work (~1 Hz decimation + sort)
    // when a new sample is due -- otherwise reuses the cached floor.
    float push(float energy, float pct, float span) {
      uint32_t now = millis();
      if (now - lastPushMs >= 1000 || count == 0) {
        lastPushMs = now;
        buf[head] = energy;
        head = (head + 1) % N;
        if (count < N) count++;
        if (count >= 4) {
          float sorted[N];
          for (int i = 0; i < count; ++i) sorted[i] = buf[i];
          // insertion sort -- count <= 64, trivial cost at ~1 Hz
          for (int i = 1; i < count; ++i) {
            float v = sorted[i]; int j = i - 1;
            while (j >= 0 && sorted[j] > v) { sorted[j+1] = sorted[j]; --j; }
            sorted[j+1] = v;
          }
          int idx = (int)(pct * (count - 1) + 0.5f);
          cachedFloor = fmaxf(sorted[idx], 1e-3f);
        }
      }
      if (count < 4) return 0.f;   // not enough history yet -- assume calm
      float exc = (energy / cachedFloor - 1.f) / fmaxf(span, 1e-6f);
      return fmaxf(0.f, fminf(1.f, exc));
    }
  } vibBaseline_;

  void initP0(float yawDeg) {
    zero7x7(P_mat_);
    P_mat_[0][0] = sqv(degToRad(P_.P0_att_roll_deg));
    P_mat_[1][1] = sqv(degToRad(P_.P0_att_pitch_deg));
    P_mat_[2][2] = sqv(degToRad(yawDeg));
    for (int i = 0; i < 3; ++i) P_mat_[3+i][3+i] = P_.P0_bg_no_static;
    P_mat_[6][6] = P_.use_mag_bias_state ? sqv(degToRad(P_.mag_bias_sigma_ss_deg)) : 1e-12f;
  }

  // ---- Measurement model (matches usque.py's _h_from_q exactly) ----
  //   h_a = R_bn @ (0,0,-1) = -(third row of R_nb): specific force at rest
  //   h_m = R_bn @ mag_ref
  static void hAccel(const float q[4], float ha[3]) {
    float w=q[0],x=q[1],y=q[2],z=q[3];
    ha[0] = -(2.f*(x*z - w*y));
    ha[1] = -(2.f*(w*x + y*z));
    ha[2] = -(w*w - x*x - y*y + z*z);
  }
  static void hMag(const float q[4], const float ref[3], float hm[3]) {
    float w=q[0],x=q[1],y=q[2],z=q[3];
    float N=ref[0], E=ref[1], D=ref[2];
    hm[0] = N*(w*w+x*x-y*y-z*z) + 2.f*E*(x*y+w*z) + 2.f*D*(x*z-w*y);
    hm[1] = 2.f*N*(x*y-w*z) + E*(w*w-x*x+y*y-z*z) + 2.f*D*(y*z+w*x);
    hm[2] = 2.f*N*(x*z+w*y) + 2.f*E*(y*z-w*x) + D*(w*w-x*x-y*y+z*z);
  }
  static void rotateYawRef(const float ref[3], float psi, float out[3]) {
    float c = cosf(psi), s = sinf(psi);
    float N=ref[0], E=ref[1], D=ref[2];
    out[0] = c*N - s*E; out[1] = s*N + c*E; out[2] = D;
  }

  // ---- TRIAD: body->nav quaternion from a specific-force sample and a mag
  // sample, matching usque.py's _triad_to_quat exactly (two-vector TRIAD,
  // gravity primary, mag secondary; magRef's HORIZONTAL direction encodes
  // declination). ----
  static void triadToQuat(const float accel[3], const float mag[3], const float magRef[3], float qOut[4]) {
    float a[3] = { -accel[0], -accel[1], -accel[2] };
    float an = norm3(a);
    if (an < 1e-9f) { qOut[0]=1.f; qOut[1]=qOut[2]=qOut[3]=0.f; return; }
    a[0]/=an; a[1]/=an; a[2]/=an;
    float m[3] = { mag[0], mag[1], mag[2] };
    float mn = norm3(m);
    if (mn < 1e-9f) { qOut[0]=1.f; qOut[1]=qOut[2]=qOut[3]=0.f; return; }
    m[0]/=mn; m[1]/=mn; m[2]/=mn;

    float b1[3] = { a[0], a[1], a[2] };
    float b2[3]; cross3(b1, m, b2);
    float b2n = norm3(b2);
    if (b2n < 1e-9f) { qOut[0]=1.f; qOut[1]=qOut[2]=qOut[3]=0.f; return; }
    b2[0]/=b2n; b2[1]/=b2n; b2[2]/=b2n;
    float b3[3]; cross3(b1, b2, b3);

    float n1[3] = { 0.f, 0.f, 1.f };
    float n2[3]; cross3(n1, magRef, n2);
    float n2n = norm3(n2);
    if (n2n < 1e-9f) { qOut[0]=1.f; qOut[1]=qOut[2]=qOut[3]=0.f; return; }
    n2[0]/=n2n; n2[1]/=n2n; n2[2]/=n2n;
    float n3[3]; cross3(n1, n2, n3);

    // R_nb (body->nav): R_nb b_i = n_i  =>  R = sum n_i (x) b_i
    float R[3][3];
    for (int i=0;i<3;++i) for (int j=0;j<3;++j)
      R[i][j] = n1[i]*b1[j] + n2[i]*b2[j] + n3[i]*b3[j];
    quatFromR(R, qOut);
  }

  static void quatFromR(const float R[3][3], float q[4]) {
    float tr = R[0][0]+R[1][1]+R[2][2];
    float qw,qx,qy,qz;
    if (tr > 0.f) {
      float S = sqrtf(tr+1.f)*2.f;
      qw=0.25f*S; qx=(R[2][1]-R[1][2])/S; qy=(R[0][2]-R[2][0])/S; qz=(R[1][0]-R[0][1])/S;
    } else if (R[0][0]>R[1][1] && R[0][0]>R[2][2]) {
      float S = sqrtf(1.f+R[0][0]-R[1][1]-R[2][2])*2.f;
      qw=(R[2][1]-R[1][2])/S; qx=0.25f*S; qy=(R[0][1]+R[1][0])/S; qz=(R[0][2]+R[2][0])/S;
    } else if (R[1][1]>R[2][2]) {
      float S = sqrtf(1.f+R[1][1]-R[0][0]-R[2][2])*2.f;
      qw=(R[0][2]-R[2][0])/S; qx=(R[0][1]+R[1][0])/S; qy=0.25f*S; qz=(R[1][2]+R[2][1])/S;
    } else {
      float S = sqrtf(1.f+R[2][2]-R[0][0]-R[1][1])*2.f;
      qw=(R[1][0]-R[0][1])/S; qx=(R[0][2]+R[2][0])/S; qy=(R[1][2]+R[2][1])/S; qz=0.25f*S;
    }
    q[0]=qw; q[1]=qx; q[2]=qy; q[3]=qz;
    quatNorm(q);
  }

  // ---- GRP (Gibbs/Rodrigues, a=1 f=4) error <-> quaternion, matching
  // usque.py's _error_vec_to_quat / _quat_to_error_vec exactly. ----
  static void errorVecToQuat(const float p[3], float q[4]) {
    const float a = 1.f, f = 4.f;
    float p2 = p[0]*p[0]+p[1]*p[1]+p[2]*p[2];
    float q0 = (-a*p2 + f*sqrtf(f*f + (1.f-a*a)*p2)) / (f*f+p2);
    float k = (a+q0)/f;
    q[0]=q0; q[1]=k*p[0]; q[2]=k*p[1]; q[3]=k*p[2];
  }
  static void quatToErrorVec(const float dqIn[4], float p[3]) {
    const float a = 1.f, f = 4.f;
    float dq[4] = { dqIn[0], dqIn[1], dqIn[2], dqIn[3] };
    if (dq[0] < 0.f) { dq[0]=-dq[0]; dq[1]=-dq[1]; dq[2]=-dq[2]; dq[3]=-dq[3]; }
    float denom = a + dq[0];
    if (denom < 1e-8f) { p[0]=f*dq[1]; p[1]=f*dq[2]; p[2]=f*dq[3]; return; }
    float k = f/denom;
    p[0]=k*dq[1]; p[1]=k*dq[2]; p[2]=k*dq[3];
  }

  // ---- sigma points: Cholesky of (L+lambda)*enforcePD(P) ----
  void getSigmaPoints(const float Pin[L][L], float chi[NSIGMA][L]) {
    // nLambda_ already equals L+lambda (see computeUkfWeights's return value) --
    // the sigma-point spread scale is nLambda_ * P, NOT (L+nLambda_) * P.
    float Psafe[L][L]; enforcePD(Pin, Psafe, L);
    float scaled[L][L];
    for (int i=0;i<L;++i) for (int j=0;j<L;++j) scaled[i][j] = nLambda_*Psafe[i][j];
    float Lm[L][L];
    bool ok = cholN(&scaled[0][0], &Lm[0][0], L);
    if (!ok) {
      zero(&Lm[0][0], L, L);
      for (int i=0;i<L;++i) Lm[i][i] = sqrtf(fmaxf(nLambda_*Psafe[i][i], 1e-12f));
    }
    for (int k=0;k<L;++k) chi[0][k]=0.f;
    for (int i=0;i<L;++i)
      for (int k=0;k<L;++k) {
        chi[1+i][k]   =  Lm[k][i];
        chi[L+1+i][k] = -Lm[k][i];
      }
  }

  // Symmetrise + Jacobi-eigenvalue-clip -- matches usque.py's _enforce_pd
  // (a real eigenvalue clip, not a diagonal-only floor).
  static void enforcePD(const float Min[L][L], float Mout[L][L], int n) {
    float A[L][L];
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) A[i][j] = 0.5f*(Min[i][j]+Min[j][i]);
    float V[L][L], eigval[L];
    jacobiEigen(A, V, eigval, n);
    for (int i=0;i<n;++i) if (eigval[i] < 1e-9f) eigval[i] = 1e-9f;
    for (int i=0;i<n;++i)
      for (int j=0;j<n;++j) {
        float s=0.f;
        for (int k=0;k<n;++k) s += V[i][k]*eigval[k]*V[j][k];
        Mout[i][j]=s;
      }
  }

  // Classic cyclic Jacobi rotation for a symmetric n x n matrix (n<=7):
  // exact to float precision in a fixed 12 sweeps for matrices this small.
  static void jacobiEigen(float A[L][L], float V[L][L], float eigval[L], int n) {
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) V[i][j] = (i==j)?1.f:0.f;
    for (int sweep = 0; sweep < 12; ++sweep) {
      for (int p = 0; p < n-1; ++p) {
        for (int q = p+1; q < n; ++q) {
          if (fabsf(A[p][q]) < 1e-10f) continue;
          float theta = (A[q][q]-A[p][p]) / (2.f*A[p][q]);
          float t = (theta >= 0.f ? 1.f : -1.f) / (fabsf(theta) + sqrtf(theta*theta + 1.f));
          float c = 1.f/sqrtf(t*t+1.f), s = t*c;
          float app=A[p][p], aqq=A[q][q], apq=A[p][q];
          A[p][p] = app - t*apq;
          A[q][q] = aqq + t*apq;
          A[p][q] = A[q][p] = 0.f;
          for (int i=0;i<n;++i) {
            if (i!=p && i!=q) {
              float aip=A[i][p], aiq=A[i][q];
              A[i][p]=A[p][i]= c*aip - s*aiq;
              A[i][q]=A[q][i]= s*aip + c*aiq;
            }
          }
          for (int i=0;i<n;++i) {
            float vip=V[i][p], viq=V[i][q];
            V[i][p] = c*vip - s*viq;
            V[i][q] = s*vip + c*viq;
          }
        }
      }
    }
    for (int i=0;i<n;++i) eigval[i]=A[i][i];
  }

  // ============================ small linalg =============================
  static void zero(float* A, int r, int c) { for (int i=0;i<r*c;++i) A[i]=0.f; }
  static void zero7x7(float A[L][L]) { for (int i=0;i<L;++i) for (int j=0;j<L;++j) A[i][j]=0.f; }
  static void copy(const float* A, float* B, int r, int c) { for (int i=0;i<r*c;++i) B[i]=A[i]; }
  static float norm3(const float v[3]) { return sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
  static void cross3(const float a[3], const float b[3], float o[3]) {
    o[0]=a[1]*b[2]-a[2]*b[1]; o[1]=a[2]*b[0]-a[0]*b[2]; o[2]=a[0]*b[1]-a[1]*b[0];
  }
  static float clip1(float v) { return v>1.f?1.f:(v<-1.f?-1.f:v); }
  static float sqv(float v) { return v*v; }
  static float degToRad(float deg) { return deg*0.01745329252f; }
  static float radToDeg(float rad) { return rad*57.2957795f; }
  static float quadForm3(const float v[3], const float Minv[3][3]) {
    float Mv[3]; for (int i=0;i<3;++i) { float s=0.f; for (int j=0;j<3;++j) s+=Minv[i][j]*v[j]; Mv[i]=s; }
    return v[0]*Mv[0]+v[1]*Mv[1]+v[2]*Mv[2];
  }

  // Cholesky-Banachiewicz, n <= 7, row-major flattened in/out.
  static bool cholN(const float* Min, float* Lout, int n) {
    float M[L][L];
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) M[i][j]=Min[i*n+j];
    for (int i=0;i<n;++i) M[i][i]+=1e-9f;
    float Lm[L][L];
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) Lm[i][j]=0.f;
    for (int i=0;i<n;++i) {
      for (int j=0;j<=i;++j) {
        float s = M[i][j];
        for (int k=0;k<j;++k) s -= Lm[i][k]*Lm[j][k];
        if (i==j) { if (s<=0.f) return false; Lm[i][i]=sqrtf(s); }
        else Lm[i][j] = s/Lm[j][j];
      }
    }
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) Lout[i*n+j]=Lm[i][j];
    return true;
  }

  static bool inv3(const float M[3][3], float I[3][3]) {
    float a=M[0][0],b=M[0][1],c=M[0][2], d=M[1][0],e=M[1][1],f=M[1][2], g=M[2][0],h=M[2][1],ii=M[2][2];
    float A=(e*ii-f*h), B=-(d*ii-f*g), C=(d*h-e*g);
    float det = a*A + b*B + c*C;
    if (fabsf(det) < 1e-20f) return false;
    float idet = 1.f/det;
    I[0][0]=A*idet; I[0][1]=-(b*ii-c*h)*idet; I[0][2]=(b*f-c*e)*idet;
    I[1][0]=B*idet; I[1][1]=(a*ii-c*g)*idet;  I[1][2]=-(a*f-c*d)*idet;
    I[2][0]=C*idet; I[2][1]=-(a*h-b*g)*idet;  I[2][2]=(a*e-b*d)*idet;
    return true;
  }

  // General n x n inverse via Gauss-Jordan with partial pivoting, n<=6
  // (only measDim=3 or 6 ever call this).
  static bool invN(const float Ain[6][6], float Aout[6][6], int n) {
    float a[6][12];
    for (int i=0;i<n;++i) {
      for (int j=0;j<n;++j) a[i][j] = Ain[i][j];
      for (int j=0;j<n;++j) a[i][n+j] = (i==j)?1.f:0.f;
    }
    for (int col=0; col<n; ++col) {
      int piv = col; float best = fabsf(a[col][col]);
      for (int r=col+1;r<n;++r) { float v=fabsf(a[r][col]); if (v>best) { best=v; piv=r; } }
      if (best < 1e-20f) return false;
      if (piv != col) for (int j=0;j<2*n;++j) { float t=a[col][j]; a[col][j]=a[piv][j]; a[piv][j]=t; }
      float inv = 1.f/a[col][col];
      for (int j=0;j<2*n;++j) a[col][j]*=inv;
      for (int r=0;r<n;++r) {
        if (r==col) continue;
        float f = a[r][col];
        if (f==0.f) continue;
        for (int j=0;j<2*n;++j) a[r][j] -= f*a[col][j];
      }
    }
    for (int i=0;i<n;++i) for (int j=0;j<n;++j) Aout[i][j]=a[i][n+j];
    return true;
  }

  // usque.py's exact formula: lambda = alpha^2*(L+kappa) - L (NOT Merwe's
  // kappa=3-n convention -- that was the OLD file's choice, this one
  // matches the PC project). Returns L+lambda.
  static float computeUkfWeights(float alpha, float beta, float kappa, float* Wm, float* Wc) {
    float lambda = alpha*alpha*((float)L + kappa) - (float)L;
    float nLambda = (float)L + lambda;
    float w0m = (fabsf(nLambda) > 1e-9f) ? (lambda/nLambda) : 0.f;
    float w0c = w0m + (1.f - alpha*alpha + beta);
    float wi = (nLambda > 1e-9f) ? 1.f/(2.f*nLambda) : 0.f;
    Wm[0]=w0m; Wc[0]=w0c;
    for (int i=1;i<NSIGMA;++i) { Wm[i]=wi; Wc[i]=wi; }
    return nLambda;
  }

  // ---------- quaternion (w,x,y,z), Hamilton, body->nav ----------
  static void quatNorm(float q[4]) {
    float n = sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if (n < 1e-12f) { q[0]=1.f; q[1]=q[2]=q[3]=0.f; return; }
    float inv = 1.f/n; for (int i=0;i<4;++i) q[i]*=inv;
  }
  static void quatMul(const float a[4], const float b[4], float o[4]) {
    float r[4];
    r[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    r[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    r[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    r[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
    o[0]=r[0]; o[1]=r[1]; o[2]=r[2]; o[3]=r[3];
  }
  // exact exp-map: q_new = normalize(q (x) dq(gyro-bg, dt)).
  static void propagateQuat(const float q[4], const float bg[3], const float gyro[3], float dt, float qout[4]) {
    float w[3] = { gyro[0]-bg[0], gyro[1]-bg[1], gyro[2]-bg[2] };
    float ang = norm3(w) * dt;
    float dq[4];
    if (ang > 1e-8f) {
      float half = ang*0.5f;
      float cosTerm = cosf(half);
      float sinTerm = dt * sinf(half) / ang;
      dq[0]=cosTerm; dq[1]=w[0]*sinTerm; dq[2]=w[1]*sinTerm; dq[3]=w[2]*sinTerm;
    } else {
      float half = ang*0.5f;
      float cosTerm = 1.f - half*half*0.5f;
      float sinTerm = dt*0.5f*(1.f - (ang*ang)/24.f);
      dq[0]=cosTerm; dq[1]=w[0]*sinTerm; dq[2]=w[1]*sinTerm; dq[3]=w[2]*sinTerm;
    }
    quatMul(q, dq, qout);
    quatNorm(qout);
  }
  static void eulerToQuat(float roll, float pitch, float yaw, float q[4]) {
    float cr=cosf(0.5f*roll),  sr=sinf(0.5f*roll);
    float cp=cosf(0.5f*pitch), sp=sinf(0.5f*pitch);
    float cy=cosf(0.5f*yaw),   sy=sinf(0.5f*yaw);
    q[0]=cr*cp*cy + sr*sp*sy;
    q[1]=sr*cp*cy - cr*sp*sy;
    q[2]=cr*sp*cy + sr*cp*sy;
    q[3]=cr*cp*sy - sr*sp*cy;
    quatNorm(q);
  }
  static void quatToEuler(const float q[4], float& roll, float& pitch, float& yaw) {
    float w=q[0],x=q[1],y=q[2],z=q[3];
    roll  = atan2f(2.f*(w*x + y*z), 1.f - 2.f*(x*x + y*y));
    float sp = 2.f*(w*y - z*x);
    sp = sp>1.f?1.f:(sp<-1.f?-1.f:sp);
    pitch = asinf(sp);
    yaw   = atan2f(2.f*(w*z + x*y), 1.f - 2.f*(y*y + z*z));
  }
};

// out-of-line static const definition (a float static needs a definition,
// not just an in-class initializer, on most Arduino toolchains)
const float USQUE_AHRS::IAE_THRESH_ = 5.99f;   // chi2(0.95, 2 dof)

#endif // USQUE_AHRS_H
