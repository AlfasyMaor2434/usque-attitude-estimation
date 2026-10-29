// ============================================================================
//  USQUE_AHRS_MagCalibration.ino
//
//  Hard-iron + (diagonal) soft-iron calibration for the ICM-20948's AK09916
//  magnetometer, matching exactly what USQUE_AHRS.ino's remapMag() applies:
//      m_cal[i] = (m_axisRemapped[i] - MAG_BIAS[i]) * MAG_SCALE[i]
//
//  UPDATED vs the old ESKF_AHRS_MagCalibration.ino: the hard-iron CENTER is
//  now a proper least-squares SPHERE fit over every sample collected
//  (streaming -- no per-sample storage, just 13 running sums), not a
//  min/max-per-axis box center. The min/max method sets each axis's bias
//  from only TWO samples (whichever happened to be the most extreme on that
//  axis) and is fully at the mercy of a single noisy outlier; the
//  least-squares fit uses every sample and is what the PC project's own
//  per-flight mag-bias correction uses (data_loader_USQUE.
//  fit_hard_iron_sphere -- see mag_bias_analysis.py / HANDOFF.md section 11
//  for why this matters: this project's own analysis found 10-40 deg,
//  non-random, per-flight-varying magnetic heading biases on real flight
//  data). Both results are printed so you can see whether they agree --
//  if they disagree by a lot, an outlier sample or poor coverage is likely.
//
//  The diagonal soft-iron SCALE still comes from a per-axis "radius": now
//  computed as the RMS distance from the least-squares CENTER (derived
//  algebraically from the same running sums once the center is known --
//  no second pass needed), instead of (max-min)/2, for the same
//  outlier-robustness reason.
//
//  Note what this does NOT do: a full 9-parameter (rotated-ellipsoid)
//  soft-iron fit. The diagonal (axis-aligned) approximation here is
//  standard and usually adequate; USQUE_AHRS.h's own dpsi_mag state also
//  absorbs whatever RESIDUAL heading bias survives an imperfect
//  calibration, so this calibration does not need to be perfect to get
//  good results in the field -- it needs to be reasonable.
//
//  HOW TO RUN: unchanged from before.
//    1. Upload this sketch, open the Serial Monitor at 115200 baud.
//    2. When it says "Rotate the board now...", pick it up (gently -- mind
//       the wires) and slowly move it through as many different
//       orientations as you can for the full calibration window:
//         - spin it flat (yaw) through a full circle
//         - tip the nose up and down through as wide a range as you can
//         - roll it side to side
//         - if you can, turn it upside down at some point
//       The goal is to visit as much of the surface of a sphere as
//       possible, not just spin flat -- a flat-only spin never exercises
//       the Z axis and leaves it uncalibrated.
//    3. When the countdown ends it prints the fitted MAG_BIAS/MAG_SCALE
//       (least-squares, RECOMMENDED) plus the old min/max numbers for
//       comparison, and a coverage warning per axis if that axis didn't
//       see enough range.
//    4. Copy the RECOMMENDED lines into USQUE_AHRS.ino, replacing the
//       existing MAG_BIAS[]/MAG_SCALE[] definitions.
// ============================================================================

#include <Wire.h>
#include "ICM_20948.h"

#define AD0_VAL 0       // match USQUE_AHRS.ino's setting
#define CAL_SECONDS 45  // how long to collect data for

ICM_20948_I2C imu;

// Same axis remap as USQUE_AHRS.ino's remapMag() -- BEFORE bias/scale, since
// that's the stage MAG_BIAS/MAG_SCALE are defined to correct.
static inline void remapMagRaw(float mx, float my, float mz, float o[3]) {
  o[0] =  my;
  o[1] =  mx;
  o[2] = -mz;
}

// ---- min/max box tracking (coverage diagnostic + comparison bias) ----
float mn[3] = {  1e9f,  1e9f,  1e9f };
float mx_[3] = { -1e9f, -1e9f, -1e9f };

// ---- streaming sums for the least-squares sphere fit ----
// Sphere: (x-cx)^2+(y-cy)^2+(z-cz)^2 = r^2  <=>  2cx.x+2cy.y+2cz.z+k = x^2+y^2+z^2
// (k = r^2-|c|^2), solved as a 4x4 linear least-squares system from these sums.
double Sx=0, Sy=0, Sz=0, Sxx=0, Syy=0, Szz=0, Sxy=0, Sxz=0, Syz=0;
double Sxr2=0, Syr2=0, Szr2=0, Sr2=0;

uint32_t nSamples = 0;
uint32_t tStart = 0;
bool done = false;

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {}
  Wire.begin();
  Wire.setClock(400000);

  bool ok = false;
  for (int i = 0; i < 5 && !ok; ++i) {
    imu.begin(Wire, AD0_VAL);
    ok = (imu.status == ICM_20948_Stat_Ok);
    if (!ok) { Serial.println(F("ICM-20948 not found, retrying...")); delay(400); }
  }
  if (!ok) { Serial.println(F("ICM-20948 init FAILED. Check wiring/address.")); while (1) delay(1000); }

  imu.swReset();  delay(50);
  imu.sleep(false);
  imu.lowPower(false);
  ICM_20948_fss_t fss; fss.a = gpm4; fss.g = dps500;
  imu.setFullScale((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), fss);
  ICM_20948_dlpcfg_t dlp; dlp.a = acc_d111bw4_n136bw; dlp.g = gyr_d119bw5_n154bw3;
  imu.setDLPFcfg((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), dlp);
  imu.enableDLPF((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), true);
  imu.startupMagnetometer();

  Serial.println(F("USQUE_AHRS mag calibration"));
  Serial.print(F("Collecting for ")); Serial.print(CAL_SECONDS); Serial.println(F(" seconds."));
  Serial.println(F("Rotate the board now: spin flat, tip nose up/down, roll side"));
  Serial.println(F("to side, and turn it upside-down at least once if you can."));
  tStart = millis();
}

void loop() {
  if (done) return;

  uint32_t elapsed = millis() - tStart;
  if (elapsed >= (uint32_t)CAL_SECONDS * 1000UL) {
    finish();
    done = true;
    return;
  }

  if (!imu.dataReady()) return;
  imu.getAGMT();
  float mxr = imu.magX(), myr = imu.magY(), mzr = imu.magZ();
  if (mxr == 0.f && myr == 0.f && mzr == 0.f) return;   // no mag data yet

  float m[3]; remapMagRaw(mxr, myr, mzr, m);
  for (int i = 0; i < 3; ++i) { if (m[i] < mn[i]) mn[i] = m[i]; if (m[i] > mx_[i]) mx_[i] = m[i]; }

  double x = m[0], y = m[1], z = m[2];
  double r2 = x*x + y*y + z*z;
  Sx += x; Sy += y; Sz += z;
  Sxx += x*x; Syy += y*y; Szz += z*z;
  Sxy += x*y; Sxz += x*z; Syz += y*z;
  Sxr2 += x*r2; Syr2 += y*r2; Szr2 += z*r2;
  Sr2 += r2;
  nSamples++;

  static uint32_t lastPrint = 0;
  if (millis() - lastPrint >= 200) {
    lastPrint = millis();
    uint32_t remain = (CAL_SECONDS*1000UL - elapsed) / 1000UL;
    Serial.print(F("magAx:")); Serial.print(m[0], 1);
    Serial.print(F(",magAy:")); Serial.print(m[1], 1);
    Serial.print(F(",magAz:")); Serial.println(m[2], 1);
    Serial.print(F("  t left: ")); Serial.print(remain);
    Serial.print(F("s   range so far (uT) X=")); Serial.print(mx_[0]-mn[0], 1);
    Serial.print(F(" Y=")); Serial.print(mx_[1]-mn[1], 1);
    Serial.print(F(" Z=")); Serial.println(mx_[2]-mn[2], 1);
  }
}

// Solve a 4x4 linear system via Gauss-Jordan with partial pivoting.
static bool solve4(double A[4][4], double b[4], double x[4]) {
  double a[4][5];
  for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) a[i][j] = A[i][j]; a[i][4] = b[i]; }
  for (int col = 0; col < 4; ++col) {
    int piv = col; double best = fabs(a[col][col]);
    for (int r = col+1; r < 4; ++r) { double v = fabs(a[r][col]); if (v > best) { best = v; piv = r; } }
    if (best < 1e-9) return false;
    if (piv != col) for (int j = 0; j < 5; ++j) { double t = a[col][j]; a[col][j] = a[piv][j]; a[piv][j] = t; }
    double inv = 1.0 / a[col][col];
    for (int j = 0; j < 5; ++j) a[col][j] *= inv;
    for (int r = 0; r < 4; ++r) {
      if (r == col) continue;
      double f = a[r][col];
      if (f == 0.0) continue;
      for (int j = 0; j < 5; ++j) a[r][j] -= f*a[col][j];
    }
  }
  for (int i = 0; i < 4; ++i) x[i] = a[i][4];
  return true;
}

void finish() {
  Serial.println();
  Serial.println(F("================ CALIBRATION RESULT ================"));
  Serial.print(F("samples: ")); Serial.println(nSamples);

  // ---- least-squares sphere fit (RECOMMENDED) ----
  double N = (double)nSamples;
  double A[4][4] = {
    { 4*Sxx, 4*Sxy, 4*Sxz, 2*Sx },
    { 4*Sxy, 4*Syy, 4*Syz, 2*Sy },
    { 4*Sxz, 4*Syz, 4*Szz, 2*Sz },
    { 2*Sx,  2*Sy,  2*Sz,  N    },
  };
  double b[4] = { 2*Sxr2, 2*Syr2, 2*Szr2, Sr2 };
  double sol[4];
  bool lsqOk = (N > 10) && solve4(A, b, sol);

  float biasLsq[3] = {0,0,0}, scaleLsq[3] = {1,1,1};
  if (lsqOk) {
    biasLsq[0] = (float)sol[0]; biasLsq[1] = (float)sol[1]; biasLsq[2] = (float)sol[2];
    // per-axis RMS distance from the LSQ center, derived from the already-
    // accumulated sums (no second pass): E[(x-c)^2] = E[x^2] - 2c*E[x] + c^2
    double varX = Sxx/N - 2*sol[0]*(Sx/N) + sol[0]*sol[0];
    double varY = Syy/N - 2*sol[1]*(Sy/N) + sol[1]*sol[1];
    double varZ = Szz/N - 2*sol[2]*(Sz/N) + sol[2]*sol[2];
    float rmsR[3] = { (float)sqrt(fmax(varX,1e-6)), (float)sqrt(fmax(varY,1e-6)), (float)sqrt(fmax(varZ,1e-6)) };
    float avgR = (rmsR[0]+rmsR[1]+rmsR[2]) / 3.0f;
    for (int i = 0; i < 3; ++i) scaleLsq[i] = (rmsR[i] > 1e-3f) ? (avgR/rmsR[i]) : 1.0f;
    Serial.print(F("fitted radius (LSQ): ")); Serial.println(sqrt(fmax(sol[3]+sol[0]*sol[0]+sol[1]*sol[1]+sol[2]*sol[2], 0.0)), 2);
  } else {
    Serial.println(F("LSQ sphere fit FAILED (too few samples or degenerate coverage) -- using min/max only."));
  }

  // ---- min/max box (comparison / fallback) ----
  float biasMinMax[3], radiusMinMax[3];
  for (int i = 0; i < 3; ++i) {
    biasMinMax[i]   = 0.5f * (mx_[i] + mn[i]);
    radiusMinMax[i] = 0.5f * (mx_[i] - mn[i]);
  }
  float avgRadiusMinMax = (radiusMinMax[0] + radiusMinMax[1] + radiusMinMax[2]) / 3.0f;
  float scaleMinMax[3];
  for (int i = 0; i < 3; ++i) scaleMinMax[i] = (radiusMinMax[i] > 1e-3f) ? (avgRadiusMinMax / radiusMinMax[i]) : 1.0f;

  const char* axName[3] = {"X","Y","Z"};
  bool lowCoverage = false;
  for (int i = 0; i < 3; ++i) {
    float range = mx_[i] - mn[i];
    Serial.print(F("  axis ")); Serial.print(axName[i]);
    Serial.print(F(": min=")); Serial.print(mn[i], 1);
    Serial.print(F(" max=")); Serial.print(mx_[i], 1);
    Serial.print(F(" range=")); Serial.print(range, 1); Serial.print(F(" uT"));
    if (range < 25.0f) { Serial.print(F("   <-- LOW: rotate this axis through a bigger swing and re-run")); lowCoverage = true; }
    Serial.println();
  }

  Serial.println();
  if (lsqOk) {
    Serial.println(F("RECOMMENDED -- paste into USQUE_AHRS.ino (replacing the existing lines):"));
    Serial.print(F("static const float MAG_BIAS[3]  = { "));
    Serial.print(biasLsq[0], 3); Serial.print(F("f, "));
    Serial.print(biasLsq[1], 3); Serial.print(F("f, "));
    Serial.print(biasLsq[2], 3); Serial.println(F("f };"));
    Serial.print(F("static const float MAG_SCALE[3] = { "));
    Serial.print(scaleLsq[0], 4); Serial.print(F("f, "));
    Serial.print(scaleLsq[1], 4); Serial.print(F("f, "));
    Serial.print(scaleLsq[2], 4); Serial.println(F("f };"));
    Serial.println();
    Serial.println(F("For comparison, the old min/max-box method gave:"));
    Serial.print(F("  MAG_BIAS  (min/max) = { "));
    Serial.print(biasMinMax[0], 3); Serial.print(F(", "));
    Serial.print(biasMinMax[1], 3); Serial.print(F(", "));
    Serial.print(biasMinMax[2], 3); Serial.println(F(" }"));
    Serial.print(F("  MAG_SCALE (min/max) = { "));
    Serial.print(scaleMinMax[0], 4); Serial.print(F(", "));
    Serial.print(scaleMinMax[1], 4); Serial.print(F(", "));
    Serial.print(scaleMinMax[2], 4); Serial.println(F(" }"));
    Serial.println(F("  (if these disagree with the recommended numbers by a lot, re-check"));
    Serial.println(F("   coverage above -- an outlier sample or a poorly-covered axis is likely)"));
  } else {
    Serial.println(F("Paste into USQUE_AHRS.ino (replacing the existing lines):"));
    Serial.print(F("static const float MAG_BIAS[3]  = { "));
    Serial.print(biasMinMax[0], 3); Serial.print(F("f, "));
    Serial.print(biasMinMax[1], 3); Serial.print(F("f, "));
    Serial.print(biasMinMax[2], 3); Serial.println(F("f };"));
    Serial.print(F("static const float MAG_SCALE[3] = { "));
    Serial.print(scaleMinMax[0], 4); Serial.print(F("f, "));
    Serial.print(scaleMinMax[1], 4); Serial.print(F("f, "));
    Serial.print(scaleMinMax[2], 4); Serial.println(F("f };"));
  }
  if (lowCoverage) {
    Serial.println();
    Serial.println(F(">>> At least one axis had low range -- its bias/scale is unreliable."));
    Serial.println(F(">>> Re-run and make sure to tilt/roll/invert the board, not just spin it flat."));
  }
  Serial.println(F("======================================================"));
}
