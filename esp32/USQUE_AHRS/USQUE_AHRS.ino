// ============================================================================
//  USQUE_AHRS.ino  --  ESP32 + ICM-20948 attitude/heading demo
//
//  Runs USQUE_AHRS.h, the faithful embedded port of the PC project's
//  usque.py (see that header's own comment for what "faithful" means and
//  what is deliberately different for embedded reasons). REPLACES
//  UKF_AHRS.ino (kept, untouched, for reference -- a different, older
//  filter design, see UKF_AHRS.h's header comment).
//
//  Board:   any ESP32 dev board
//  Sensor:  ICM-20948 breakout on I2C (SparkFun / generic).
//  Library: "SparkFun ICM-20948 Arduino Library"  (Library Manager)
//
//  Wiring (I2C):  VIN/3V3->3V3, GND->GND, SDA->GPIO21, SCL->GPIO22,
//                 AD0->GND (addr 0x68; AD0->3V3 => 0x69, set AD0_VAL 1)
//
//  What it does, matching main_usque.py's pipeline as closely as a REAL-TIME
//  filter can:
//    - reads accel/gyro at the IMU's native rate (~225 Hz), magnetometer at
//      its own ~100 Hz rate
//    - predict() every IMU sample (RAW gyro -- main_usque.py's LPF_MODE=
//      'accel_only' filters accel only, not gyro)
//    - a CAUSAL 4th-order Butterworth low-pass (15 Hz cutoff) on accel only,
//      the real-time analogue of main_usque.py's offline zero-phase
//      filtfilt() -- same passband, but NOT zero-phase (adds a few ms of
//      group delay filtfilt does not; this is unavoidable in real time)
//    - update() at a DECIMATED ~30 Hz rate (matches main_usque.py's
//      FUSION_MODE='decimate' @ FUSION_RATE_HZ=30 -- fusing every IMU
//      sample double/triple-counts a low-passed, slowly-varying signal as
//      independent information and makes the filter's own uncertainty
//      collapse far below its true error)
//    - a causal vibration/disturbance proxy (accel high-frequency residual
//      relative to a rolling baseline) feeds the VIBRATION adaptive branch
//    - prints roll/pitch/yaw, quaternion, attitude+dpsi_mag 1-sigma,
//      biases, and live adaptive-R/mag-gate diagnostics
//
//  FIRST RUN -- three things to check:
//    1. AXIS / SIGN. Board flat and still: "accel FRD: x y z" must read
//       approx 0.0 0.0 -9.81. If not, edit remapAccel()/remapGyro()/
//       remapMag() below.
//    2. MAG CALIBRATION is built in (no separate sketch). At boot the sketch
//       (a) checks the magnetometer is delivering real data (board still:
//       readings must repeat to within a few uT), then (b) if no calibration
//       is stored in flash -- or you hold the BOOT button / send "C" during
//       the 3 s boot window -- asks you to rotate the board through all
//       orientations for MAG_CAL_SECONDS, fits offset + per-axis scale,
//       checks the fit and stores it in flash. Send "C" at any time to
//       redo it (the board restarts into the calibration). When the health
//       check fails, a register-level DIAGNOSTIC (magDiagnostic, written
//       against the ICM-20948 datasheet) prints why. If the health
//       check or the fit fails, the magnetometer is switched OFF and the
//       filter runs on gyro + accel: roll/pitch are unaffected, heading is
//       then RELATIVE to the boot direction.
//    3. LOCAL MAGNETIC REFERENCE. Set MAG_INCLINATION_DEG/MAG_DECLINATION_DEG
//       for your site (a WMM lookup for your lat/lon/date). Getting this
//       wrong by a lot is now RECOVERABLE (dpsi_mag will converge to correct
//       it within roughly one MAG_BIAS_TAU_S, ~5 min at the default), but
//       accuracy is best from the start if it's right.
// ============================================================================

#include <Wire.h>
#include <Preferences.h>    // ESP32 flash key/value store (magnetometer calibration)
#include <algorithm>
#include "ICM_20948.h"      // SparkFun ICM-20948 library
#include "USQUE_AHRS.h"

#define AD0_VAL   0          // 0 if AD0 pin -> GND (addr 0x68), 1 if -> 3V3
ICM_20948_I2C imu;
USQUE_AHRS    ahrs;

// ---- magnetometer calibration (fitted at boot, stored in flash) ----
//  m_cal[i] = (m_raw[i] - magBias[i]) * magScale[i]      (units: uT, FRD axes)
static float magBias[3]  = { 0.f, 0.f, 0.f };
static float magScale[3] = { 1.f, 1.f, 1.f };
static bool  useMag = false;          // false: health check or calibration failed -> gyro + accel only
Preferences prefs;

const uint32_t MAG_CAL_SECONDS       = 30;     // rotation time for the calibration
const float    MAG_CAL_MAX_RESIDUAL  = 0.10f;  // reject a fit that leaves |mag| varying more than this
const float    MAG_HEALTH_MAX_STD_UT = 5.0f;   // board still: per-axis std above this = not real field data
const float    MAG_VALID_MAX_UT      = 300.0f; // a single raw sample above this is discarded (Earth ~25-65 uT)
const int      BOOT_BUTTON_PIN       = 0;      // ESP32 dev-board BOOT button (LOW when pressed)

// ---- local magnetic reference -- SET FOR YOUR SITE (WMM lookup) ----
// Inclination is only the starting model: with MAG_REF_FROM_BOOT the filter takes the
// inclination and |mag| baseline measured in the boot still window (indoors the field is
// distorted -- 11..36 deg measured at the desk instead of 49 -- and a model value there makes
// the mag gate reject every sample).
static const bool  MAG_REF_FROM_BOOT   = true;
// Magnetometer measurement variance. 0.05 was tuned with the dpsi_mag state on; with it off the
// filter otherwise follows every few-degree field change within seconds (phone tests).
static const float MAG_NOISE_VAR       = 0.3f;
static const float MAG_INCLINATION_DEG = 48.986f;  // dip angle, positive DOWN
static const float MAG_DECLINATION_DEG = 5.038f;     // degrees EAST of true north

// dpsi_mag (magnetic-heading-bias state). OFF: with the magnetometer
// calibrated (USQUE_AHRS_MagCalibration) there is no slow heading bias for
// it to learn. When on, it absorbs transient errors during motion and then
// relaxes back toward 0 at psiMag/mag_bias_tau_s (300 s) once still --
// and since the magnetometer only observes (yaw - psiMag), yaw is dragged
// along: a slow stationary yaw drift (measured with the phone test:
// -1.08 deg yaw vs -1.13 deg psiMag over 17 s, predicted 1.07 deg).
static const bool USE_MAG_BIAS_STATE = false;

// Gyro-bias seed. true (normal): the bias measured in the boot still window starts the filter,
// so the estimate is right from the first second. false: the filter starts from ZERO bias with a
// loose prior and has to find the bias itself -- a demonstration of the bias states converging
// (x and y through the accelerometer; z needs the magnetometer and stays where it started
// without one). Sending Z (or the viewer's button) restarts the board for ONE run without the
// seed, whatever this constant says.
static const bool SEED_GYRO_BIAS = true;
static bool seedGyroBias = SEED_GYRO_BIAS;

// Accelerometer low-pass before the update. true: the causal 15 Hz Butterworth below. false: the
// update uses the raw accelerometer sample. Offline study on the indoor flights
// (final_usque_results/lpf_study): a REAL-TIME 15 Hz filter scored 74.5 % of estimates under
// 1 deg, no filter 80.4 % (both with accel_noise_var 1e-2) -- the filter's delay costs more than
// its smoothing gains. This constant is the default; sending L (or the viewer's button) flips
// the choice, stores it in flash and restarts, so the two can be compared on the same board.
// The vibration proxy keeps using the low-passed signal either way.
static const bool USE_ACCEL_LPF = true;
static bool useAccelLpf = USE_ACCEL_LPF;

// ---- accelerometer scale correction (uniform) ----
//  See USQUE_AHRS_AccelCalibration.ino: rest the board flat & still, read
//  the uncorrected |accel|, set ACCEL_SCALE_CORR = 9.81 / measured. A
//  per-axis 6-position calibration is available in that sketch too but is
//  NOT what this filter applies by default -- see its header comment for why.
static const float ACCEL_SCALE_CORR = 1.0535f;

// ------------------------------------------------------------------ axis remap
static inline void remapAccel(float ax, float ay, float az, float o[3]) {
  const float MG2MS2 = 9.80665f / 1000.0f;
  o[0] = -(ax * MG2MS2) * ACCEL_SCALE_CORR;
  o[1] = -(ay * MG2MS2) * ACCEL_SCALE_CORR;
  o[2] = -(az * MG2MS2) * ACCEL_SCALE_CORR;
}
static inline void remapGyro(float gx, float gy, float gz, float o[3]) {
  const float D2R = 0.01745329252f;
  o[0] = gx * D2R; o[1] = gy * D2R; o[2] = gz * D2R;
}
static inline void magAxes(float mx, float my, float mz, float o[3]) {
  // AK09916 mounted rotated vs the accel/gyro die (see datasheet). Uncalibrated, FRD axes.
  o[0] = my; o[1] = mx; o[2] = -mz;
}
static inline void remapMag(float mx, float my, float mz, float o[3]) {
  float r[3]; magAxes(mx, my, mz, r);
  for (int i = 0; i < 3; ++i) o[i] = (r[i] - magBias[i]) * magScale[i];
}
// One raw sample is usable if the AK09916 did not flag a sensor overflow (ST2.HOFL) and it is
// not absurdly large.
static inline bool magSampleValid(float mx, float my, float mz) {
  if (imu.agmt.magStat2 & 0x08) return false;
  return fabsf(mx) < MAG_VALID_MAX_UT && fabsf(my) < MAG_VALID_MAX_UT && fabsf(mz) < MAG_VALID_MAX_UT;
}

// ------------------------------------------------------------------ boot-time magnetometer helpers
// Blocks until a NEW magnetometer sample arrives (or timeoutMs passes). raw = uncalibrated FRD uT.
// valid = false when the sample fails magSampleValid(). gyroNorm = |gyro| of the same read (rad/s).
static bool waitFreshMag(float raw[3], bool &valid, float &gyroNorm, uint32_t timeoutMs) {
  static float last[3] = { 1e30f, 1e30f, 1e30f };
  uint32_t t0 = millis();
  while (millis() - t0 < timeoutMs) {
    if (!imu.dataReady()) { delay(1); continue; }
    imu.getAGMT();
    float mx = imu.magX(), my = imu.magY(), mz = imu.magZ();
    if (mx == last[0] && my == last[1] && mz == last[2]) continue;
    last[0] = mx; last[1] = my; last[2] = mz;
    float g[3]; remapGyro(imu.gyrX(), imu.gyrY(), imu.gyrZ(), g);
    gyroNorm = sqrtf(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);
    valid = magSampleValid(mx, my, mz);
    magAxes(mx, my, mz, raw);
    return true;
  }
  return false;
}

// Board still for ~2 s: a working magnetometer repeats its reading to within a few uT. Readings
// that jump by hundreds of uT between samples are not field data (bad chip / bad I2C link), and
// no calibration can fix that.
static bool magHealthCheck() {
  for (int attempt = 0; attempt < 3; ++attempt) {
    double sum[3] = {0,0,0}, sum2[3] = {0,0,0};
    int n = 0, nBad = 0, nTotal = 0;
    uint32_t t0 = millis();
    while (millis() - t0 < 2000) {
      float raw[3]; bool valid; float wn;
      if (!waitFreshMag(raw, valid, wn, 200)) continue;
      if (wn > 0.10f) { t0 = millis(); n = 0; nBad = 0; nTotal = 0;      // moved: start over
                        for (int i = 0; i < 3; ++i) { sum[i] = 0; sum2[i] = 0; } continue; }
      ++nTotal;
      if (!valid) { ++nBad; continue; }
      for (int i = 0; i < 3; ++i) { sum[i] += raw[i]; sum2[i] += (double)raw[i]*raw[i]; }
      ++n;
    }
    float sd = 0.f, mean[3] = {0,0,0};
    if (n > 1) for (int i = 0; i < 3; ++i) {
      mean[i] = sum[i] / n;
      float v = sqrtf(fmaxf(0.f, (float)(sum2[i]/n - (double)mean[i]*mean[i])));
      if (v > sd) sd = v;
    }
    float norm = sqrtf(mean[0]*mean[0] + mean[1]*mean[1] + mean[2]*mean[2]);
    Serial.print(F("mag health: ")); Serial.print(nTotal); Serial.print(F(" samples in 2 s, "));
    Serial.print(nBad); Serial.print(F(" invalid, per-axis std (max) ")); Serial.print(sd, 2);
    Serial.print(F(" uT, mean |raw| ")); Serial.print(norm, 1);
    Serial.print(F(" uT, AK09916 WIA2=0x")); Serial.println(imu.readMag(AK09916_REG_WIA2), HEX);
    bool ok = n >= 30 && nBad * 10 <= nTotal && sd < MAG_HEALTH_MAX_STD_UT && norm > 5.f;
    if (ok) return true;
    Serial.println(F("mag health: FAILED -- restarting the magnetometer and retrying..."));
    imu.startupMagnetometer();
    delay(200);
  }
  return false;
}

// 4x4 linear solve (Gaussian elimination with partial pivoting). Returns false if singular.
static bool solve4(double A[4][5]) {
  for (int c = 0; c < 4; ++c) {
    int piv = c;
    for (int r = c + 1; r < 4; ++r) if (fabs(A[r][c]) > fabs(A[piv][c])) piv = r;
    if (fabs(A[piv][c]) < 1e-9) return false;
    if (piv != c) for (int k = 0; k < 5; ++k) { double t = A[c][k]; A[c][k] = A[piv][k]; A[piv][k] = t; }
    for (int r = 0; r < 4; ++r) if (r != c) {
      double f = A[r][c] / A[c][c];
      for (int k = c; k < 5; ++k) A[r][k] -= f * A[c][k];
    }
  }
  return true;
}

// Rotate-the-board calibration: hard iron = centre of the least-squares sphere through the
// samples; per-axis scale from the 2nd..98th percentile half-ranges (only when every octant was
// visited, otherwise scale = 1). Accepted only if the corrected |mag| is constant to within
// MAG_CAL_MAX_RESIDUAL. Stores the result in flash.
static const int CAL_MAX = 900;
static float calBuf[CAL_MAX][3];
static bool runMagCalibration() {
  Serial.print(F("\nMAG CALIBRATION: slowly rotate the board through EVERY orientation for "));
  Serial.print(MAG_CAL_SECONDS);
  Serial.println(F(" s -- each face up and down, each edge down, in a slow tumbling figure-8, away from metal."));
  for (int c = 3; c > 0; --c) { Serial.print(F("starting in ")); Serial.println(c); delay(1000); }
  Serial.println(F("ROTATE NOW."));
  int n = 0, nBad = 0;
  uint32_t t0 = millis(), tNext = 0, tProg = millis();
  while (millis() - t0 < MAG_CAL_SECONDS * 1000UL) {
    float raw[3]; bool valid; float wn;
    if (!waitFreshMag(raw, valid, wn, 200)) continue;
    if (!valid) { ++nBad; continue; }
    if (millis() >= tNext && n < CAL_MAX) {             // ~30 Hz is plenty and bounds the buffer
      tNext = millis() + (MAG_CAL_SECONDS * 1000UL) / CAL_MAX;
      calBuf[n][0] = raw[0]; calBuf[n][1] = raw[1]; calBuf[n][2] = raw[2]; ++n;
    }
    if (millis() - tProg > 5000) {
      tProg = millis();
      Serial.print(F("  ...")); Serial.print((MAG_CAL_SECONDS * 1000UL - (millis() - t0)) / 1000);
      Serial.print(F(" s left, ")); Serial.print(n); Serial.println(F(" samples"));
    }
  }
  Serial.print(F("rotation done: ")); Serial.print(n); Serial.print(F(" samples (")); Serial.print(nBad);
  Serial.println(F(" invalid discarded). Fitting..."));
  if (n < 200) { Serial.println(F("MAG CALIBRATION REJECTED: too few samples.")); return false; }

  // sphere:  |m|^2 = 2 c.m + d
  double A[4][5] = {{0}};
  for (int k = 0; k < n; ++k) {
    double row[4] = { 2.0*calBuf[k][0], 2.0*calBuf[k][1], 2.0*calBuf[k][2], 1.0 };
    double b = (double)calBuf[k][0]*calBuf[k][0] + (double)calBuf[k][1]*calBuf[k][1] + (double)calBuf[k][2]*calBuf[k][2];
    for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) A[i][j] += row[i]*row[j]; A[i][4] += row[i]*b; }
  }
  if (!solve4(A)) { Serial.println(F("MAG CALIBRATION REJECTED: degenerate data (board not rotated?).")); return false; }
  float c[3] = { (float)(A[0][4]/A[0][0]), (float)(A[1][4]/A[1][1]), (float)(A[2][4]/A[2][2]) };

  // octant coverage around the centre
  int oct[8] = {0};
  for (int k = 0; k < n; ++k)
    ++oct[(calBuf[k][0] > c[0]) + 2*(calBuf[k][1] > c[1]) + 4*(calBuf[k][2] > c[2])];
  int covered = 0; for (int i = 0; i < 8; ++i) if (oct[i] >= 5) ++covered;

  // per-axis scale from robust half-ranges
  float sc[3] = { 1.f, 1.f, 1.f };
  if (covered == 8) {
    static float tmp[CAL_MAX];
    float hr[3];
    for (int i = 0; i < 3; ++i) {
      for (int k = 0; k < n; ++k) tmp[k] = calBuf[k][i];
      std::sort(tmp, tmp + n);
      hr[i] = 0.5f * (tmp[(int)(0.98f*(n-1))] - tmp[(int)(0.02f*(n-1))]);
    }
    float hm = (hr[0] + hr[1] + hr[2]) / 3.f;
    if (hr[0] > 1.f && hr[1] > 1.f && hr[2] > 1.f)
      for (int i = 0; i < 3; ++i) sc[i] = hm / hr[i];
  }

  // residual: how constant is |mag| after the correction (try with and without the scale)
  float best = 1e9f, radius = 0.f; bool bestScaled = false;
  for (int pass = 0; pass < 2; ++pass) {
    double sm = 0, sm2 = 0;
    for (int k = 0; k < n; ++k) {
      float v[3]; for (int i = 0; i < 3; ++i) v[i] = (calBuf[k][i] - c[i]) * (pass ? sc[i] : 1.f);
      float r = sqrtf(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
      sm += r; sm2 += (double)r*r;
    }
    float mean = sm / n, frac = sqrtf(fmaxf(0.f, (float)(sm2/n - (double)mean*mean))) / fmaxf(mean, 1e-6f);
    if (frac < best) { best = frac; radius = mean; bestScaled = (pass == 1); }
  }
  if (!bestScaled) sc[0] = sc[1] = sc[2] = 1.f;

  Serial.print(F("fit: offset ")); Serial.print(c[0], 1); Serial.print(' '); Serial.print(c[1], 1); Serial.print(' ');
  Serial.print(c[2], 1); Serial.print(F(" uT, scale ")); Serial.print(sc[0], 3); Serial.print(' ');
  Serial.print(sc[1], 3); Serial.print(' '); Serial.print(sc[2], 3);
  Serial.print(F(", field ")); Serial.print(radius, 1); Serial.print(F(" uT (Earth 25-65), |mag| varies "));
  Serial.print(100.f*best, 1); Serial.print(F(" %, octants ")); Serial.print(covered); Serial.println(F("/8"));

  if (best > MAG_CAL_MAX_RESIDUAL || covered < 6 || radius < 15.f || radius > 90.f) {
    Serial.println(F("MAG CALIBRATION REJECTED (needs: variation <= 10 %, >= 6/8 octants, field 15-90 uT)."));
    return false;
  }
  for (int i = 0; i < 3; ++i) { magBias[i] = c[i]; magScale[i] = sc[i]; }
  prefs.begin("usque", false);
  prefs.putBytes("magBias", magBias, sizeof(magBias));
  prefs.putBytes("magScale", magScale, sizeof(magScale));
  prefs.putBool("magOk", true);
  prefs.end();
  Serial.println(F("MAG CALIBRATION ACCEPTED and stored in flash."));
  return true;
}

// ------------------------------------------------------------------ magnetometer diagnostic
// Runs when the health check fails (or send D in the boot window). Every check refers to the
// ICM-20948 datasheet DS-000189 rev 1.3:
//   sec. 13.1  WIA2 (01h) is fixed at 09h
//   sec. 13.2  ST1 (10h): bit0 DRDY, bit1 DOR, bits 7..2 always 0
//   sec. 13.3  HXL..HZH (11h..16h): two's complement, little endian, range -32752..32752,
//              0.15 uT per count (max 4912 uT)
//   sec. 13.4  ST2 (18h): bit3 HOFL (sensor overflow -> data not correct), bits 7 and 2..0
//              always 0, bits 6..4 reserved; ST2 must be read to end every data read
//   sec. 13.5  CNTL2 (31h): 01000b = continuous mode 4, 10000b = self-test, 00000b = power-down
//   sec. 8.11  I2C_MST_STATUS (bank 0, 17h): bit0 I2C_SLV0_NACK, bit4 I2C_SLV4_NACK, bit5 I2C_LOST_ARB
// and to AKM's AK09916 datasheet (015007392-E-02):
//   sec. 9.4.3.2  read sequence: poll DRDY, read HXL..HZH, then ST2 (required, it ends the read)
//   sec. 9.4.3.6  overflow: valid data needs |X|+|Y|+|Z| < 4912 uT; beyond it HOFL = 1
//   sec. 9.4.4.1  self-test sequence: power-down, self-test mode, poll DRDY, read HXL..HZH
//   sec. 9.4.4.2  self-test judgment: -200 <= HX <= 200, -200 <= HY <= 200, -1000 <= HZ <= -200
//   sec. 9.3      at least 100 us in power-down before another mode is set
// The magnetometer is a separate die read by the ICM's own I2C master (peripheral 0 copies
// ST1..ST2 into EXT_SLV_SENS_DATA at the gyro rate). The diagnostic compares that AUTOMATIC path
// with DIRECT register reads of the same chip, which tells a read-path problem from a bad sensor.
struct MagStats {
  int n = 0, hofl = 0, dor = 0, drdy = 0, st1Bad = 0, st2Bad = 0, range = 0, sumOver = 0;
  double sum[3] = {0,0,0}, sum2[3] = {0,0,0};
  void add(uint8_t st1, int16_t x, int16_t y, int16_t z, uint8_t st2) {
    ++n;
    if (st1 & 0x01) ++drdy;
    if (st1 & 0x02) ++dor;
    if (st1 & 0xFC) ++st1Bad;                 // bits that the datasheet fixes at 0
    if (st2 & 0x08) ++hofl;
    if (st2 & 0x87) ++st2Bad;                 // bits that the datasheet fixes at 0
    int16_t v[3] = { x, y, z };
    bool out = false;
    for (int i = 0; i < 3; ++i) {
      if (v[i] > 32752 || v[i] < -32752) out = true;
      float u = v[i] * 0.15f; sum[i] += u; sum2[i] += (double)u * u;
    }
    if (out) ++range;
    // AKM 9.4.3.6: |X|+|Y|+|Z| must stay below 4912 uT; at or above it the chip must flag HOFL
    if ((fabsf(x*0.15f) + fabsf(y*0.15f) + fabsf(z*0.15f)) >= 4912.f && !(st2 & 0x08)) ++sumOver;
  }
  float maxStd() const {
    float m = 0.f;
    for (int i = 0; i < 3 && n > 1; ++i) {
      double mu = sum[i] / n; float sd = sqrtf(fmaxf(0.f, (float)(sum2[i] / n - mu * mu)));
      if (sd > m) m = sd;
    }
    return m;
  }
  float meanNorm() const {
    if (n == 0) return 0.f;
    float a = sum[0] / n, b = sum[1] / n, c = sum[2] / n;
    return sqrtf(a*a + b*b + c*c);
  }
  void print(const __FlashStringHelper *label) const {
    Serial.print(label); Serial.print(F(": ")); Serial.print(n); Serial.print(F(" samples | DRDY set ")); Serial.print(drdy);
    Serial.print(F(", DOR set ")); Serial.print(dor); Serial.print(F(", HOFL (overflow) set ")); Serial.print(hofl);
    Serial.print(F(" | impossible bytes: ST1 ")); Serial.print(st1Bad); Serial.print(F(", ST2 ")); Serial.print(st2Bad);
    Serial.print(F(", data out of range ")); Serial.print(range);
    Serial.print(F(", |X|+|Y|+|Z| >= 4912 uT without HOFL ")); Serial.print(sumOver);
    Serial.print(F(" | per-axis std (max) ")); Serial.print(maxStd(), 1); Serial.print(F(" uT, |mean field| "));
    Serial.print(meanNorm(), 1); Serial.println(F(" uT"));
  }
};
static void printMagSample(const __FlashStringHelper *label, uint8_t st1, int16_t x, int16_t y, int16_t z, uint8_t st2) {
  Serial.print(label); Serial.print(F(" ST1=0x")); Serial.print(st1, HEX);
  Serial.print(F(" X=")); Serial.print(x); Serial.print(F(" Y=")); Serial.print(y); Serial.print(F(" Z=")); Serial.print(z);
  Serial.print(F(" counts (")); Serial.print(x*0.15f, 1); Serial.print(' '); Serial.print(y*0.15f, 1); Serial.print(' ');
  Serial.print(z*0.15f, 1); Serial.print(F(" uT) ST2=0x")); Serial.println(st2, HEX);
}
// One coherent sample by direct register reads: wait for DRDY, read HXL..HZH, then ST2 (which
// ends the read, sec. 13.4). Returns false on timeout.
static bool magDirectRead(uint8_t &st1, int16_t &x, int16_t &y, int16_t &z, uint8_t &st2, uint32_t timeoutMs) {
  uint32_t t0 = millis();
  do { st1 = imu.readMag(AK09916_REG_ST1); } while (!(st1 & 0x01) && millis() - t0 < timeoutMs);
  if (!(st1 & 0x01)) return false;
  uint8_t b[6];
  b[0] = imu.readMag(AK09916_REG_HXL); b[1] = imu.readMag(AK09916_REG_HXH);
  b[2] = imu.readMag(AK09916_REG_HYL); b[3] = imu.readMag(AK09916_REG_HYH);
  b[4] = imu.readMag(AK09916_REG_HZL); b[5] = imu.readMag(AK09916_REG_HZH);
  st2 = imu.readMag(AK09916_REG_ST2);
  x = (int16_t)((b[1] << 8) | b[0]); y = (int16_t)((b[3] << 8) | b[2]); z = (int16_t)((b[5] << 8) | b[4]);
  return true;
}
static void magDiagnostic() {
  Serial.println(F("---- MAGNETOMETER DIAGNOSTIC (keep the board still) ----"));
  // A. identity and mode
  uint8_t wia2 = imu.readMag(AK09916_REG_WIA2), cntl2 = imu.readMag(AK09916_REG_CNTL2);
  Serial.print(F("A. WIA2=0x")); Serial.print(wia2, HEX); Serial.print(wia2 == 0x09 ? F(" (ok, datasheet 09h)") : F(" (WRONG, datasheet 09h)"));
  Serial.print(F("  CNTL2=0x")); Serial.print(cntl2, HEX);
  Serial.println(cntl2 == 0x08 ? F(" (ok, continuous mode 4)") : F(" (expected 08h = continuous mode 4)"));
  // B. the ICM's I2C master
  uint8_t mst = 0; imu.setBank(0); imu.read(AGB0_REG_I2C_MST_STATUS, &mst, 1);
  Serial.print(F("B. I2C_MST_STATUS=0x")); Serial.print(mst, HEX);
  Serial.print(F("  SLV0_NACK=")); Serial.print(mst & 0x01); Serial.print(F(" SLV4_NACK=")); Serial.print((mst >> 4) & 1);
  Serial.print(F(" LOST_ARB=")); Serial.println((mst >> 5) & 1);
  // C. automatic path (what the filter was using)
  MagStats autoS; int shown = 0; int16_t lx = 0, ly = 0, lz = 0;
  for (uint32_t t0 = millis(); millis() - t0 < 2000; ) {
    if (!imu.dataReady()) { delay(1); continue; }
    imu.getAGMT();
    int16_t x = imu.agmt.mag.axes.x, y = imu.agmt.mag.axes.y, z = imu.agmt.mag.axes.z;
    if (autoS.n > 0 && x == lx && y == ly && z == lz) continue;      // same sample repeated
    lx = x; ly = y; lz = z;
    autoS.add(imu.agmt.magStat1, x, y, z, imu.agmt.magStat2);
    if (shown < 5) { printMagSample(F("   auto  "), imu.agmt.magStat1, x, y, z, imu.agmt.magStat2); ++shown; }
  }
  autoS.print(F("C. AUTOMATIC path"));
  // D. direct path: stop the automatic read, then read the registers one by one
  imu.i2cControllerConfigurePeripheral(0, MAG_AK09916_I2C_ADDR, AK09916_REG_ST1, 9, true, false, false, false, false);
  delay(30);
  { uint8_t d = imu.readMag(AK09916_REG_ST2); (void)d; }             // release any read left open
  MagStats dirS; int timeouts = 0;
  for (int k = 0; k < 25; ++k) {
    uint8_t st1, st2; int16_t x, y, z;
    if (!magDirectRead(st1, x, y, z, st2, 100)) { ++timeouts; continue; }
    dirS.add(st1, x, y, z, st2);
    if (k < 5) printMagSample(F("   direct"), st1, x, y, z, st2);
  }
  dirS.print(F("D. DIRECT path"));
  if (timeouts) { Serial.print(F("   DRDY never came in ")); Serial.print(timeouts); Serial.println(F(" of 25 tries")); }
  // E. self-test mode (sec. 13.5): the chip measures its own internal test field
  uint8_t m = AK09916_mode_power_down; imu.writeMag(AK09916_REG_CNTL2, &m); delay(20);
  m = AK09916_mode_self_test; imu.writeMag(AK09916_REG_CNTL2, &m);
  { uint8_t st1, st2; int16_t x, y, z;
    if (magDirectRead(st1, x, y, z, st2, 300)) {
      printMagSample(F("E. SELF-TEST"), st1, x, y, z, st2);
      // judgment window: AKM AK09916 datasheet sec. 9.4.4.2
      bool pass = x >= -200 && x <= 200 && y >= -200 && y <= 200 && z >= -1000 && z <= -200 && !(st2 & 0x08);
      Serial.println(pass ? F("   self-test PASSED (AK09916 datasheet 9.4.4.2: |HX|,|HY| <= 200, -1000 <= HZ <= -200)")
                          : F("   self-test FAILED (AK09916 datasheet 9.4.4.2: |HX|,|HY| <= 200, -1000 <= HZ <= -200)"));
    } else Serial.println(F("E. SELF-TEST: no data ready within 300 ms"));
  }
  m = AK09916_mode_power_down; imu.writeMag(AK09916_REG_CNTL2, &m); delay(20);
  m = AK09916_mode_cont_100hz; imu.writeMag(AK09916_REG_CNTL2, &m); delay(20);
  imu.i2cControllerConfigurePeripheral(0, MAG_AK09916_I2C_ADDR, AK09916_REG_ST1, 9, true, true, false, false, false);
  // F. verdict
  bool directOk = dirS.n >= 10 && dirS.hofl == 0 && dirS.st1Bad == 0 && dirS.st2Bad == 0
                  && dirS.maxStd() < MAG_HEALTH_MAX_STD_UT && dirS.meanNorm() > 5.f && dirS.meanNorm() < 300.f;
  Serial.print(F("F. VERDICT: "));
  if (directOk)
    Serial.println(F("the sensor gives steady, valid data when read DIRECTLY, so the chip works and the "
                     "automatic read path is what corrupts it -- fixable in software."));
  else if (dirS.n > 0 && dirS.hofl * 2 > dirS.n)
    Serial.println(F("the sensor itself reports overflow (HOFL) on most direct reads: its data is not valid "
                     "(datasheet 13.4). A strong magnet next to the board, or a faulty sensor/supply."));
  else if (dirS.n == 0)
    Serial.println(F("the sensor never signalled data ready on direct reads: it is not measuring."));
  else
    Serial.println(F("the data is unsteady or malformed even when read directly: the fault is in the "
                     "sensor or its link inside the module, not in how the sketch reads it."));
  Serial.println(F("---- END OF DIAGNOSTIC ----"));
}

static bool loadMagCalibration() {
  prefs.begin("usque", true);
  bool ok = prefs.getBool("magOk", false)
            && prefs.getBytes("magBias", magBias, sizeof(magBias)) == sizeof(magBias)
            && prefs.getBytes("magScale", magScale, sizeof(magScale)) == sizeof(magScale);
  prefs.end();
  if (!ok) { for (int i = 0; i < 3; ++i) { magBias[i] = 0.f; magScale[i] = 1.f; } }
  return ok;
}

// ------------------------------------------------------------------ Euler angles for PRINTING
// The filter state is a quaternion and has no singularity. Roll/pitch/yaw (ZYX) do: at pitch
// = +-90 deg only (roll -+ yaw) is defined, and within a few degrees of it a 1 deg change of the
// real attitude moves the standard roll and yaw by 1/cos(pitch) -- 14 deg per degree at 86 deg --
// in opposite directions. So near +-90 the printed yaw is HELD at its last value and roll/pitch
// are taken from Rz(-yaw)*R, which carries the whole rotation smoothly (pitch may then pass 90).
// Hysteresis (enter above 75 deg, leave below 70 deg) stops the two formulas alternating when
// the board is held near the edge of the band. nearGL reports which formula was used.
static void eulerDegHeld(const float q[4], float &roll, float &pitch, float &yaw, bool &nearGL) {
  static bool  held = false, haveYaw = false;
  static float lastYaw = 0.f;
  const float R2D = 57.29578f;
  float w = q[0], x = q[1], y = q[2], z = q[3];
  float R11 = 1.f-2.f*(y*y+z*z), R12 = 2.f*(x*y-w*z), R13 = 2.f*(x*z+w*y);
  float R21 = 2.f*(x*y+w*z),     R22 = 1.f-2.f*(x*x+z*z), R23 = 2.f*(y*z-w*x);
  float R31 = 2.f*(x*z-w*y),     R32 = 2.f*(y*z+w*x),     R33 = 1.f-2.f*(x*x+y*y);
  float pStd = asinf(fmaxf(-1.f, fminf(1.f, -R31))) * R2D;
  if (!held && fabsf(pStd) > 75.f && haveYaw) held = true;
  else if (held && fabsf(pStd) < 70.f) held = false;
  nearGL = held;
  if (!held) {
    roll = atan2f(R32, R33) * R2D; pitch = pStd; yaw = atan2f(R21, R11) * R2D;
    lastYaw = yaw; haveYaw = true;
    return;
  }
  yaw = lastYaw;
  float c = cosf(yaw / R2D), sn = sinf(yaw / R2D);
  // Rp = Rz(-yaw) * R  ~=  Ry(pitch) * Rx(roll)
  float Rp11 =  c*R11 + sn*R21;
  float Rp22 = -sn*R12 + c*R22, Rp23 = -sn*R13 + c*R23;
  roll  = atan2f(-Rp23, Rp22) * R2D;
  pitch = atan2f(-R31, Rp11) * R2D;
}

// ------------------------------------------------------------------ causal accel LPF
// 4th-order Butterworth as two cascaded biquads (RBJ cookbook LPF form),
// Q1=0.541196/Q2=1.306563 -- the standard pole-angle split for a 4-pole
// Butterworth. This is the CAUSAL, real-time analogue of main_usque.py's
// offline zero-phase filtfilt(butter(4, ...)): same magnitude passband,
// but adds group delay (~1/(2*pi*fc) order of a few ms at 15 Hz) that a
// zero-phase filter does not -- unavoidable for anything running online.
struct Biquad {
  float b0=0, b1=0, b2=0, a1=0, a2=0, z1=0, z2=0;
  void design(float fc, float fs, float Q) {
    float w0 = 2.0f * PI * fc / fs;
    float cw = cosf(w0), sw = sinf(w0);
    float alpha = sw / (2.0f*Q);
    float a0 = 1.0f + alpha;
    b0 = ((1.0f-cw)/2.0f) / a0; b1 = (1.0f-cw) / a0; b2 = b0;
    a1 = (-2.0f*cw) / a0; a2 = (1.0f-alpha) / a0;
    z1 = z2 = 0.0f;
  }
  float process(float x) {
    float y = b0*x + z1;
    z1 = b1*x - a1*y + z2;
    z2 = b2*x - a2*y;
    return y;
  }
};
struct AccelLpf {
  Biquad stage1[3], stage2[3];
  bool designed = false;
  void design(float fc, float fs) {
    for (int i = 0; i < 3; ++i) { stage1[i].design(fc, fs, 0.541196f); stage2[i].design(fc, fs, 1.306563f); }
    designed = true;
  }
  void process(const float in[3], float out[3]) {
    for (int i = 0; i < 3; ++i) out[i] = stage2[i].process(stage1[i].process(in[i]));
  }
} accelLpf;
const float ACCEL_LPF_CUTOFF_HZ = 15.0f;

// ------------------------------------------------------------------ robust IMU acquisition
// The 5 Oct static log showed the SENSOR DATA failing, not the filter: single corrupt reads
// (gyro at full scale for one sample -> a 30 deg heading jump), the gyro noise rising 10x at
// 40 s, and the accelerometer output stepping to |a| = 16 m/s^2 at 84 s and staying there
// (-> pitch -15 deg with the board flat), all with the board at rest. The library's
// dataReady()/getAGMT() also write the bank-select register ~4 times per sample, so a
// disturbed I2C bus has thousands of chances per second to corrupt a read or mis-address a
// write. The flight loop therefore:
//   * reads accel+gyro (+mag) with ONE burst read and NO register writes
//   * rejects samples that are malformed or jump faster than a hand can move the board, and
//     holds the last good one instead
//   * checks once per second that the sensor's configuration and offset registers are still
//     what was set at boot, reports any change and re-configures the sensor
//   * stops using the accelerometer (gyro only) while |a| is far from gravity at rest
//   * runs the bus at 100 kHz and the serial port fast enough that printing cannot starve it
//     (the long print line at 115200 baud had cut the IMU loop to ~60 Hz)
const uint32_t I2C_CLOCK_HZ  = 100000;
const uint32_t SERIAL_BAUD   = 460800;     // the viewer must open the port at the same speed
const float    IMU_RATE_HZ   = 250.0f;
const int32_t  MAX_GYRO_STEP = 16375;      // counts between consecutive samples: 250 deg/s at 65.5 LSB per deg/s
const int32_t  MAX_ACC_STEP  = 16384;      // 2 g at 8192 LSB per g
static uint32_t imuBadSamples = 0, imuReadErrors = 0, imuCfgRepairs = 0, imuHzLast = 0;
static bool     accFault = false;

static const int CFG_N = 23;
static uint8_t cfgRef[CFG_N];
static const char *cfgName(int i) {
  static const char *b0[] = { "LP_CONFIG", "PWR_MGMT_1", "PWR_MGMT_2" };
  if (i < 3)  return b0[i];
  if (i < 11) return "accel offset (bank 1, 0x14..0x1B)";
  if (i == 11) return "GYRO_SMPLRT_DIV";
  if (i == 12) return "GYRO_CONFIG_1";
  if (i == 13) return "GYRO_CONFIG_2";
  if (i < 20) return "gyro offset (bank 2, 0x03..0x08)";
  if (i == 20) return "ACCEL_SMPLRT_DIV_2";
  if (i == 21) return "ACCEL_CONFIG";
  return "ACCEL_CONFIG_2";
}
// Reads the registers that define how the sensor measures. Leaves bank 0 selected and checks it.
static bool readImuConfig(uint8_t *out) {   // out: CFG_N bytes
  bool ok = true;
  ok &= imu.setBank(0) == ICM_20948_Stat_Ok; ok &= imu.read(AGB0_REG_LP_CONFIG, out, 3) == ICM_20948_Stat_Ok;
  ok &= imu.setBank(1) == ICM_20948_Stat_Ok; ok &= imu.read(AGB1_REG_XA_OFFS_H, out + 3, 8) == ICM_20948_Stat_Ok;
  ok &= imu.setBank(2) == ICM_20948_Stat_Ok; ok &= imu.read(AGB2_REG_GYRO_SMPLRT_DIV, out + 11, 9) == ICM_20948_Stat_Ok;
  ok &= imu.read(AGB2_REG_ACCEL_SMPLRT_DIV_2, out + 20, 1) == ICM_20948_Stat_Ok;
  ok &= imu.read(AGB2_REG_ACCEL_CONFIG, out + 21, 2) == ICM_20948_Stat_Ok;
  uint8_t who = 0;
  for (int k = 0; k < 3; ++k) {             // back to bank 0, verified by WHO_AM_I (0xEA lives in bank 0 only)
    imu.setBank(0);
    if (imu.read(0x00, &who, 1) == ICM_20948_Stat_Ok && who == 0xEA) break;
  }
  return ok && who == 0xEA;
}
static void configureImu(bool withMag) {
  imu.swReset();  delay(50);
  imu.sleep(false);
  imu.lowPower(false);
  ICM_20948_fss_t fss; fss.a = gpm4; fss.g = dps500;
  imu.setFullScale((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), fss);
  ICM_20948_dlpcfg_t dlp; dlp.a = acc_d111bw4_n136bw; dlp.g = gyr_d119bw5_n154bw3;
  imu.setDLPFcfg((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), dlp);
  imu.enableDLPF((ICM_20948_Internal_Acc | ICM_20948_Internal_Gyr), true);
  if (withMag) imu.startupMagnetometer();
  else imu.i2cMasterEnable(false);          // no usable magnetometer: stop the sensor's internal bus too
  imu.setBank(0);
}
// Once per second: has the sensor's configuration changed under us?
static void imuWatchdog() {
  uint8_t cur[CFG_N];
  if (!readImuConfig(cur)) { ++imuReadErrors; return; }
  int bad = -1;
  for (int i = 0; i < CFG_N; ++i) if (cur[i] != cfgRef[i]) { bad = i; break; }
  if (bad < 0) return;
  ++imuCfgRepairs;
  Serial.print(F("IMU CONFIG CHANGED: ")); Serial.print(cfgName(bad)); Serial.print(F(" [")); Serial.print(bad);
  Serial.print(F("] was 0x")); Serial.print(cfgRef[bad], HEX); Serial.print(F(" now 0x")); Serial.print(cur[bad], HEX);
  Serial.println(F(" -> re-configuring the sensor"));
  configureImu(useMag);
  if (readImuConfig(cur)) {
    for (int i = 0; i < CFG_N; ++i) if (cur[i] != cfgRef[i]) {
      Serial.print(F("  after re-configuring, ")); Serial.print(cfgName(i)); Serial.print(F(" is 0x")); Serial.print(cur[i], HEX);
      Serial.println(F(" (accepted as the new reference)"));
      cfgRef[i] = cur[i];
    }
  }
}
// One burst read, no writes. Returns false when the bus reports an error.
static bool readImuRaw(int16_t a[3], int16_t g[3], int16_t m[3], uint8_t &st1, uint8_t &st2) {
  uint8_t buf[23];
  uint32_t n = useMag ? 23 : 12;
  if (imu.read(AGB0_REG_ACCEL_XOUT_H, buf, n) != ICM_20948_Stat_Ok) return false;
  bool same = true;
  for (int i = 1; i < 12; ++i) if (buf[i] != buf[0]) { same = false; break; }
  if (same) return false;                   // 12 identical bytes (all 0x00 / 0xFF): not sensor data
  for (int i = 0; i < 3; ++i) {
    a[i] = (int16_t)((buf[2*i] << 8) | buf[2*i + 1]);
    g[i] = (int16_t)((buf[6 + 2*i] << 8) | buf[7 + 2*i]);
  }
  if (useMag) {
    st1 = buf[14]; st2 = buf[22];
    for (int i = 0; i < 3; ++i) m[i] = (int16_t)((buf[16 + 2*i] << 8) | buf[15 + 2*i]);   // little endian
  } else { st1 = st2 = 0; m[0] = m[1] = m[2] = 0; }
  return true;
}

// ------------------------------------------------------------------ timing
uint32_t lastMicros = 0;
uint32_t lastPrint  = 0;
uint32_t nextUpdateMs = 0;     // 30 Hz measurement-update decimation deadline
const float FUSION_RATE_HZ = 30.0f;
bool     started    = false;
bool     lpfPrimed  = false;   // one IMU-rate estimate needed before designing the biquads

void setup() {
  Serial.setTxBufferSize(4096);   // a whole print line fits: printing never blocks the IMU loop
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 3000) {}
  Serial.println(F("boot: Serial up"));
  delay(50);

  Wire.begin();
  Wire.setClock(I2C_CLOCK_HZ);
  Serial.println(F("boot: Wire.begin() done, probing ICM-20948..."));
  delay(50);

  bool ok = false;
  for (int i = 0; i < 5 && !ok; ++i) {
    Serial.print(F("boot: imu.begin() attempt ")); Serial.println(i);
    delay(20);
    imu.begin(Wire, AD0_VAL);
    ok = (imu.status == ICM_20948_Stat_Ok);
    if (!ok) {
      Serial.print(F("ICM-20948 not found (status=")); Serial.print((int)imu.status);
      Serial.println(F("), retrying...")); delay(400);
    }
  }
  if (!ok) { Serial.println(F("ICM-20948 init FAILED. Check wiring/address.")); while (1) delay(1000); }
  Serial.println(F("boot: ICM-20948 found, configuring..."));

  configureImu(true);

  USQUE_AHRS::Params p;
  // ICM-20948 gyro noise ~0.015 dps/sqrtHz -> ~2.6e-4 rad/s/sqrtHz; the
  // PC project's own tuned value (see main_usque.py FILTER_PARAMS) already
  // includes headroom for the cheap part + mounting vibration -- reuse it.
  p.gyro_noise_var = 9.0e-5f;
  p.accel_noise_var = 1.0e-2f;
  p.mag_noise_var   = MAG_NOISE_VAR;
  p.mag_inclination_deg = MAG_INCLINATION_DEG;
  p.mag_declination_deg = MAG_DECLINATION_DEG;
  p.mag_ref_from_boot   = MAG_REF_FROM_BOOT;
  p.use_mag_bias_state  = USE_MAG_BIAS_STATE;
  ahrs.setParams(p);

  // ---- magnetometer bring-up: health check -> (re)calibration -> use it, or run without it ----
  pinMode(BOOT_BUTTON_PIN, INPUT_PULLUP);
  prefs.begin("usque", false);
  bool forceCal = prefs.getBool("forceCal", false);
  if (forceCal) prefs.putBool("forceCal", false);
  if (prefs.getBool("noSeedOnce", false)) { prefs.putBool("noSeedOnce", false); seedGyroBias = false; }
  useAccelLpf = prefs.getBool("accLpf", USE_ACCEL_LPF);
  Serial.println(useAccelLpf ? F("accelerometer low-pass: ON (causal 15 Hz). Send L to switch it off.")
                             : F("accelerometer low-pass: OFF (raw samples). Send L to switch it on."));
  prefs.end();
  bool forceDiag = false;
  bool haveCal = loadMagCalibration();
  Serial.println(haveCal ? F("stored magnetometer calibration found. To redo it: hold the BOOT button or send C within 3 s...")
                         : F("no stored magnetometer calibration -- it will be done now."));
  for (uint32_t t0 = millis(); (haveCal || millis() - t0 < 1000) && !forceCal && millis() - t0 < 3000; ) {
    if (digitalRead(BOOT_BUTTON_PIN) == LOW) forceCal = true;
    while (Serial.available()) {
      char ch = Serial.read();
      if (ch == 'C' || ch == 'c') forceCal = true;
      if (ch == 'D' || ch == 'd') forceDiag = true;
    }
    delay(10);
  }
  if (forceDiag) magDiagnostic();
  Serial.println(F("Keep the board STILL for the magnetometer health check (2 s)..."));
  if (!magHealthCheck()) {
    useMag = false;
    if (!forceDiag) magDiagnostic();
    Serial.println(F("MAGNETOMETER NOT USABLE: its readings jump while the board is still, so they are not "
                     "field data and cannot be calibrated. Running on gyro + accel only: roll/pitch are "
                     "unaffected, heading is RELATIVE to the boot direction."));
  } else {
    useMag = haveCal;
    if (!haveCal || forceCal) {
      bool okCal = runMagCalibration();
      if (okCal) useMag = true;
      else if (haveCal) { loadMagCalibration(); Serial.println(F("keeping the previously stored calibration.")); }
      else { useMag = false; Serial.println(F("no valid calibration -> magnetometer OFF (gyro + accel only, relative heading).")); }
    }
    if (useMag) {
      Serial.print(F("magnetometer ON. calibration: offset ")); Serial.print(magBias[0], 1); Serial.print(' ');
      Serial.print(magBias[1], 1); Serial.print(' '); Serial.print(magBias[2], 1);
      Serial.print(F(" uT, scale ")); Serial.print(magScale[0], 3); Serial.print(' ');
      Serial.print(magScale[1], 3); Serial.print(' '); Serial.println(magScale[2], 3);
    }
  }

  // from here on the loop reads the sensor itself: fixed configuration, reference snapshot, bank 0
  if (!useMag) configureImu(false);
  delay(50);
  if (!readImuConfig(cfgRef)) Serial.println(F("WARNING: could not read back the IMU configuration."));
  accelLpf.design(ACCEL_LPF_CUTOFF_HZ, IMU_RATE_HZ); lpfPrimed = true;
  Serial.println(F("USQUE_AHRS ready. Hold the board flat & still for init..."));
  lastMicros = micros();
}

void loop() {
  // ---- fixed-rate sampling (no data-ready polling: that costs a register write per sample)
  static uint32_t nextImuUs = 0;
  const uint32_t periodUs = (uint32_t)(1e6f / IMU_RATE_HZ);
  uint32_t now = micros();
  if ((int32_t)(now - nextImuUs) < 0) return;
  nextImuUs += periodUs;
  if ((int32_t)(now - nextImuUs) > (int32_t)(5 * periodUs)) nextImuUs = now + periodUs;   // fell behind: resync

  // ---- once per second: configuration check, sample-rate count
  static uint32_t lastWatchMs = 0, imuHzCount = 0;
  if (millis() - lastWatchMs >= 1000) {
    lastWatchMs = millis(); imuHzLast = imuHzCount; imuHzCount = 0;
    imuWatchdog();
    now = micros();
  }

  // ---- read one sample; a failed or implausible one is replaced by the last good one
  static int16_t aLast[3] = { 0, 0, 0 }, gLast[3] = { 0, 0, 0 }, mLast[3] = { 0, 0, 0 };
  static bool haveLast = false;
  static uint8_t rejectRun = 0;
  int16_t aC[3], gC[3], mC[3]; uint8_t st1 = 0, st2 = 0;
  bool okRead = readImuRaw(aC, gC, mC, st1, st2);
  if (!okRead) ++imuReadErrors;
  bool okSample = okRead;
  if (okSample && haveLast && rejectRun < 10) {
    for (int i = 0; i < 3; ++i)
      if (abs((int32_t)gC[i] - gLast[i]) > MAX_GYRO_STEP || abs((int32_t)aC[i] - aLast[i]) > MAX_ACC_STEP) okSample = false;
    if (!okSample) ++imuBadSamples;
  }
  if (okSample) {
    for (int i = 0; i < 3; ++i) { aLast[i] = aC[i]; gLast[i] = gC[i]; mLast[i] = mC[i]; }
    haveLast = true; rejectRun = 0; ++imuHzCount;
  } else {
    if (rejectRun < 255) ++rejectRun;
    if (!haveLast) return;                              // nothing valid yet
  }
  imu.agmt.magStat2 = st2;                              // magSampleValid() reads the overflow flag from here

  float dt = (now - lastMicros) * 1e-6f;
  lastMicros = now;
  if (dt > 0.05f) dt = 0.05f;                           // a stall must not turn into one huge integration step

  float accRaw[3], gyr[3];
  remapAccel(aLast[0] / 8.192f, aLast[1] / 8.192f, aLast[2] / 8.192f, accRaw);   // +-4 g: 8192 LSB/g -> mg
  remapGyro (gLast[0] / 65.5f,  gLast[1] / 65.5f,  gLast[2] / 65.5f,  gyr);      // +-500 dps: 65.5 LSB per deg/s
  float accLp[3];
  if (lpfPrimed) accelLpf.process(accRaw, accLp);
  else { accLp[0]=accRaw[0]; accLp[1]=accRaw[1]; accLp[2]=accRaw[2]; }

  // Vibration/disturbance proxy: high-frequency residual (raw - low-passed)
  // relative to a causal rolling baseline -- see USQUE_AHRS.h's VibBaseline.
  // Updated every loop (internally rate-limited to ~1 Hz); the LATEST value
  // is what gets passed to update() below.
  float throttlePct = ahrs.updateVibrationProxy(accRaw, accLp);
  const float *accUse = useAccelLpf ? accLp : accRaw;   // what the measurement update gets

  // ---- accelerometer plausibility: at rest it must read gravity. If the low-passed |a| stays
  // far from 9.81 while the gyro is quiet, the accelerometer is not measuring gravity (5 Oct:
  // it stepped to 16 m/s^2). Stop using it (gyro only), say so, and re-configure the sensor.
  {
    static uint32_t faultSince = 0, lastFix = 0;
    float anLp = sqrtf(accLp[0]*accLp[0] + accLp[1]*accLp[1] + accLp[2]*accLp[2]);
    float wnQ  = sqrtf(gyr[0]*gyr[0] + gyr[1]*gyr[1] + gyr[2]*gyr[2]);
    bool off = fabsf(anLp - 9.81f) > 2.5f && wnQ < 0.2f;
    if (!off) { faultSince = 0; if (accFault) Serial.println(F("accelerometer reads gravity again -> using it again")); accFault = false; }
    else if (faultSince == 0) faultSince = millis();
    else if (millis() - faultSince > 1000 && (!accFault || millis() - lastFix > 5000)) {
      accFault = true; lastFix = millis(); ++imuCfgRepairs;
      Serial.print(F("ACCELEROMETER FAULT: |a| = ")); Serial.print(anLp, 2);
      Serial.println(F(" m/s^2 at rest -> not used (gyro only); re-configuring the sensor"));
      configureImu(useMag);
    }
  }

  // Stale-mag-sample gating: the AK09916 only refreshes at ~100 Hz while
  // accel/gyro run at ~225 Hz -- imu.magX/Y/Z() silently repeats the same
  // sample across several loop iterations. Only treat it as NEW information
  // when the raw reading actually changed.
  static float lastMagRaw[3] = { 1e30f, 1e30f, 1e30f };
  static float magCal[3] = { 0.f, 0.f, 0.f };
  static bool  magEverOk = false;
  static uint32_t magHzWindowStart = 0;
  static uint32_t magHzCount = 0;
  static uint32_t magHzLast = 0;
  float mx = mLast[0] * 0.15f, my = mLast[1] * 0.15f, mz = mLast[2] * 0.15f;     // 0.15 uT per count
  if (!okSample) { mx = lastMagRaw[0]; my = lastMagRaw[1]; mz = lastMagRaw[2]; }   // not a fresh sample
  bool magOk = magEverOk || (mx != 0.f || my != 0.f || mz != 0.f);
  bool magFresh = (mx != 0.f || my != 0.f || mz != 0.f) &&
                  (mx != lastMagRaw[0] || my != lastMagRaw[1] || mz != lastMagRaw[2]);
  static uint32_t magBadCount = 0;
  if (magFresh && !magSampleValid(mx, my, mz)) {      // remember it (so it is not re-tested) but do not use it
    lastMagRaw[0] = mx; lastMagRaw[1] = my; lastMagRaw[2] = mz;
    ++magBadCount; magFresh = false;
  }
  if (!useMag) { magOk = true; }                      // init must not wait for a magnetometer that is off
  if (magFresh) ++magHzCount;
  if (millis() - magHzWindowStart >= 1000) { magHzLast = magHzCount; magHzCount = 0; magHzWindowStart = millis(); }
  if (magFresh) {
    lastMagRaw[0] = mx; lastMagRaw[1] = my; lastMagRaw[2] = mz;
    remapMag(mx, my, mz, magCal);
    magEverOk = true; magOk = true;
  }

  // ---- one-time init: TRIAD, averaged over a still window (equivalent to
  // main_usque.py's 'mag_triad' init mode -- roll/pitch/yaw all from
  // sensors, TRIAD against the configured MAG_INCLINATION/DECLINATION_DEG).
  // A residual heading BIAS from this (declination error, uncalibrated
  // hard-iron) is no longer permanently frozen in the way it was for
  // UKF_AHRS.ino -- dpsi_mag converges it out over the first few minutes of
  // flight. See setHeadingDeg() below for an equivalent of 'rtk_yaw' (an
  // externally-verified heading, applied once you can check one). ----
  const float MAG_STABLE_SPREAD_UT = 3.0f;
  const uint32_t MAG_STABLE_MAX_WAIT_MS = 5000;
  // Gyro-bias seed window. The seed is the gyro's mean while still, and with no magnetometer
  // nothing corrects the z bias afterwards: every 0.01 deg/s of seed error is 0.6 deg/min of
  // heading drift. The 5 Oct logs showed the old 0.6 s window, taken right after the sensor was
  // configured, missing the true bias by up to 0.1 deg/s (gyro still settling). So: wait
  // GYRO_SETTLE_MS after start-up, average over BIAS_WINDOW_MS, and accept the window only if
  // its two halves agree to BIAS_HALVES_MAX_DPS -- otherwise the gyro is still settling or the
  // board is being moved, and the window starts again.
  const uint32_t GYRO_SETTLE_MS = 2000;
  const uint32_t BIAS_WINDOW_MS = 3000;
  const float    BIAS_HALVES_MAX_DPS = 0.04f;

  if (!started) {
    static uint32_t stillStart = 0;
    static uint32_t lastWaitPrint = 0;
    static float accAccum[3] = {0,0,0};
    static float magAccum[3] = {0,0,0};
    static float gyrAccum[3] = {0,0,0};
    static float magMin[3] = {1e9f,1e9f,1e9f};
    static float magMax[3] = {-1e9f,-1e9f,-1e9f};
    static uint32_t nAcc = 0, nMag = 0;
    static float gyrHalf[3] = {0,0,0};                   // sum over the first half of the window
    static uint32_t nHalf = 0;
    static uint32_t initEnterMs = 0;
    if (initEnterMs == 0) initEnterMs = millis();
    float an = sqrtf(accRaw[0]*accRaw[0]+accRaw[1]*accRaw[1]+accRaw[2]*accRaw[2]);
    float wn = sqrtf(gyr[0]*gyr[0]+gyr[1]*gyr[1]+gyr[2]*gyr[2]);
    bool still = fabsf(an - 9.81f) < 0.4f && wn < 0.05f && magOk
                 && (millis() - initEnterMs) > GYRO_SETTLE_MS;
    if (millis() - lastWaitPrint > 1000) {
      lastWaitPrint = millis();
      Serial.print(F("waiting for still+mag:  |accel|=")); Serial.print(an, 3);
      Serial.print(F(" (want ~9.81)  |gyro|=")); Serial.print(wn, 3);
      Serial.print(F(" rad/s (want <0.05)  magOk=")); Serial.println(magOk ? "yes" : "NO");
    }
    if (still) {
      if (stillStart == 0) {
        stillStart = millis();
        accAccum[0]=accAccum[1]=accAccum[2]=0; magAccum[0]=magAccum[1]=magAccum[2]=0;
        gyrAccum[0]=gyrAccum[1]=gyrAccum[2]=0;
        gyrHalf[0]=gyrHalf[1]=gyrHalf[2]=0; nHalf = 0;
        magMin[0]=magMin[1]=magMin[2]=1e9f; magMax[0]=magMax[1]=magMax[2]=-1e9f;
        nAcc = 0; nMag = 0;
      }
      accAccum[0]+=accRaw[0]; accAccum[1]+=accRaw[1]; accAccum[2]+=accRaw[2];
      gyrAccum[0]+=gyr[0]; gyrAccum[1]+=gyr[1]; gyrAccum[2]+=gyr[2];
      nAcc++;
      if (millis() - stillStart <= BIAS_WINDOW_MS / 2) {
        gyrHalf[0]+=gyr[0]; gyrHalf[1]+=gyr[1]; gyrHalf[2]+=gyr[2]; nHalf++;
      }
      if (magFresh) {
        magAccum[0]+=magCal[0]; magAccum[1]+=magCal[1]; magAccum[2]+=magCal[2]; nMag++;
        for (int i = 0; i < 3; ++i) { if (magCal[i]<magMin[i]) magMin[i]=magCal[i]; if (magCal[i]>magMax[i]) magMax[i]=magCal[i]; }
      }
      float magSpread = 0.f;
      for (int i = 0; i < 3; ++i) { float sp = magMax[i]-magMin[i]; if (sp > magSpread) magSpread = sp; }
      bool magSettled = !useMag || ((nMag >= 3) && (magSpread < MAG_STABLE_SPREAD_UT));
      bool magTimedOut = (millis() - stillStart) > MAG_STABLE_MAX_WAIT_MS;
      if (!useMag) nMag = 1;                             // nothing to average
      bool windowDone = millis() - stillStart > BIAS_WINDOW_MS && nAcc > 0 && nMag > 0 && (magSettled || magTimedOut);
      if (windowDone && nHalf > 0 && nAcc > nHalf) {
        // first half against second half: a settled gyro on a resting board gives the same mean
        float worst = 0.f;
        for (int i = 0; i < 3; ++i) {
          float h1 = gyrHalf[i] / nHalf, h2 = (gyrAccum[i] - gyrHalf[i]) / (nAcc - nHalf);
          worst = fmaxf(worst, fabsf(h1 - h2) * 57.2958f);
        }
        if (worst > BIAS_HALVES_MAX_DPS) {
          Serial.print(F("bias window rejected: its two halves differ by ")); Serial.print(worst, 3);
          Serial.println(F(" deg/s (gyro still settling, or the board moved) -- repeating"));
          stillStart = 0; windowDone = false;
        }
      }
      if (windowDone) {
        float accAvg[3] = { accAccum[0]/nAcc, accAccum[1]/nAcc, accAccum[2]/nAcc };
        float magAvg[3] = { magAccum[0]/nMag, magAccum[1]/nMag, magAccum[2]/nMag };
        float gyrAvg[3] = { gyrAccum[0]/nAcc, gyrAccum[1]/nAcc, gyrAccum[2]/nAcc };
        if (!magSettled) {
          Serial.print(F("WARNING: mag never settled within "));
          Serial.print(MAG_STABLE_MAX_WAIT_MS); Serial.print(F("ms (spread "));
          Serial.print(magSpread, 2); Serial.println(F("uT) -- heading reference may be unreliable. "
                       "Once you can verify the board's actual heading externally (phone compass, "
                       "landmark), send \"N<deg>\" to correct it -- dpsi_mag will also slowly self-"
                       "correct a residual bias on its own over the next few minutes of flight."));
        }
        if (useMag) {
          ahrs.begin(accAvg, magAvg);
        } else {
          // no magnetometer: roll/pitch from the accelerometer, heading defined as 0 at boot.
          // The stand-in vector only gives TRIAD a second direction; its yaw is overridden.
          float standIn[3] = { 1.f, 0.f, 0.5f };
          ahrs.beginWithHeading(accAvg, standIn, 0.f);
        }
        if (seedGyroBias) ahrs.setGyroBias(gyrAvg);
        else Serial.println(F("gyro bias NOT seeded (demonstration run): the filter starts from zero bias, "
                              "1-sigma 0.57 deg/s, and has to converge to the value below."));
        started = true;
        nextUpdateMs = millis();
        Serial.println(F("Initialised."));
        Serial.print(F("accel FRD (avg over ")); Serial.print(nAcc); Serial.print(F(" samples):  "));
        Serial.print(accAvg[0], 2); Serial.print(F("  ")); Serial.print(accAvg[1], 2); Serial.print(F("  ")); Serial.println(accAvg[2], 2);
        Serial.print(F("mag FRD (avg over ")); Serial.print(nMag); Serial.print(F(" fresh samples, spread "));
        Serial.print(magSpread, 2); Serial.print(F("uT):  "));
        Serial.print(magAvg[0], 2); Serial.print(F("  ")); Serial.print(magAvg[1], 2); Serial.print(F("  ")); Serial.println(magAvg[2], 2);
        // also in deg/s, on one parseable line: the reference the bias estimate is compared with
        Serial.print(F("still_gyro_dps:")); Serial.print(gyrAvg[0]*57.2958f, 4); Serial.print(',');
        Serial.print(gyrAvg[1]*57.2958f, 4); Serial.print(','); Serial.print(gyrAvg[2]*57.2958f, 4);
        Serial.print(F(",seeded:")); Serial.println(seedGyroBias ? 1 : 0);
        Serial.print(F("gyro bias seed, rad/s (avg over ")); Serial.print(nAcc); Serial.print(F(" samples):  "));
        Serial.print(gyrAvg[0], 5); Serial.print(F("  ")); Serial.print(gyrAvg[1], 5); Serial.print(F("  ")); Serial.println(gyrAvg[2], 5);
      }
    } else {
      stillStart = 0;
    }
    return;
  }

  // ---- manual heading calibration ----
  // Send "N<heading_deg>\n" once you've verified, by some means OUTSIDE this
  // filter, that the board is CURRENTLY pointing at that heading. Equivalent
  // to main_usque.py's 'rtk_yaw' init mode, applied post-hoc: a magnetometer
  // alone can never establish an absolute bearing on its own.
  {
    static char cmdBuf[16];
    static uint8_t cmdLen = 0;
    while (Serial.available()) {
      char c = Serial.read();
      if (c == '\n' || c == '\r') {
        if (cmdLen > 0 && (cmdBuf[0] == 'L' || cmdBuf[0] == 'l')) {
          // flip the accelerometer low-pass, remember it, and restart for a clean run
          prefs.begin("usque", false); prefs.putBool("accLpf", !useAccelLpf); prefs.end();
          Serial.println(useAccelLpf ? F("accelerometer low-pass -> OFF, restarting...") : F("accelerometer low-pass -> ON, restarting..."));
          delay(100);
          ESP.restart();
        }
        if (cmdLen > 0 && (cmdBuf[0] == 'Z' || cmdBuf[0] == 'z')) {
          // one run without the gyro-bias seed (bias-convergence demonstration)
          prefs.begin("usque", false); prefs.putBool("noSeedOnce", true); prefs.end();
          Serial.println(F("restarting for one run WITHOUT the gyro-bias seed..."));
          delay(100);
          ESP.restart();
        }
        if (cmdLen > 0 && (cmdBuf[0] == 'C' || cmdBuf[0] == 'c')) {
          // redo the magnetometer calibration: restart into it (the boot code runs it)
          prefs.begin("usque", false); prefs.putBool("forceCal", true); prefs.end();
          Serial.println(F("restarting into the magnetometer calibration..."));
          delay(100);
          ESP.restart();
        }
        if (cmdLen > 0 && (cmdBuf[0] == 'N' || cmdBuf[0] == 'n') && started) {
          cmdBuf[cmdLen] = 0;
          float heading = atof(cmdBuf + 1);
          ahrs.setHeadingDeg(heading);
          Serial.print(F("Heading calibrated to ")); Serial.print(heading, 1); Serial.println(F(" deg"));
        }
        cmdLen = 0;
      } else if (cmdLen < sizeof(cmdBuf) - 1) {
        cmdBuf[cmdLen++] = c;
      }
    }
  }

  // ---- filter: predict every IMU sample (RAW gyro), update decimated to
  // ~30 Hz (LOW-PASSED accel) -- matches main_usque.py's FUSION_MODE=
  // 'decimate' @ FUSION_RATE_HZ=30 and LPF_MODE='accel_only' exactly. ----
  ahrs.predict(gyr, dt);
  // raw gyro averaged over each print interval (the single sample at print time would be noise)
  static float gyrSum[3] = { 0.f, 0.f, 0.f };
  static uint32_t gyrN = 0;
  gyrSum[0] += gyr[0]; gyrSum[1] += gyr[1]; gyrSum[2] += gyr[2]; ++gyrN;

  if ((int32_t)(millis() - nextUpdateMs) >= 0) {
    nextUpdateMs = millis() + (uint32_t)(1000.0f / FUSION_RATE_HZ);
    if (!accFault) ahrs.update(accUse, (useMag && magFresh) ? magCal : nullptr, throttlePct);
  }

  // ---- print at ~20 Hz, Arduino Serial Plotter "label:value" format ----
  if (millis() - lastPrint >= 50) {
    lastPrint = millis();
    float r, p, y, s[3], bg[3];
    float qq[4]; ahrs.getQuaternion(qq);
    bool nearGL;
    eulerDegHeld(qq, r, p, y, nearGL);   // gimbal-safe angles for display; the quaternion is the real output
    ahrs.getAttitudeStdDeg(s);
    ahrs.getGyroBias(bg);
    Serial.print(F("roll_deg:"));    Serial.print(r, 2);
    Serial.print(F(",pitch_deg:"));  Serial.print(p, 2);
    Serial.print(F(",yaw_deg:"));    Serial.print(y, 2);
    Serial.print(F(",qw:")); Serial.print(qq[0], 5);
    Serial.print(F(",qx:")); Serial.print(qq[1], 5);
    Serial.print(F(",qy:")); Serial.print(qq[2], 5);
    Serial.print(F(",qz:")); Serial.print(qq[3], 5);
    Serial.print(F(",sigRoll_deg:"));  Serial.print(s[0], 2);
    Serial.print(F(",sigPitch_deg:")); Serial.print(s[1], 2);
    Serial.print(F(",sigYaw_deg:"));   Serial.print(s[2], 2);
    Serial.print(F(",gyroBiasX_dps:")); Serial.print(bg[0]*57.2958f, 3);
    Serial.print(F(",gyroBiasY_dps:")); Serial.print(bg[1]*57.2958f, 3);
    Serial.print(F(",gyroBiasZ_dps:")); Serial.print(bg[2]*57.2958f, 3);
    // dpsi_mag: the magnetic HEADING BIAS state (see USQUE_AHRS.h's header
    // comment / mag_bias_analysis.py) -- a large, slowly-converging value
    // here means the boot-time TRIAD/calibration was off by that much; it
    // is being corrected automatically, not a fault.
    Serial.print(F(",psiMag_deg:")); Serial.print(ahrs.getPsiMagDeg(), 2);
    Serial.print(F(",sigPsiMag_deg:")); Serial.print(ahrs.getPsiMagStdDeg(), 2);
    Serial.print(F(",accelInfl:")); Serial.print(ahrs.getLastAccelInflation(), 2);
    Serial.print(F(",magInfl:")); Serial.print(ahrs.getLastMagInflation(), 2);
    Serial.print(F(",vibProxy:")); Serial.print(throttlePct, 2);
    Serial.print(F(",magHz:")); Serial.print(magHzLast);
    Serial.print(F(",yawHeld:")); Serial.print(nearGL ? 1 : 0);
    // the accelerometer (low-passed for readability, FRD, m/s^2) and the tilt it implies on its
    // own -- lets a log show whether a roll/pitch value comes from the sensor or from the filter.
    // accLpf says whether the UPDATE used the low-passed signal (1) or raw samples (0).
    Serial.print(F(",accX:")); Serial.print(accLp[0], 3);
    Serial.print(F(",accY:")); Serial.print(accLp[1], 3);
    Serial.print(F(",accZ:")); Serial.print(accLp[2], 3);
    Serial.print(F(",accRoll_deg:"));  Serial.print(atan2f(-accLp[1], -accLp[2]) * 57.29578f, 2);
    Serial.print(F(",accPitch_deg:")); Serial.print(atan2f(accLp[0], sqrtf(accLp[1]*accLp[1] + accLp[2]*accLp[2])) * 57.29578f, 2);
    // gyro: RAW rate (mean since the last print, deg/s, FRD) and the 1-sigma of the bias
    // estimate. Corrected rate = gyr - gyroBias: ~0 at rest once the bias estimate is right.
    float sbg[3]; ahrs.getGyroBiasStd(sbg);
    float gN = gyrN ? 57.2958f / gyrN : 0.f;
    Serial.print(F(",gyrX_dps:")); Serial.print(gyrSum[0]*gN, 3);
    Serial.print(F(",gyrY_dps:")); Serial.print(gyrSum[1]*gN, 3);
    Serial.print(F(",gyrZ_dps:")); Serial.print(gyrSum[2]*gN, 3);
    gyrSum[0] = gyrSum[1] = gyrSum[2] = 0.f; gyrN = 0;
    Serial.print(F(",sigBiasX_dps:")); Serial.print(sbg[0]*57.2958f, 4);
    Serial.print(F(",sigBiasY_dps:")); Serial.print(sbg[1]*57.2958f, 4);
    Serial.print(F(",sigBiasZ_dps:")); Serial.print(sbg[2]*57.2958f, 4);
    Serial.print(F(",biasSeeded:")); Serial.print(seedGyroBias ? 1 : 0);
    Serial.print(F(",accLpf:")); Serial.print(useAccelLpf ? 1 : 0);
    // data-quality counters: good samples per second, rejected samples, bus errors, sensor
    // re-configurations, accelerometer currently ignored
    Serial.print(F(",imuHz:")); Serial.print(imuHzLast);
    Serial.print(F(",badSmp:")); Serial.print(imuBadSamples);
    Serial.print(F(",rdErr:")); Serial.print(imuReadErrors);
    Serial.print(F(",cfgFix:")); Serial.print(imuCfgRepairs);
    Serial.print(F(",accFault:")); Serial.print(accFault ? 1 : 0);
    Serial.print(F(",magOn:")); Serial.print(useMag ? 1 : 0);
    Serial.print(F(",magBad:")); Serial.print(magBadCount);
    if (useMag) {   // without a usable magnetometer these six fields only cost serial bandwidth
      Serial.print(F(",magRawX_uT:")); Serial.print(mx, 2);
      Serial.print(F(",magRawY_uT:")); Serial.print(my, 2);
      Serial.print(F(",magRawZ_uT:")); Serial.print(mz, 2);
      Serial.print(F(",magCalX_uT:")); Serial.print(magCal[0], 2);
      Serial.print(F(",magCalY_uT:")); Serial.print(magCal[1], 2);
      Serial.print(F(",magCalZ_uT:")); Serial.print(magCal[2], 2);
    }
    Serial.println();
  }
}
