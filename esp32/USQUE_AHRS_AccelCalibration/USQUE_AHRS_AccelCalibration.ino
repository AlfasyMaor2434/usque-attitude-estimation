// ============================================================================
//  USQUE_AHRS_AccelCalibration.ino
//
//  UPDATED vs the old ESKF_AHRS_AccelCalibration.ino: that sketch's PRIMARY
//  output (per-axis ACCEL_BIAS[3]/ACCEL_SCALE[3], "paste into ESKF_AHRS.ino")
//  did not match what the actual filter sketch consumes -- USQUE_AHRS.ino
//  (like UKF_AHRS.ino before it) applies a single UNIFORM scale correction,
//  ACCEL_SCALE_CORR, not a per-axis bias+scale pair. That was a deliberate,
//  documented reversion: an earlier attempt at per-axis calibration left
//  |accel| stuck outside the still-init gate's tolerance, and the project
//  went back to the simpler uniform-scale approach, but this sketch's
//  printed "paste into ..." lines were never updated to match, so they were
//  producing output for a code path nothing actually uses.
//
//  This version's PRIMARY output is now the single ACCEL_SCALE_CORR value
//  that USQUE_AHRS.ino actually applies (computed the SAME way as that
//  sketch's own comment describes: rest flat & still, ACCEL_SCALE_CORR =
//  9.80665 / measured |accel|). The full 6-position per-axis bias/scale
//  data collection is KEPT and still printed, now explicitly labelled as a
//  DIAGNOSTIC (useful to see whether your unit has a real per-axis offset
//  worth caring about) rather than as something to paste anywhere.
//
//  Method (unchanged): gravity has a known, constant magnitude
//  (9.80665 m/s^2). Rest the board on each of its 6 faces in turn and each
//  axis will, at some point, see close to +g and close to -g. Tracking the
//  running min/max per axis across all the faces gives the per-axis
//  diagnostic; the flat-and-still capture (face 1, or any single capture)
//  gives the uniform scale correction.
//
//  HOW TO RUN:
//    1. Upload, open Serial Monitor at 115200. Close it in the main sketch
//       first -- one port, one owner.
//    2. It will prompt "HOLD STILL". Rest the board flat on a face and keep
//       it motionless for ~1.5s; it captures automatically.
//    3. It then prompts "MOVE TO A NEW FACE" -- pick it up, flip it to a
//       DIFFERENT face, and hold still again. Repeat for all 6 faces: flat,
//       upside-down, both sides, nose-up, nose-down. Order doesn't matter.
//    4. After 6 captures (or once the time limit is hit) it prints
//       ACCEL_SCALE_CORR (paste this one into USQUE_AHRS.ino) plus the
//       per-axis diagnostic.
// ============================================================================

#include <Wire.h>
#include "ICM_20948.h"

#define AD0_VAL      0
#define N_FACES      6       // captures to collect
#define TIME_LIMIT_S 180      // give up after this long regardless

ICM_20948_I2C imu;

static inline void remapAccelRaw(float ax, float ay, float az, float o[3]) {
  const float MG2MS2 = 9.80665f / 1000.0f;
  o[0] = -(ax * MG2MS2);
  o[1] = -(ay * MG2MS2);
  o[2] = -(az * MG2MS2);
}

float mn[3] = {  1e9f,  1e9f,  1e9f };
float mx_[3] = { -1e9f, -1e9f, -1e9f };
int nCaptured = 0;
uint32_t tStart = 0;
float firstCaptureNorm = -1.0f;   // |accel| on the FIRST capture -- gives ACCEL_SCALE_CORR directly if that face was flat

enum State { WAIT_STILL, HOLDING, WAIT_MOTION, FINISHED };
State state = WAIT_STILL;
uint32_t stillStart = 0;
float holdAccum[3] = {0,0,0};
uint32_t holdN = 0;

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

  Serial.println(F("USQUE_AHRS accelerometer calibration (6-position)"));
  Serial.print(F("Need ")); Serial.print(N_FACES); Serial.println(F(" faces, ~1.5s still each."));
  Serial.println(F("IMPORTANT: capture face 1 with the board FLAT (as it normally flies) --"));
  Serial.println(F("that capture is what ACCEL_SCALE_CORR (the primary output) is computed from."));
  Serial.println(F(">>> HOLD STILL, flat, now <<<"));
  tStart = millis();
}

void loop() {
  if (state == FINISHED) return;
  if (millis() - tStart > (uint32_t)TIME_LIMIT_S * 1000UL) { finish(); return; }
  if (!imu.dataReady()) return;
  imu.getAGMT();

  float f[3]; remapAccelRaw(imu.accX(), imu.accY(), imu.accZ(), f);
  float gx = imu.gyrX(), gy = imu.gyrY(), gz = imu.gyrZ();   // dps, raw is fine for a motion gate
  float an = sqrtf(f[0]*f[0]+f[1]*f[1]+f[2]*f[2]);
  float wn = sqrtf(gx*gx+gy*gy+gz*gz);
  bool still = fabsf(an - 9.81f) < 0.5f && wn < 3.0f;   // dps -- looser than the main sketch's rad/s gate
  bool moving = wn > 15.0f || fabsf(an - 9.81f) > 1.5f;

  static uint32_t lastPrint = 0;
  if (millis() - lastPrint > 300) {
    lastPrint = millis();
    Serial.print(F("accX:")); Serial.print(f[0],2);
    Serial.print(F(",accY:")); Serial.print(f[1],2);
    Serial.print(F(",accZ:")); Serial.print(f[2],2);
    Serial.print(F(",accNorm:")); Serial.println(an,2);
  }

  switch (state) {
    case WAIT_STILL:
      if (still) { stillStart = millis(); holdAccum[0]=holdAccum[1]=holdAccum[2]=0; holdN=0; state = HOLDING; }
      break;
    case HOLDING:
      if (!still) { state = WAIT_STILL; break; }   // moved before hold completed, restart
      holdAccum[0]+=f[0]; holdAccum[1]+=f[1]; holdAccum[2]+=f[2]; holdN++;
      if (millis() - stillStart > 1500 && holdN > 0) {
        float avg[3] = { holdAccum[0]/holdN, holdAccum[1]/holdN, holdAccum[2]/holdN };
        for (int i = 0; i < 3; ++i) { if (avg[i]<mn[i]) mn[i]=avg[i]; if (avg[i]>mx_[i]) mx_[i]=avg[i]; }
        nCaptured++;
        if (nCaptured == 1) {
          firstCaptureNorm = sqrtf(avg[0]*avg[0]+avg[1]*avg[1]+avg[2]*avg[2]);
        }
        Serial.print(F(">>> captured face ")); Serial.print(nCaptured); Serial.print(F("/")); Serial.print(N_FACES);
        Serial.print(F(":  ")); Serial.print(avg[0],2); Serial.print(F("  ")); Serial.print(avg[1],2);
        Serial.print(F("  ")); Serial.println(avg[2],2);
        if (nCaptured >= N_FACES) { finish(); break; }
        Serial.println(F(">>> MOVE TO A NEW FACE, then hold still <<<"));
        state = WAIT_MOTION;
      }
      break;
    case WAIT_MOTION:
      if (moving) state = WAIT_STILL;
      break;
    default: break;
  }
}

void finish() {
  state = FINISHED;
  float bias[3], radius[3];
  for (int i = 0; i < 3; ++i) {
    bias[i]   = 0.5f * (mx_[i] + mn[i]);
    radius[i] = 0.5f * (mx_[i] - mn[i]);
  }
  const float G = 9.80665f;
  float scale[3];
  for (int i = 0; i < 3; ++i) scale[i] = (radius[i] > 0.5f) ? (G / radius[i]) : 1.0f;

  Serial.println();
  Serial.println(F("================ CALIBRATION RESULT ================"));
  Serial.print(F("faces captured: ")); Serial.println(nCaptured);

  Serial.println();
  if (firstCaptureNorm > 0.5f) {
    float accelScaleCorr = G / firstCaptureNorm;
    Serial.println(F("PRIMARY OUTPUT -- paste into USQUE_AHRS.ino (replacing the existing line):"));
    Serial.print(F("  (from face 1, |accel|=")); Serial.print(firstCaptureNorm, 4);
    Serial.println(F(" -- this MUST have been the flat/level face for this number to be meaningful)"));
    Serial.print(F("static const float ACCEL_SCALE_CORR = "));
    Serial.print(accelScaleCorr, 4); Serial.println(F("f;"));
  } else {
    Serial.println(F("No usable first capture -- could not compute ACCEL_SCALE_CORR. Re-run."));
  }

  Serial.println();
  Serial.println(F("Per-axis DIAGNOSTIC only (NOT applied by USQUE_AHRS.ino -- it uses the single"));
  Serial.println(F("uniform ACCEL_SCALE_CORR above, not a per-axis bias+scale; see this sketch's"));
  Serial.println(F("header comment for why). Useful to see if your unit has a real per-axis offset:"));
  const char* axName[3] = {"X","Y","Z"};
  bool lowCoverage = false;
  for (int i = 0; i < 3; ++i) {
    Serial.print(F("  axis ")); Serial.print(axName[i]);
    Serial.print(F(": min=")); Serial.print(mn[i], 3);
    Serial.print(F(" max=")); Serial.print(mx_[i], 3);
    Serial.print(F(" (radius ")); Serial.print(radius[i], 3); Serial.print(F(", want ~9.8)"));
    Serial.print(F("  bias=")); Serial.print(bias[i], 3);
    Serial.print(F("  scale=")); Serial.print(scale[i], 4);
    if (radius[i] < 8.0f) { Serial.print(F("   <-- LOW: this axis never pointed both up AND down, re-run and include that face")); lowCoverage = true; }
    Serial.println();
  }
  if (lowCoverage) {
    Serial.println();
    Serial.println(F(">>> At least one axis never saw both its + and - extremes -- its"));
    Serial.println(F(">>> diagnostic bias/scale is unreliable. Re-run and make sure every axis"));
    Serial.println(F(">>> points straight up AND straight down at some point (all 6 faces)."));
  }
  Serial.println(F("======================================================"));
}
