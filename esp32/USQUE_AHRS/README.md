# USQUE_AHRS: ESP32 + ICM-20948

This is the USQUE filter of `usque.py`, ported line by line to C++. It runs in real time on an ESP32 with an ICM-20948
9-DOF IMU, with no GPS.

* **`USQUE_AHRS.h`:** the filter. It is header-only, with no dependencies and no dynamic allocation. Eigenvalue clip,
  Cholesky and inverse are hand-written for the small fixed sizes.
* **`USQUE_AHRS.ino`:** the sketch, using the SparkFun ICM-20948 Arduino library. It does three things:
  * reads accel and gyro at the IMU rate and the magnetometer at its own rate
  * runs a prediction on every gyro sample
  * runs a measurement update at about 30 Hz, with the accelerometer low-passed by a causal 15 Hz Butterworth filter
* **`orientation_3d_viewer.html`:** a browser 3D view. It reads the quaternion from the serial line.
* **`plot_esp32_log.py`:** plots a saved serial log (`python plot_esp32_log.py <log.txt> [out_dir] [t_end_s]`).
* **Calibration sketches:** `../USQUE_AHRS_MagCalibration/` (sphere fit for the magnetometer) and
  `../USQUE_AHRS_AccelCalibration/`.

Results of the hardware tests are in `evaluation/hardware_tests/esp32/`:

* at rest, roll and pitch stayed within 0.03°
* a 90° turn was measured as 90.6°
* returning to the start pose was within 0.7°
* yaw drifted about 0.8°/min

## Wiring (I2C)

```
ICM-20948      ESP32
---------      -----
VIN / 3V3      3V3
GND            GND
SDA            GPIO 21
SCL            GPIO 22
AD0 -> GND     => address 0x68   (AD0 -> 3V3 => 0x69: set #define AD0_VAL 1)
```

## First run

1. **Axes.** Frames are NED world and FRD body. Held flat and still, `accel FRD` must read about `0 0 -9.81`. If it
   does not, fix `remapAccel()`, `remapGyro()` and `remapMag()` in the `.ino`.
2. **Magnetometer calibration.** Run `USQUE_AHRS_MagCalibration.ino` once, then paste `MAG_BIAS[3]` and
   `MAG_SCALE[3]` into `USQUE_AHRS.ino`.
3. **Local magnetic reference.** Set `MAG_INCLINATION_DEG` and `MAG_DECLINATION_DEG` for your site, from a WMM lookup.
   A residual heading bias is estimated online by the magnetic-bias state (`δψ_mag`).

## Parameters (`USQUE_AHRS::Params`)

All noise fields are variances, not standard deviations, exactly as in `usque.py`.

| Parameter | Meaning | Default |
|---|---|---|
| `gyro_noise_var` | gyro white noise, rad²/s | `9e-5` |
| `bias_rw_var` | gyro bias random walk, rad²/s³ | `1e-10` |
| `accel_noise_var` | accelerometer unit-vector variance | `1e-2` |
| `mag_noise_var` | magnetometer unit-vector variance | `0.05` |
| `use_mag_bias_state` | enables the 7th state, `δψ_mag` | `true` |
| `mag_bias_tau_s` / `mag_bias_sigma_ss_deg` | `δψ_mag` time constant and steady-state σ | `300 s` / `15°` |
| `mag_gate_mag_frac` / `mag_gate_incl_deg` | magnetic gate thresholds: field strength and inclination | `0.15` / `10°` |
| `max_mag_gate_inflation` | magnetometer R multiplier when the gate fires | `50` |
| `phys_dir_thresh_deg` / `max_phys_inflation` | accelerometer direction test | `5°` / `5` |
| `max_vib_inflation` / `max_vib_inflation_mag` | vibration inflation caps | `3` / `8` |
| `max_iae_inflation` | innovation-test inflation cap | `15` |

## Serial output

The output is `label:value` pairs, so the Arduino Serial Plotter can read it directly. It includes:

* roll, pitch, yaw and their 1σ values
* the gyro bias
* `psiMag_deg` and `sigPsiMag_deg`, the heading-bias estimate and its 1σ
* `accelInfl` and `magInfl`, the adaptive-R inflation of the last update (1 = fully trusted)
* `vibProxy`
* the IMU and magnetometer rates

Sending `N<heading_deg>` sets the heading manually, for example from a known landmark.
