# USQUE_AHRS: MATLAB port and live phone test

This folder holds a MATLAB port of the USQUE filter. It is the same algorithm and uses the same variable names as the
ESP32 header, with MATLAB's own `eig`, `chol` and `mrdivide`. There is also a live test that streams a phone's IMU and
magnetometer into the filter through MATLAB Mobile. The phone is only the sensor; the filter and the 3D view run on the
computer.

## Files

* **`USQUE_AHRS.m`:** the filter (`classdef USQUE_AHRS < handle`).
* **`live_phone_usque.m`:** connects to the phone with `mobiledev`, then runs the filter live with a 3D view:
  * initializes from a still window, using TRIAD and a gyro bias seed
  * runs the predict/update loop
  * with `RECORD_RAW = true`, also writes `usque_phone_raw_<timestamp>.csv`
* **`replay_phone_usque.m`:** re-runs a raw recording offline, optionally with other parameters, for example
  `replay_phone_usque(f, struct('mag_noise_var', 0.3))`.
* **`replay_phone_raw.py`, `plot_phone_log.py`:** the same replay through `usque.py`, and plots of a live log.
* **`verify_port.py`, `verify_replay_dump.m`:** check that `USQUE_AHRS.m` and `usque.py` are the same filter (see
  below).
* **`verify_math.py`:** checks `usque.py` against the reference papers, with scipy's `Rotation` as an independent
  implementation.

## Equivalence with `usque.py`

```bash
python verify_port.py <usque_phone_raw_*.csv>      # MATLAB on PATH, or set MATLAB_EXE
```

The script replays one raw phone recording through both implementations and compares them at every update:
attitude, gyro bias, magnetic heading bias and the 1σ values from the covariance. It does this in four modes that
together cover every branch of the filter:

* live settings
* the heading-bias state switched on
* accelerometer-only updates
* a seeded start heading

Results on two recordings from `evaluation/hardware_tests/phone/data`:

| Recording | Updates | Largest attitude difference | Largest bias difference |
|---|---|---|---|
| erratic movement | 2,130 | 5.3·10⁻¹⁰° | 2.5·10⁻¹¹ °/s |
| rest and turns (20261001_153534) | 7,024 | 1.3·10⁻⁹° | 7.1·10⁻¹¹ °/s |

The differences are floating-point round-off: the two implementations compute the same filter.

The default parameters in `USQUE_AHRS.m` are the phone and ESP32 tuning. The INSANE results use
`evaluation/final_config.json`. The parameter values differ between platforms, but the logic is the same.

## Setup

1. Install **MATLAB Mobile** on the phone and sign in with the same MathWorks account as the computer.
2. In MATLAB Mobile open "Sensors" once, so it is ready to stream.
3. Run `live_phone_usque.m`. If `mobiledev` is undefined, search "MATLAB Mobile" in the Add-On Explorer
   and add the sensor support package once.
4. Hold the phone flat and still when prompted. The printed FRD accelerometer vector must be close to `(0, 0, -9.81)`.
   If it is not, adjust `remapDeviceToFRD()`.
