# USQUE: Real-Time Attitude Estimation for Drones

Final project by **Maor Alphassy**. Supervisor: **Uzi Chester**.

A drone has to know its orientation (roll, pitch and yaw) at every moment. A gyroscope alone drifts, and an
accelerometer and a magnetometer are noisy and easily disturbed. This project fuses all three sensors of a 9-DOF IMU
with **USQUE** (UnScented QUaternion Estimator, Crassidis & Markley 2003), an unscented Kalman filter that keeps the
attitude as a quaternion and filters a 3-dimensional error state.

* **State:** a nominal quaternion plus an error state of attitude error (generalized Rodrigues parameters, 3), gyro
  bias (3) and an optional magnetic heading-bias state (1). That gives 15 sigma points.
* **Prediction** runs on every gyro sample. **Update** with the accelerometer and magnetometer runs at 30 Hz.
* **Initialization** uses TRIAD, with gravity as the primary vector and the magnetic field as the secondary one.
* **Adaptive measurement noise (R):** four mechanisms raise the noise of suspicious measurements in real time:
  * an innovation test per channel (IAE)
  * a physical direction test on the accelerometer
  * a vibration measure
  * a magnetic gate on field strength and inclination

## Results

The filter was tested on real flights from the [INSANE data set](https://arxiv.org/abs/2210.09114), with the final
settings in `evaluation/final_config.json`:

| Flights | Reference | Error < 1° | Error < 3° | Median | RMSE |
|---|---|---|---|---|---|
| indoor_1–3 | motion capture, full attitude | 77% | 100% | 0.69° | **0.88°** |
| mars_5–7 | dual-RTK heading, yaw | 47% | 87% | 1.06° | **2.18°** |

* The RTK heading reference is itself noisy, about 1°, so part of the outdoor error belongs to the reference.
* Flights that were not used for tuning, and the ablations (each adaptive mechanism switched off), are in
  `evaluation/run_extra_tests.py` and `evaluation/extra_tests.csv`.

The same filter was also run in real time:

* **ESP32 with an ICM-20948 IMU** (`esp32/`):
  * at rest, roll and pitch stayed within 0.03°
  * a 90° turn was measured as 90.6°
  * returning to the start pose was within 0.7°
  * yaw drifted about 0.8°/min
* **Phone sensors streamed to MATLAB** (`matlab/`)

## Repository layout

```
usque.py               the filter (predict / update / adaptive R / TRIAD)
main_usque.py          runs the filter on INSANE flights, scores accuracy and consistency, plots
data_loader*.py        INSANE loading, frames, calibration, preprocessing
metrics*.py            error and consistency metrics (qAE, RMSE, NEES, NIS) and plots
attitude_reference.py  independent roll/pitch reference checks
project_utils.py       quaternion math
data_path.py           where the data set is (INSANE_DATA_DIR)

evaluation/            final results (book chapter 5)
  usque_eval.py          shared evaluation core
  run_final_usque.py     final run on indoor_1-3 and mars_5-7  -> summary.csv, figures/
  run_extra_tests.py     ablations, held-out flights, initialization source
  tune_final_usque.py    parameter sweep (tuning_log.csv)
  mars_tuning/           yaw tuning and diagnostics against the RTK heading
  lpf_study/             does the accelerometer low-pass help, and does a causal filter match?
  sensor_compare/        PX4 IMU vs LSM9DS1 IMU on the indoor flights
  hardware_tests/        ESP32 and phone recordings and their evaluation
analysis/              earlier studies: magnetometer bias and calibration, vibration spectra, static noise

esp32/                 the filter on an ESP32 + ICM-20948 (Arduino IDE), with calibration sketches
matlab/                MATLAB port of the filter, live phone streaming, and the checks against usque.py
simulation/            MATLAB simulation (book chapter 6): single sensors vs fusion, Euler vs quaternion filters
```

## Getting started

```bash
pip install -r requirements.txt
```

1. Download the INSANE data set and point to the folder that holds the flight folders (`indoor_1_sensors`,
   `mars_5_sensors`, ...). The folder must be named `9DOF`, because `data_loader.py` looks for it by name. Either set
   `INSANE_DATA_DIR` to that folder, or place it at `data/9DOF` inside the repository.
2. Reproduce the results:

```bash
cd evaluation
python run_final_usque.py      # summary.csv, params_used.json, figures/
python run_extra_tests.py      # extra_tests.csv, ablation / held-out figures
```

`python main_usque.py` runs the filter directly, with the site and mode chosen at the top of the file.

### Hardware

* **ESP32:** open `esp32/USQUE_AHRS/USQUE_AHRS.ino` in the Arduino IDE. Wiring, calibration and the serial output are
  in `esp32/USQUE_AHRS/README.md`.
* **Phone:** run `matlab/USQUE_AHRS_MATLAB/live_phone_usque.m`. It needs MATLAB Mobile; see that folder's README.
* **Checks:** both ports are checked against `usque.py`. The MATLAB port is replayed sample by sample by
  `matlab/USQUE_AHRS_MATLAB/verify_port.py`. The largest attitude difference over all filter branches is about 1e-9°,
  which is floating-point round-off.

### Simulation

In MATLAB, from `simulation/chapter6`, run:

```matlab
ch6_run_all        % full run, 40 seeds, about 20 minutes
ch6_run_all(3)     % quick check
```

## Notes

* Frames: NED world, FRD body. Quaternions are Hamilton, `[w x y z]`, body to world.
* The tuning values differ between platforms: INSANE uses `evaluation/final_config.json`, and the ESP32 and phone
  use the defaults in their own files. The filter logic is the same everywhere.
* Some code comments mention earlier designs (an ESKF/UKF hybrid and an older ESP32 filter). Those are not part of
  this repository.

## References

* J. L. Crassidis and F. L. Markley, "Unscented Filtering for Spacecraft Attitude Estimation," *Journal of Guidance,
  Control, and Dynamics*, 26(4), 536–542, 2003. doi:10.2514/2.5102
* F. L. Markley, "Attitude Error Representations for Kalman Filtering," *Journal of Guidance, Control, and Dynamics*,
  26(2), 311–317, 2003.
* C. Brommer et al., "INSANE: Cross-Domain UAV Data Sets with Increased Number of Sensors for developing Advanced and
  Novel Estimators," arXiv, 2022. doi:10.48550/arXiv.2210.09114
