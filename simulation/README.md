# Simulation (book chapter 6)

This is a MATLAB simulation of a 9-DOF IMU with a constant gyro bias plus a random walk, white noise, motor vibration
and acceleration bursts. It answers two questions.

## 1. Why fuse the sensors?

On a single axis, a gyroscope alone, an accelerometer alone and the fused estimate are compared:

| Estimate | RMS error | Why |
|---|---|---|
| gyro only | 10.9° | it drifts |
| accelerometer only | 2.6° | it is noisy, and the vibration and bursts disturb it |
| fused | 1.3° | |

## 2. Euler angles or quaternions?

Five filters are compared over 40 random seeds per manoeuvre:

* complementary filter (CF)
* EKF and UKF on Euler angles
* EKF and UKF on quaternions (QEKF, QUKF)

The manoeuvres pitch up to angles between 30° and 120°.

* In regular flight all five filters reach about the same accuracy.
* When the pitch passes 90°, the Euler-angle UKF lost the attitude in 37.5% of the runs and the Euler EKF in 3%.
* The quaternion filters never lost it.

## Running

```matlab
cd chapter6
ch6_run_all        % full run, about 20 minutes
ch6_run_all(3)     % quick check with 3 seeds
```

The output goes to `chapter6/figures/*.png` and `chapter6/ch6_results.json`. The Monte Carlo runs are cached in
`ch6_mc_cache_<n>seeds.mat`; delete that file to recompute.

## Files

* `chapter6/ch6_run_all.m`: all runs and figures.
* `chapter6/ch6_params.m`: sensor parameters. It builds on `icm20948_params.m`.
* `chapter6/ch6_motion.m`: the true motion.
* `chapter6/ch6_add_noise.m`: the noise model.
* `chapter6/ch6_run_filters.m`: runs the five filters.
* `filters/`: `cf_filter`, `ekf_euler_filter`, `qekf_filter`, `ukf_euler_filter`, `qukf_filter`.
* `utils/imu_utils.m`: shared helpers (quaternion and Euler conversions, initial angles from the accelerometer and magnetometer).
