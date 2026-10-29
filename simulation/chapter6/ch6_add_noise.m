function [gyro_meas, acc_meas, mag_meas, bias_gyro] = ch6_add_noise(gyro_true, acc_true, mag_true, dt, p, seed, acc_extra_std)
%CH6_ADD_NOISE  White noise + constant bias + slow bias random walk, with a selectable seed.
% (add_imu_noise.m uses one fixed seed, so every run sees the same noise.)
%   seed           random seed of this run
%   acc_extra_std  optional 1xN extra accelerometer noise (m/s^2), e.g. motor vibration

if nargin < 7, acc_extra_std = zeros(1, size(gyro_true, 2)); end
N  = size(gyro_true, 2);
rs = RandStream('mt19937ar', 'Seed', seed);
isd = 1/sqrt(dt);

wn_gyro = p.gyro_noise_density_rad * isd * randn(rs, 3, N);
wn_acc  = p.accel_noise_density_g * 9.81 * isd * randn(rs, 3, N) + acc_extra_std .* randn(rs, 3, N);
wn_mag  = p.mag_noise_std * randn(rs, 3, N);

bias_gyro = p.gyro_bias0_rad * randn(rs, 3, 1) + cumsum(p.gyro_bias_rw_rad * sqrt(dt) * randn(rs, 3, N), 2);
bias_acc  = p.accel_bias0    * randn(rs, 3, 1) + cumsum(p.accel_bias_rw    * sqrt(dt) * randn(rs, 3, N), 2);
bias_mag  = p.mag_bias0      * randn(rs, 3, 1) + cumsum(p.mag_bias_rw      * sqrt(dt) * randn(rs, 3, N), 2);

gyro_meas = gyro_true + wn_gyro + bias_gyro;
acc_meas  = acc_true  + wn_acc  + bias_acc;
mag_meas  = mag_true  + wn_mag  + bias_mag;
end
