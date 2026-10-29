function p = ch6_params()
%CH6_PARAMS  Sensor parameters for the chapter 6 simulations.
% Same fields as icm20948_params (the filters read their Q and R from them), with sensor
% errors closer to a real low-cost IMU: a CONSTANT bias per run plus a slow random walk,
% instead of a fast random walk that reaches several degrees within 30 s.
%
% Gyro bias 0.3 deg/s (1-sigma) is the order of the uncalibrated ICM-20948 measured in
% chapter 5 (0.2 - 1.1 deg/s).

p = icm20948_params();
d2r = pi/180;

% constant bias, drawn once per run (1-sigma)
p.gyro_bias0_rad  = 0.3 * d2r;      % rad/s
p.accel_bias0     = 0.02;           % m/s^2  (about 0.1 deg of tilt)
p.mag_bias0       = 0.003;          % Gauss  (about 0.6 deg of heading)

% slow random walk on top of it (per sqrt(second))
p.gyro_bias_rw_rad = 0.002 * d2r;   % rad/s/sqrt(s)
p.accel_bias_rw    = 0.001;         % m/s^2/sqrt(s)
p.mag_bias_rw      = 0.0002;        % Gauss/sqrt(s)
end
