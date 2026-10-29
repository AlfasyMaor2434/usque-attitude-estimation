function p = icm20948_params()
% ICM-20948 Parameters definition
% Units: Rad/s for Gyro, m/s^2 for Accel, Gauss for Mag

%% 1. Gravity & Constants
p.g = 9.81; % m/s^2

%% 2. Gyroscope
p.gyro_range_dps    = 2000;
p.gyro_range_rad    = p.gyro_range_dps * (pi/180);

% Noise Density from Datasheet: ~0.005 deg/s/sqrt(Hz)
% Note: This needs to be converted to sigma depending on dt in the simulation loop!
p.gyro_noise_density_dps = 0.005; 
p.gyro_noise_density_rad = p.gyro_noise_density_dps * (pi/180);

% Bias Instability (Random Walk)
p.gyro_bias_std_dps = 0.1; 
p.gyro_bias_std_rad = p.gyro_bias_std_dps * (pi/180);

%% 3. Accelerometer
p.accel_range_g     = 4; 
% Noise Density: 100 ug/sqrt(Hz) = 0.0001 g/sqrt(Hz)
p.accel_noise_density_g = 0.0001; 
p.accel_bias_std_g      = 0.01;

%% 4. Magnetometer
% Earth magnetic field vector in NED (North-East-Down)
% For Northern Hemisphere, Z component (Down) is POSITIVE.
% Approx values: North=0.3 Gauss, Down=0.5 Gauss
p.mag_field_ned     = [0.3; 0; 0.5]; % Corrected for Northern Hemisphere
p.mag_noise_std     = 0.002;          % [Gauss]
p.mag_bias_std      = 0.01;           % [Gauss]

end