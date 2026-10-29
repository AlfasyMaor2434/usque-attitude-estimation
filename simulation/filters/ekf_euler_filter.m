function [euler_est, bias_est, sigma_est] = ekf_euler_filter(t, gyro, acc, mag, p)
%EKF_EULER_FILTER Extended Kalman Filter using Euler Angles
% Returns Euler angles and Estimated Gyro Bias.
% Optional third output: the filter's own 1-sigma of roll / pitch / yaw (rad), from diag(P).

    N = numel(t);
    dt = t(2) - t(1);
    
    n_states = 6; 
    
    euler_est = zeros(3, N);
    bias_est  = zeros(3, N); % <--- New initialization
    sigma_est = zeros(3, N);
    
    % Initialization
    [phi0, theta0, psi0] = imu_utils('accmag_to_euler', acc(:,1), mag(:,1), p);
    bg0 = [0; 0; 0]; 
    x = [phi0; theta0; psi0; bg0];
    
    euler_est(:,1) = x(1:3);
    bias_est(:,1)  = x(4:6);
    
    P = diag([deg2rad(5)^2, deg2rad(5)^2, deg2rad(10)^2, ... 
              (0.1)^2, (0.1)^2, (0.1)^2]);

    q_angle = (p.gyro_noise_density_rad * 10)^2 * dt; 
    q_bias  = (p.gyro_bias_std_rad)^2 * dt;
    Q = diag([q_angle, q_angle, q_angle, q_bias, q_bias, q_bias]);
    sigma_est(:,1) = sqrt(diag(P(1:3,1:3)));
    
    r_acc = (p.accel_noise_density_g * 9.81 * 100)^2;
    r_mag = (p.mag_noise_std * 10)^2;
    R = diag([r_acc, r_acc, r_acc, r_mag, r_mag, r_mag]);

    for k = 2:N
        u = gyro(:, k);
        z_meas = [acc(:, k); mag(:, k)];
        
        %% --- Prediction ---
        x_prev = x;
        
        x_pred = imu_utils('f_euler', x_prev, u, dt);
        x_pred(1:3) = imu_utils('wrapToPi', x_pred(1:3));
        
        fun_f = @(xx) imu_utils('f_euler', xx, u, dt);
        F = imu_utils('numerical_jacobian', fun_f, x_prev);
        
        P_pred = F * P * F' + Q;
        
        %% --- Correction ---
        z_pred = imu_utils('h_euler', x_pred, p);
        
        y = z_meas - z_pred;
        
        fun_h = @(xx) imu_utils('h_euler', xx, p);
        H = imu_utils('numerical_jacobian', fun_h, x_pred);
        
        S = H * P_pred * H' + R;
        K = P_pred * H' / S; 
        
        x = x_pred + K * y;
        
        % Normalize angles
        x(1:3) = imu_utils('wrapToPi', x(1:3));
        
        P = (eye(n_states) - K * H) * P_pred;
        
        euler_est(:, k) = x(1:3);
        bias_est(:, k)  = x(4:6); % <--- Save estimated bias (Euler state: 4-6)
        sigma_est(:, k) = sqrt(abs(diag(P(1:3,1:3))));
    end
end