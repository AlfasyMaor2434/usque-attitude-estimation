function [quat_est, euler_est, bias_est] = qekf_filter(t, gyro, acc, mag, p)
%QEKF_FILTER Extended Kalman Filter using Quaternions
% Returns Quaternions, Euler angles, and Estimated Bias.

    N = numel(t);
    dt = t(2) - t(1);
    
    n_states = 7; 
    
    quat_est  = zeros(4, N);
    euler_est = zeros(3, N);
    bias_est  = zeros(3, N); % <--- New initialization
    
    [phi0, theta0, psi0] = imu_utils('accmag_to_euler', acc(:,1), mag(:,1), p);
    q_init = imu_utils('eulerZYX2quat', [phi0; theta0; psi0]);
    
    x = [q_init; 0; 0; 0];
    
    quat_est(:,1)  = x(1:4);
    euler_est(:,1) = [phi0; theta0; psi0];
    bias_est(:,1)  = x(5:7);
    
    P = eye(n_states) * 0.01;
    P(5:7, 5:7) = eye(3) * (0.1)^2; 
    
    q_quat = (p.gyro_noise_density_rad * 5)^2 * dt; 
    q_bias = (p.gyro_bias_std_rad)^2 * dt;
    Q = diag([repmat(q_quat, 1, 4), repmat(q_bias, 1, 3)]);
    
    r_acc = (p.accel_noise_density_g * 9.81 * 100)^2;
    r_mag = (p.mag_noise_std * 10)^2;
    R = diag([r_acc, r_acc, r_acc, r_mag, r_mag, r_mag]);
    
    for k = 2:N
        u = gyro(:, k);
        z_meas = [acc(:, k); mag(:, k)];
        
        %% --- Prediction ---
        x_prev = x;
        
        x_pred = imu_utils('f_quat', x_prev, u, dt);
        
        fun_f = @(xx) imu_utils('f_quat', xx, u, dt);
        F = imu_utils('numerical_jacobian', fun_f, x_prev);
        
        P_pred = F * P * F' + Q;
        
        %% --- Correction ---
        z_pred = imu_utils('h_quat', x_pred, p);
        y = z_meas - z_pred;
        
        fun_h = @(xx) imu_utils('h_quat', xx, p);
        H = imu_utils('numerical_jacobian', fun_h, x_pred);
        
        S = H * P_pred * H' + R;
        K = P_pred * H' / S;
        
        x = x_pred + K * y;
        
        % Normalize Quaternion
        q_norm = norm(x(1:4));
        if q_norm > 1e-6
            x(1:4) = x(1:4) / q_norm;
        end
        
        P = (eye(n_states) - K * H) * P_pred;
        P = (P + P') / 2;
        
        quat_est(:, k) = x(1:4);
        euler_est(:, k) = imu_utils('quat2eulerZYX', x(1:4));
        bias_est(:, k)  = x(5:7); % <--- Save bias (Quat state: 5-7)
    end
end