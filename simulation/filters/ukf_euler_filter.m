function [euler_est, bias_est] = ukf_euler_filter(t, gyro, acc, mag, p)
%UKF_EULER_FILTER Unscented Kalman Filter using Euler Angles
% Returns Euler angles and Estimated Gyro Bias.

    N = numel(t);
    dt = t(2) - t(1);
    
    L = 6;              
    alpha = 1e-3;       
    ki    = 0;          
    beta  = 2;          
    
    lambda = alpha^2 * (L + ki) - L;
    c = L + lambda;     
    
    Wm = zeros(1, 2*L+1);
    Wc = zeros(1, 2*L+1);
    
    Wm(1) = lambda / c;
    Wc(1) = lambda / c + (1 - alpha^2 + beta);
    for i = 2:(2*L+1)
        Wm(i) = 1 / (2*c);
        Wc(i) = 1 / (2*c);
    end
    
    euler_est = zeros(3, N);
    bias_est  = zeros(3, N); % <--- New initialization
    
    [phi0, theta0, psi0] = imu_utils('accmag_to_euler', acc(:,1), mag(:,1), p);
    x = [phi0; theta0; psi0; 0; 0; 0];
    
    euler_est(:,1) = x(1:3);
    bias_est(:,1)  = x(4:6);
    
    P = diag([deg2rad(5)^2, deg2rad(5)^2, deg2rad(10)^2, ... 
              (0.1)^2, (0.1)^2, (0.1)^2]);
          
    q_angle = (p.gyro_noise_density_rad * 10)^2 * dt; 
    q_bias  = (p.gyro_bias_std_rad)^2 * dt;
    Q = diag([q_angle, q_angle, q_angle, q_bias, q_bias, q_bias]);
    
    r_acc = (p.accel_noise_density_g * 9.81 * 100)^2;
    r_mag = (p.mag_noise_std * 10)^2;
    R = diag([r_acc, r_acc, r_acc, r_mag, r_mag, r_mag]);
    
    for k = 2:N
        u = gyro(:, k);
        z_meas = [acc(:, k); mag(:, k)];
        
        %% --- Prediction ---
        S = robust_chol_local(P, L);
        sqrtP = sqrt(c) * S;
        
        X = zeros(L, 2*L+1);
        X(:,1) = x;
        for i = 1:L
            X(:, i+1)   = x + sqrtP(:,i);
            X(:, i+1+L) = x - sqrtP(:,i);
        end
        
        X_pred = zeros(size(X));
        for i = 1:(2*L+1)
            X_pred(:,i) = imu_utils('f_euler', X(:,i), u, dt);
            X_pred(1:3, i) = imu_utils('wrapToPi', X_pred(1:3, i));
        end
        
        x_pred = zeros(L,1);
        for i = 1:(2*L+1)
            x_pred = x_pred + Wm(i) * X_pred(:,i);
        end
        x_pred(1:3) = imu_utils('wrapToPi', x_pred(1:3)); 
        
        P_pred = Q; 
        for i = 1:(2*L+1)
            diff = X_pred(:,i) - x_pred;
            diff(1:3) = imu_utils('wrapToPi', diff(1:3));
            P_pred = P_pred + Wc(i) * (diff * diff');
        end
        
        %% --- Update ---
        n_z = 6; 
        Z_sig = zeros(n_z, 2*L+1);
        for i = 1:(2*L+1)
            Z_sig(:,i) = imu_utils('h_euler', X_pred(:,i), p);
        end
        
        z_pred = zeros(n_z, 1);
        for i = 1:(2*L+1)
            z_pred = z_pred + Wm(i) * Z_sig(:,i);
        end
        
        S_cov = R; 
        Pxz = zeros(L, n_z);
        for i = 1:(2*L+1)
            z_diff = Z_sig(:,i) - z_pred;
            x_diff = X_pred(:,i) - x_pred;
            x_diff(1:3) = imu_utils('wrapToPi', x_diff(1:3));
            S_cov = S_cov + Wc(i) * (z_diff * z_diff');
            Pxz   = Pxz   + Wc(i) * (x_diff * z_diff');
        end
        
        K = Pxz / S_cov;
        
        y = z_meas - z_pred;
        x = x_pred + K * y;
        x(1:3) = imu_utils('wrapToPi', x(1:3));
        
        P = P_pred - K * S_cov * K';
        P = (P + P') / 2;
        
        euler_est(:, k) = x(1:3);
        bias_est(:, k)  = x(4:6); % <--- Save estimated bias (Euler state: 4-6)
    end
end

function S = robust_chol_local(P, dim)
    try
        S = chol(P, 'lower');
    catch
        P = (P + P') / 2;
        epsilon = 1e-9;
        max_attempts = 10;
        for i = 1:max_attempts
            try
                P_temp = P + epsilon * eye(dim);
                S = chol(P_temp, 'lower');
                return; 
            catch
                epsilon = epsilon * 10; 
            end
        end
        S = diag(sqrt(abs(diag(P))));
    end
end