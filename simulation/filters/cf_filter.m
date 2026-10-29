function euler_est = cf_filter(t, gyro, acc, mag, p)
%CF_FILTER Implementation of a simple Complementary Filter for Attitude Estimation
%
% Inputs:
%   t     - time vector [1xN]
%   gyro  - gyroscope measurements [3xN] (rad/s)
%   acc   - accelerometer measurements [3xN] (m/s^2)
%   mag   - magnetometer measurements [3xN] (Gauss)
%   p     - parameter structure
%
% Output:
%   euler_est - Estimated Euler angles [3xN] (rad) [roll; pitch; yaw]

    N = numel(t);
    dt = t(2) - t(1);
    
    % הקצאת זיכרון
    euler_est = zeros(3, N);
    
    %% 1. אתחול (Initialization)
    % שימוש במדידה הראשונה של מד התאוצה והמגנטומטר לקביעת מצב התחלתי
    [phi0, theta0, psi0] = imu_utils('accmag_to_euler', acc(:,1), mag(:,1), p);
    euler_est(:,1) = [phi0; theta0; psi0];
    
    %% 2. פרמטר ה-Gain (Alpha)
    % alpha קובע כמה אנחנו סומכים על הג'יירו לעומת המד-תאוצה/מגנטומטר.
    % alpha קרוב ל-1 (למשל 0.98) אומר: סמוך בעיקר על הג'יירו (High Pass),
    % ותקן לאט בעזרת המד-תאוצה (Low Pass).
    alpha = 0.98; 
    
    %% 3. לולאת ריצה (Main Loop)
    for k = 2:N
        % שליפת המצב הקודם
        prev_phi   = euler_est(1, k-1);
        prev_theta = euler_est(2, k-1);
        prev_psi   = euler_est(3, k-1);
        
        % --- שלב א': חיזוי בעזרת ג'יירו (Prediction) ---
        % הג'יירו מודד מהירות זוויתית בגוף (p, q, r). צריך להמיר לקצבי אוילר.
        w_body = gyro(:, k);
        wx = w_body(1); wy = w_body(2); wz = w_body(3);
        
        % מטריצת המרה (Kinematic Jacobian)
        % (שים לב: יש כאן סינגולריות ב-Pitch=90, שזה החיסרון של אוילר)
        sinPhi = sin(prev_phi); cosPhi = cos(prev_phi);
        tanTheta = tan(prev_theta); secTheta = 1/cos(prev_theta);
        
        phi_dot   = wx + wy * sinPhi * tanTheta + wz * cosPhi * tanTheta;
        theta_dot =      wy * cosPhi            - wz * sinPhi;
        psi_dot   =      wy * sinPhi * secTheta + wz * cosPhi * secTheta;
        
        % אינטגרציה (חיזוי)
        pred_phi   = prev_phi   + phi_dot * dt;
        pred_theta = prev_theta + theta_dot * dt;
        pred_psi   = prev_psi   + psi_dot * dt;
        
        % --- שלב ב': מדידה ממד-תאוצה ומגנטומטר (Measurement) ---
        % שימוש בפונקציית העזר הקיימת כדי לחשב זוויות מוחלטות
        [meas_phi, meas_theta, meas_psi] = imu_utils('accmag_to_euler', acc(:,k), mag(:,k), p);
        
        % --- שלב ג': מיזוג (Fusion) ---
        % הנוסחה הקלאסית: est = alpha * pred + (1-alpha) * meas
        % אבל! עבור זוויות, חיבור רגיל נכשל במעבר מחזוריות (למשל 179 ומינוס 179).
        % לכן נשתמש בנוסחה מבוססת הפרש: est = pred + (1-alpha) * error
        
        % חישוב השגיאה תוך טיפול במחזוריות (Shortest angular distance)
        err_phi   = imu_utils('wrapToPi', meas_phi   - pred_phi);
        err_theta = imu_utils('wrapToPi', meas_theta - pred_theta);
        err_psi   = imu_utils('wrapToPi', meas_psi   - pred_psi);
        
        % עדכון סופי
        euler_est(1, k) = pred_phi   + (1 - alpha) * err_phi;
        euler_est(2, k) = pred_theta + (1 - alpha) * err_theta;
        euler_est(3, k) = pred_psi   + (1 - alpha) * err_psi;
    end
end