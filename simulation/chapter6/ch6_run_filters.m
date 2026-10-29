function r = ch6_run_filters(t, gyro, acc, mag, p, quat_true, euler_true)
%CH6_RUN_FILTERS  Run the five filters on one data set and measure two kinds of error.
%   r.names         {'CF','EKF','QEKF','UKF','QUKF'}  (EKF, UKF = Euler-angle state)
%   r.euler{i}      3xN roll / pitch / yaw reported by filter i (rad)
%   r.att_err(i,:)  total attitude error (deg): the single rotation angle between the true
%                   attitude and the attitude the filter's output represents
%   r.ang_err(i,:)  largest error among the three reported angles (deg) -- what a user who
%                   reads roll / pitch / yaw from the filter sees
%   r.sigma_ekf     3xN the Euler EKF's own 1-sigma of roll / pitch / yaw (deg)

r.names = {'CF', 'EKF', 'QEKF', 'UKF', 'QUKF'};
e_cf = cf_filter(t, gyro, acc, mag, p);
[e_ekf, ~, s_ekf] = ekf_euler_filter(t, gyro, acc, mag, p);
[q_qekf, e_qekf]  = qekf_filter(t, gyro, acc, mag, p);
e_ukf = ukf_euler_filter(t, gyro, acc, mag, p);
[q_qukf, e_qukf]  = qukf_filter(t, gyro, acc, mag, p);

r.euler = {e_cf, e_ekf, e_qekf, e_ukf, e_qukf};
quats   = {e2q(e_cf), e2q(e_ekf), q_qekf, e2q(e_ukf), q_qukf};
r.sigma_ekf = s_ekf * 180/pi;

N = numel(t); r.att_err = zeros(5, N); r.ang_err = zeros(5, N);
for i = 1:5
    d = sum(quats{i} .* quat_true, 1);
    r.att_err(i,:) = 2*acos(min(1, abs(d))) * 180/pi;
    de = mod(r.euler{i} - euler_true + pi, 2*pi) - pi;
    r.ang_err(i,:) = max(abs(de), [], 1) * 180/pi;
end
r.att_err(~isfinite(r.att_err)) = 180;       % a filter that returns NaN has lost the attitude
r.ang_err(~isfinite(r.ang_err)) = 180;
end

function q = e2q(e)
q = zeros(4, size(e, 2));
for k = 1:size(e, 2), q(:,k) = imu_utils('eulerZYX2quat', e(:,k)); end
end
