function [quat, euler, gyro, acc, mag] = ch6_motion(t, p, kind, arg)
%CH6_MOTION  True motion and ideal sensor outputs for the chapter 6 simulations.
%   kind = 'regular'   gentle flight, pitch and roll within about +-40 deg (no argument)
%   kind = 'pitchup'   pitch up from level by ARG degrees, rotate about the body x and z axes
%                      while pitched up, then pitch back down. ARG = 90 puts the nose
%                      straight up; ARG > 90 goes past vertical. No linear acceleration,
%                      so the only difference between runs is how close they get to 90 deg.
%   kind = 'roll_axis' single-axis roll manoeuvre for the sensor-fusion demonstration;
%                      ARG = 1xN lateral acceleration in the navigation frame (m/s^2)
% Outputs: quat 4xN (body to NED, w first), euler 3xN [roll; pitch; yaw] (rad),
%          gyro 3xN (rad/s), acc 3xN specific force (m/s^2), mag 3xN (Gauss).

dt = t(2) - t(1); N = numel(t); d2r = pi/180;
w = zeros(3, N); a_n = zeros(3, N);

switch kind
    case 'regular'
        w(1,:) = 5*d2r * sin(0.2*t);
        w(2,:) = 3*d2r * sin(0.15*t + 0.5);
        w(3,:) = 4*d2r * sin(0.1*t - 0.3);
    case 'pitchup'
        r = arg/4 * d2r;                         % pitch up over 4 s
        w(2, t >= 1  & t < 5)  = r;
        w(1, t >= 6  & t < 12) = 20*d2r;         % roll about the body x axis while pitched up
        w(3, t >= 6  & t < 12) = 10*d2r;         % and a yaw rate about the body z axis
        w(2, t >= 13 & t < 17) = -r;
    case 'roll_axis'
        % roll to +20 deg at 10-14 s, back to 0 at 30-34 s (smooth rate profile)
        prof = @(t0) (20*d2r/4) * (1 - cos(2*pi*(t - t0)/4)) .* (t >= t0 & t < t0 + 4);
        w(1,:) = prof(10) - prof(30);
        a_n(2,:) = arg;
    otherwise
        error('ch6_motion: unknown kind "%s"', kind);
end

% integrate the attitude exactly over each step (same rule as simulate_true_motion.m)
quat = zeros(4, N); quat(:,1) = [1; 0; 0; 0];
for k = 2:N
    wk = w(:,k-1); th = norm(wk)*dt; q = quat(:,k-1);
    if th > 1e-12
        ax = wk/norm(wk); dq = [cos(th/2); ax*sin(th/2)];
        q = [q(1)*dq(1) - q(2)*dq(2) - q(3)*dq(3) - q(4)*dq(4);
             q(1)*dq(2) + q(2)*dq(1) + q(3)*dq(4) - q(4)*dq(3);
             q(1)*dq(3) - q(2)*dq(4) + q(3)*dq(1) + q(4)*dq(2);
             q(1)*dq(4) + q(2)*dq(3) - q(3)*dq(2) + q(4)*dq(1)];
    end
    quat(:,k) = q/norm(q);
end
euler = imu_utils('quat2eulerZYX', quat);
gyro = w;

acc = zeros(3, N); mag = zeros(3, N); g_n = [0; 0; p.g];
for k = 1:N
    R_nb = imu_utils('quat2R', quat(:,k));
    acc(:,k) = R_nb' * (a_n(:,k) - g_n);        % specific force
    mag(:,k) = R_nb' * p.mag_field_ned;
end
end
