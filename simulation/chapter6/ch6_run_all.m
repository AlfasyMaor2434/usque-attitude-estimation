function ch6_run_all(n_seeds)
%CH6_RUN_ALL  All simulations and figures of chapter 6.
%
%   ch6_run_all          full run (40 random seeds per manoeuvre, about 20 minutes)
%   ch6_run_all(3)       quick check
%
% Output: chapter6/figures/*.png and chapter6/ch6_results.json
%
% Part 1 (section 6.2)  why one sensor is not enough: gyro only / accelerometer only / fused
% Part 2 (section 6.3)  Euler-angle filters against quaternion filters:
%                       regular flight, one pitch-up to vertical in detail, and many
%                       repetitions with different noise for a range of pitch-up angles.
%
% Nothing is added to any filter output. Two errors are measured for every filter:
%   attitude error  the single rotation angle between the true and the estimated attitude
%   angle error     the largest error among the reported roll / pitch / yaw
% A filter can have a small attitude error and a large angle error: near 90 deg of pitch two
% different sets of Euler angles describe the same attitude.

if nargin < 1, n_seeds = 40; end
here = fileparts(mfilename('fullpath')); proj = fileparts(here);
addpath(proj, fullfile(proj, 'filters'), fullfile(proj, 'utils'), here);
figdir = fullfile(here, 'figures'); if ~exist(figdir, 'dir'), mkdir(figdir); end
set(0, 'DefaultFigureVisible', 'off');
p = ch6_params(); d2r = pi/180; R = struct();
col = struct('CF', [0.85 0.2 0.2], 'EKF', [0.2 0.7 0.2], 'QEKF', [0 0.2 0.9], 'UKF', [0 0.75 0.85], 'QUKF', [0.85 0 0.85]);
LOST_DEG = 30;       % attitude error above this = the filter has lost the attitude
WRONG_DEG = 45;      % reported angle off by more than this ...
WRONG_SEC = 1.0;     % ... for longer than this = the reported angles are wrong
DEF_PITCH = 80;      % angle error is only counted while the true |pitch| is below this (deg)

%% ===== Part 1: why sensor fusion is needed (single axis: roll) =====
dt = 0.01; t = 0:dt:60; N = numel(t);
burst = 2.0 * 0.5*(1 - cos(2*pi*(t - 40)/3)) .* (t >= 40 & t < 43);    % sideways acceleration, peak 2 m/s^2
vib   = 0.5 * (t >= 20 & t < 50);                                       % motor vibration, 0.5 m/s^2 (1-sigma)
[~, et, g0, a0, m0] = ch6_motion(t, p, 'roll_axis', burst);
seed1 = 1; best = inf;                                                  % a run with a typical gyro bias (0.3 deg/s)
for s = 1:60
    [~, ~, ~, b] = ch6_add_noise(g0, a0, m0, dt, p, s, vib);
    if abs(abs(b(1,1))/d2r - 0.3) < best, best = abs(abs(b(1,1))/d2r - 0.3); seed1 = s; end
end
[g, a, ~, b] = ch6_add_noise(g0, a0, m0, dt, p, seed1, vib);
roll_true = et(1,:)/d2r;
roll_gyro = cumsum([0, g(1,1:end-1)]*dt)/d2r;
roll_acc  = atan2(-a(2,:), -a(3,:))/d2r;
tau = 5; alpha = 1 - dt/tau; roll_fused = zeros(1, N);
for k = 2:N
    roll_fused(k) = alpha*(roll_fused(k-1) + g(1,k)*dt/d2r) + (1 - alpha)*roll_acc(k);
end
rmsf = @(x) sqrt(mean((x - roll_true).^2));
R.fusion = struct('gyro_bias_dps', abs(b(1,1))/d2r, 'vib_std', 0.5, 'burst_peak', 2.0, 'tau', tau, 'seed', seed1, ...
    'rms_gyro', rmsf(roll_gyro), 'rms_acc', rmsf(roll_acc), 'rms_fused', rmsf(roll_fused), ...
    'end_gyro', abs(roll_gyro(end) - roll_true(end)), 'max_acc', max(abs(roll_acc - roll_true)), 'max_fused', max(abs(roll_fused - roll_true)), ...
    'acc_std_quiet', std(roll_acc(t < 10)), 'acc_std_vib', std(roll_acc(t >= 20 & t < 30) - roll_true(t >= 20 & t < 30)), ...
    'acc_err_burst', max(abs(roll_acc(t >= 40 & t < 43) - roll_true(t >= 40 & t < 43))), 'fused_err_burst', max(abs(roll_fused(t >= 40 & t < 43) - roll_true(t >= 40 & t < 43))));

f = figure('Color', 'w', 'Position', [50 50 900 820]); tl = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
ttl = {'Gyroscope only: the integrated angle drifts', 'Accelerometer only: noisy with vibration, wrong during acceleration', 'Fused estimate (gyroscope + accelerometer)'};
est = {roll_gyro, roll_acc, roll_fused}; cc = {[0.85 0.1 0.1], [0.3 0.45 1], [0 0.6 0.1]}; lab = {'gyro integration', 'accelerometer angle', 'fused estimate'};
for i = 1:3
    ax = nexttile; hold(ax, 'on'); grid(ax, 'on');
    patch(ax, [20 50 50 20], [-40 -40 40 40], [0.9 0.9 0.9], 'EdgeColor', 'none', 'FaceAlpha', 0.6, 'DisplayName', 'motor vibration');
    patch(ax, [40 43 43 40], [-40 -40 40 40], [1 0.85 0.6], 'EdgeColor', 'none', 'FaceAlpha', 0.8, 'DisplayName', 'sideways acceleration');
    plot(ax, t, est{i}, 'Color', cc{i}, 'LineWidth', 1.0 + 0.5*(i ~= 2), 'DisplayName', lab{i});
    plot(ax, t, roll_true, 'k--', 'LineWidth', 1.6, 'DisplayName', 'true angle');
    ylim(ax, [-12 34]); ylabel(ax, 'Roll angle [deg]'); title(ax, ttl{i});
    legend(ax, 'Location', 'northeast', 'NumColumns', 2, 'FontSize', 8);
end
xlabel(tl, 'Time [s]');
exportgraphics(f, fullfile(figdir, 'ch6_fusion_need.png'), 'Resolution', 200); close(f);

%% ===== Part 2a: regular flight =====
dt = 0.01; t = 0:dt:30;
[qt, et, g0, a0, m0] = ch6_motion(t, p, 'regular');
acc_rms = zeros(5, n_seeds);
for s = 1:n_seeds
    [g, a, m] = ch6_add_noise(g0, a0, m0, dt, p, s);
    r = ch6_run_filters(t, g, a, m, p, qt, et);
    acc_rms(:, s) = sqrt(mean(r.att_err(:, t > 2).^2, 2));
end
R.regular = struct('names', {r.names}, 'max_pitch', max(abs(et(2,:)))/d2r, 'max_roll', max(abs(et(1,:)))/d2r, ...
    'rms_mean', mean(acc_rms, 2)', 'rms_max', max(acc_rms, [], 2)', 'n', n_seeds);

%% ===== Part 2b: pitch-up manoeuvre, many repetitions =====
t = 0:dt:20;
peaks = [30 60 80 85 88 90 92 95 100 120];
[PK, SD] = ndgrid(peaks, 1:n_seeds); nrun = numel(PK);
maxatt = zeros(nrun, 5); t_lost = zeros(nrun, 5); t_wrong = zeros(nrun, 5); sig_ekf = zeros(nrun, 1); pitch_ekf = zeros(nrun, 1);
cache = fullfile(here, sprintf('ch6_mc_cache_%dseeds.mat', n_seeds));    % delete this file to recompute
if exist(cache, 'file')
    load(cache, 'maxatt', 't_lost', 't_wrong', 'sig_ekf', 'pitch_ekf');
    fprintf('pitch-up manoeuvre: %d runs loaded from %s\n', nrun, cache);
else
fprintf('pitch-up manoeuvre: %d runs ...\n', nrun);
parfor j = 1:nrun
    [qt_, et_, g0_, a0_, m0_] = ch6_motion(t, p, 'pitchup', PK(j));
    [g_, a_, m_] = ch6_add_noise(g0_, a0_, m0_, dt, p, SD(j));
    r_ = ch6_run_filters(t, g_, a_, m_, p, qt_, et_);
    def = abs(et_(2,:)) < DEF_PITCH*d2r;
    maxatt(j,:)  = max(r_.att_err, [], 2)';
    t_lost(j,:)  = sum(r_.att_err > LOST_DEG, 2)' * dt;
    t_wrong(j,:) = sum(r_.ang_err(:, def) > WRONG_DEG, 2)' * dt;
    sig_ekf(j)   = max(max(r_.sigma_ekf([1 3], :)));
    pitch_ekf(j) = max(abs(r_.euler{2}(2,:)))/d2r;
end
save(cache, 'maxatt', 't_lost', 't_wrong', 'sig_ekf', 'pitch_ekf', 'peaks', 'n_seeds');
end
names = {'CF', 'EKF', 'QEKF', 'UKF', 'QUKF'};
lost_pct = zeros(numel(peaks), 5); wrong_pct = lost_pct; med_max = lost_pct; worst = lost_pct;
for i = 1:numel(peaks)
    m_ = PK(:) == peaks(i);
    lost_pct(i,:)  = 100*mean(maxatt(m_,:) > LOST_DEG, 1);
    wrong_pct(i,:) = 100*mean(t_wrong(m_,:) > WRONG_SEC, 1);
    med_max(i,:)   = median(maxatt(m_,:), 1);
    worst(i,:)     = max(maxatt(m_,:), [], 1);
end
near = PK(:) >= 90;
R.mc = struct('names', {names}, 'peaks', peaks, 'n_seeds', n_seeds, 'lost_deg', LOST_DEG, 'wrong_deg', WRONG_DEG, 'wrong_sec', WRONG_SEC, 'def_pitch', DEF_PITCH, ...
    'lost_pct', lost_pct, 'wrong_pct', wrong_pct, 'median_max_att', med_max, 'worst_att', worst, ...
    'lost_pct_ge90', 100*mean(maxatt(near,:) > LOST_DEG, 1), 'wrong_pct_ge90', 100*mean(t_wrong(near,:) > WRONG_SEC, 1), ...
    'lost_pct_lt90', 100*mean(maxatt(~near & PK(:) <= 85,:) > LOST_DEG, 1), 'worst_att_le85', max(maxatt(PK(:) <= 85,:), [], 1), ...
    'max_t_lost', max(t_lost, [], 1), 'median_t_lost_when_lost', arrayfun(@(i) median([t_lost(maxatt(:,i) > LOST_DEG, i); NaN], 'omitnan'), 1:5), ...
    'ekf_sigma_max_ge90_median', median(sig_ekf(near)), 'ekf_sigma_max_le85', max(sig_ekf(PK(:) <= 85)), ...
    'ekf_pitch_state_max', max(pitch_ekf), 'ekf_pitch_state_out_pct_ge90', 100*mean(pitch_ekf(near) > 91), 'n_runs_ge90', sum(near));

f = figure('Color', 'w', 'Position', [50 50 900 620]); tl = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
dat = {lost_pct, wrong_pct};
ttl = {sprintf('Runs in which the filter lost the attitude (attitude error above %d deg)', LOST_DEG), ...
       sprintf('Runs in which the reported roll / pitch / yaw were wrong (by more than %d deg for more than %g s)', WRONG_DEG, WRONG_SEC)};
x = 1:numel(peaks); mk = {'o', 's', '^', 'd', 'v'};
for k = 1:2
    ax = nexttile; hold(ax, 'on'); grid(ax, 'on');
    for i = 1:5
        plot(ax, x + (i - 3)*0.06, dat{k}(:,i), ['-' mk{i}], 'Color', col.(names{i}), 'MarkerFaceColor', col.(names{i}), 'LineWidth', 1.4, 'MarkerSize', 6, 'DisplayName', disp_name(names{i}));
    end
    xline(ax, find(peaks == 90), 'k:', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    set(ax, 'XTick', x, 'XTickLabel', string(peaks)); xlim(ax, [0.5, numel(peaks) + 0.5]); ylim(ax, [-5 105]); ylabel(ax, 'Share of runs [%]'); title(ax, ttl{k}, 'FontSize', 10);
    if k == 1, legend(ax, 'Location', 'northwest', 'FontSize', 8); end
end
xlabel(tl, 'Pitch-up angle of the manoeuvre [deg]  (90 = nose straight up)');
exportgraphics(f, fullfile(figdir, 'ch6_failure_rate.png'), 'Resolution', 200); close(f);

%% ===== Part 2c: two example runs in detail =====
% Example A: the usual outcome when the nose reaches vertical -- the Euler EKF keeps the
% attitude but its angles end up wrong. The first such run at a pitch-up of 90 deg.
jA = find(PK(:) == 90 & t_wrong(:,2) > WRONG_SEC & maxatt(:,2) < LOST_DEG, 1);
[qt, et, g0, a0, m0] = ch6_motion(t, p, 'pitchup', PK(jA));
[g, a, m] = ch6_add_noise(g0, a0, m0, dt, p, SD(jA));
r = ch6_run_filters(t, g, a, m, p, qt, et);
R.exampleA = struct('seed', SD(jA), 'peak', PK(jA), 'max_att', max(r.att_err, [], 2)', ...
    'ekf_pitch_state_max', max(abs(r.euler{2}(2,:)))/d2r, 'ekf_sigma_roll_max', max(r.sigma_ekf(1,:)), 'ekf_sigma_yaw_max', max(r.sigma_ekf(3,:)), ...
    'ekf_sigma_roll_before', median(r.sigma_ekf(1, t > 2 & t < 4)), 'ang_err_end', r.ang_err(:, end)', 'att_err_end', r.att_err(:, end)');

% -- figure: the Euler angles and the Euler EKF around the vertical
f = figure('Color', 'w', 'Position', [50 50 900 900]); tl = tiledlayout(4, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
idx = [2 1 3]; ylab = {'Pitch [deg]', 'Roll [deg]', 'Yaw [deg]'};
for k = 1:3
    ax = nexttile; hold(ax, 'on'); grid(ax, 'on');
    plot(ax, t, brk(et(idx(k),:)/d2r), 'k', 'LineWidth', 2.2, 'DisplayName', 'true angle');
    plot(ax, t, brk(r.euler{2}(idx(k),:)/d2r), 'Color', col.EKF, 'LineWidth', 1.3, 'DisplayName', 'EKF (Euler-angle state)');
    plot(ax, t, brk(r.euler{3}(idx(k),:)/d2r), '--', 'Color', col.QEKF, 'LineWidth', 1.3, 'DisplayName', 'QEKF (quaternion state)');
    ylabel(ax, ylab{k}); if k == 1, yline(ax, 90, 'k:', 'HandleVisibility', 'off'); legend(ax, 'Location', 'southeast', 'FontSize', 8); end
end
ax = nexttile; hold(ax, 'on'); grid(ax, 'on');
semilogy(ax, t, max(r.sigma_ekf(1,:), 1e-2), 'Color', col.EKF, 'LineWidth', 1.4, 'DisplayName', 'EKF: its own 1-sigma of roll');
semilogy(ax, t, max(r.sigma_ekf(3,:), 1e-2), ':', 'Color', [0 0.4 0], 'LineWidth', 1.4, 'DisplayName', 'EKF: its own 1-sigma of yaw');
semilogy(ax, t, min(1./abs(cos(et(2,:))), 1e4), 'k--', 'LineWidth', 1.0, 'DisplayName', '1 / cos(pitch): gain of the Euler-angle kinematics (clipped at 10^4)');
set(ax, 'YScale', 'log'); ylim(ax, [0.05 2e4]); ylabel(ax, '[deg]  /  [-]'); legend(ax, 'Location', 'northeast', 'FontSize', 8);
xlabel(tl, 'Time [s]');
exportgraphics(f, fullfile(figdir, 'ch6_euler_angles_at_90.png'), 'Resolution', 200); close(f);

% Example B: a run in which both Euler-angle Kalman filters lose the attitude (the run with the
% largest such error among the manoeuvres that reach vertical).
jv = find(PK(:) >= 90);
[~, ib] = max(min(maxatt(jv, 2), maxatt(jv, 4)));
jB = jv(ib);
[qt, et, g0, a0, m0] = ch6_motion(t, p, 'pitchup', PK(jB));
[g, a, m] = ch6_add_noise(g0, a0, m0, dt, p, SD(jB));
r = ch6_run_filters(t, g, a, m, p, qt, et);
R.exampleB = struct('seed', SD(jB), 'peak', PK(jB), 'max_att', max(r.att_err, [], 2)', 't_lost', sum(r.att_err > LOST_DEG, 2)'*dt, ...
    'att_err_end', r.att_err(:, end)');

% -- figure: attitude error of the five filters, whole run and zoom
f = figure('Color', 'w', 'Position', [50 50 900 640]); tl = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
tz = t(find(max(r.att_err([2 4], :), [], 1) > LOST_DEG, 1)); if isempty(tz), tz = 5; end
win = {[0 20], [tz - 0.6, tz + 0.9]}; ttl = {'Attitude error, whole manoeuvre', 'Zoom on the moment the Euler-angle filters lose the attitude'};
for k = 1:2
    ax = nexttile; hold(ax, 'on'); grid(ax, 'on');
    for i = [1 3 5 2 4]
        st = '-'; if any(i == [3 5]), st = '--'; end
        plot(ax, t, r.att_err(i,:), st, 'Color', col.(names{i}), 'LineWidth', 1.5, 'DisplayName', disp_name(names{i}));
    end
    xlim(ax, win{k}); ylim(ax, [0 185]); ylabel(ax, 'Attitude error [deg]'); title(ax, ttl{k}, 'FontSize', 10);
    if k == 1, legend(ax, 'Location', 'northeast', 'FontSize', 8); end
end
xlabel(tl, 'Time [s]');
exportgraphics(f, fullfile(figdir, 'ch6_attitude_error_at_90.png'), 'Resolution', 200); close(f);

%% ===== save =====
fid = fopen(fullfile(here, 'ch6_results.json'), 'w'); fprintf(fid, '%s', jsonencode(R, 'PrettyPrint', true)); fclose(fid);
disp(jsonencode(R, 'PrettyPrint', true));
end

function s = disp_name(n)
switch n
    case 'CF',   s = 'CF (complementary, Euler angles)';
    case 'EKF',  s = 'EKF (Euler-angle state)';
    case 'UKF',  s = 'UKF (Euler-angle state)';
    case 'QEKF', s = 'QEKF (quaternion state)';
    case 'QUKF', s = 'QUKF (quaternion state)';
end
end

function y = brk(y)
% break the line where a wrapped angle jumps, and wrap to +-180
y = mod(y + 180, 360) - 180;
y([false, abs(diff(y)) > 180]) = NaN;
end
