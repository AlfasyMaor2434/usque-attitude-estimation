function live_phone_usque()
% LIVE_PHONE_USQUE  Stream a phone's IMU/magnetometer via MATLAB Mobile, run
% USQUE_AHRS.m (port of arduino/USQUE_AHRS/USQUE_AHRS.h / usque.py) on it
% live, and show a rotating 3D board. Run with:  >> live_phone_usque
%
% The live loop runs on a background timer, so the function returns and the
% Command Window stays free (MATLAB Online only draws figures while MATLAB
% is idle). Stop by CLOSING THE FIGURE WINDOW -- that stops the timer,
% releases the phone and saves the log. If the figure is already gone but
% something is still running:  stop(timerfind('Name','live_phone_usque'))
% (The old script version did `clear` at the top, which orphaned the
% previous run's mobiledev connection without closing it; stale connections
% then made the next mobiledev(...) call stall or report "Connection to
% this device already exists".)
%
% Outputs, next to this file:
%   usque_phone_log_<timestamp>.txt  -- everything printed to the Command Window
%   usque_phone_raw_<timestamp>.csv  -- raw gyro/accel/mag + app orientation, for
%                                       offline replay (RECORD_RAW)
%   phone_live_state.txt             -- latest state, for orientation_3d_viewer.html's
%                                       "Connect to file (MATLAB)" button
%
% SETUP: MATLAB Mobile app on the phone, signed in to the same MathWorks
% account, open on its Sensors screen.
%
% SEQUENCE: connect -> magnetometer calibration (MAG_CAL_MODE 'flat': spin
% the phone one full turn flat on the table; '3d': tumble it through all
% orientations; saved to phone_mag_cal.mat) -> put
% it down flat and still (waits until it is) -> boot still window -> 3D view.
%
% AXIS REMAP: remapDeviceToFRD assumes phone flat, screen up, top edge =
% forward. The boot print of the FRD accel vector must read ~(0, 0, -9.81).

DEVICE_NAME         = 'Xiaomi 15T Pro';
MAG_INCLINATION_DEG = 48.932;   % WMM at your site (only used as a sanity reference, see MAG_REF_FROM_BOOT)
MAG_DECLINATION_DEG = 5.006;    % WMM at your site
MAG_REF_FROM_BOOT   = true;     % take inclination + |mag| baseline from the boot still window
MAG_FIELD_UT        = 45.386;   % WMM total intensity at your site -- sanity check for the calibration only

% dpsi_mag (the filter's magnetic-heading-bias state) is built for the
% drone: a slowly varying per-flight/per-site heading bias. Here the
% magnetometer is calibrated at startup, so there is no such bias to learn.
% With it on, the state soaks up transient errors while you move the phone
% (psiMag reached ~20-28 deg in your 16:40 log), then relaxes back toward 0
% at psiMag/300 s once you stop -- and because the magnetometer only sees
% (yaw - psiMag), yaw is dragged along: the stationary drift you saw.
USE_MAG_BIAS_STATE  = false;

% Start heading and magnetometer weight (chosen by replaying the raw
% recordings of 1 Oct 15:35 and 2 Oct 12:01 through the filter):
%  * INIT_YAW_FROM_APP: roll/pitch come from the accelerometer as before, the
%    START heading is taken ONCE from the phone's own orientation (azimuth +
%    declination), and the magnetic reference is rebuilt from that attitude --
%    the same seeding the flights use (RTK heading on mars, mocap indoors).
%    Nothing from the app is used after that instant. It removes the 3-5 deg
%    constant offset between our magnetometer calibration and Android's.
%  * MAG_NOISE_VAR 3.0 (was 0.05): indoors the field changes by several
%    degrees when the phone moves a few cm; with 0.05 the filter followed
%    every such step within seconds (yaw sliding while the phone lay still).
%    With 3.0 the gyro carries the heading and the magnetometer only trims
%    it slowly. Replay result, phone at rest / slow motion: median difference
%    to the app 0.4-0.7 deg, worst 3.5 deg (was 3.0-4.5 deg, worst 7-12 deg).
INIT_YAW_FROM_APP   = true;
MAG_NOISE_VAR       = 3.0;

GIMBAL_BAND_DEG     = 85;       % |pitch| above this: printed yaw is held, roll carries the rotation (see eulerDegHeld)

% Initial gyro-bias uncertainty (1-sigma). The bias is seeded from the boot
% still window, which pins it to ~0.005 deg/s; the filter default (P0_bg =
% 1e-6 rad^2/s^2, i.e. 0.057 deg/s) let the bias state soak up magnetometer
% heading errors after each turn (bgZ reached +0.1 deg/s in the 16:35 log
% while the boot seed was -0.006 deg/s). Floor for the value measured from
% the boot window's own noise.
GYRO_BIAS_SIGMA0_DPS = 0.01;
% Boot-window acceptance (retried until met): gyro mean and per-axis std in rad/s
BOOT_MAX_GYRO_MEAN    = 0.003;   % 0.17 deg/s -- a real phone gyro bias is far below this
BOOT_MAX_GYRO_STD     = 0.005;   % 0.29 deg/s -- resting noise was 0.04-0.12 deg/s in the raw log
BOOT_MAX_HALVES_DIFF  = 0.0026;  % 0.15 deg/s -- the two halves of the window must give the same mean
                                 % (same check as the ESP32 sketch: catches a gyro or a phone that is
                                 % still settling even when the whole-window mean and std look fine)
BOOT_MAX_INCL_ERR_DEG = 20;      % warn if the boot inclination is further than this from WMM

% Raw recording: every gyro sample and every update's accel/mag (device
% frame, before calibration) plus the app orientation, so a run can be
% replayed offline with other parameters (usque_phone_raw_<timestamp>.csv).
RECORD_RAW = true;

% Magnetometer calibration: rotate the phone through all orientations for
% MAG_CAL_SECONDS, fit an ellipsoid, remove the phone's own offset (hard
% iron) and per-axis scale. The result is saved to MAG_CAL_FILE; set
% USE_SAVED_MAG_CAL = true to skip the rotation and reuse it (the offset
% belongs to the phone, so it stays valid until the phone's own magnetic
% environment changes -- new case, magnet, etc.).
MAG_CAL_SECONDS      = 25;
% 'flat': phone lies flat on the table, you turn it slowly through one full
%         circle; fits the horizontal offset/scale only -- which is what sets
%         the heading while the phone is roughly flat. Much easier to do
%         cleanly than the 3D tumble (no carrying the phone through the
%         laptop's field). The vertical offset only shifts inclination,
%         which MAG_REF_FROM_BOOT already absorbs.
% '3d':   tumble through all orientations; full ellipsoid fit.
MAG_CAL_MODE         = 'flat';
USE_SAVED_MAG_CAL    = false;
MAG_CAL_MAX_RESIDUAL = 0.10;   % reject a fit that leaves |mag| varying more than this -- a bad fit is worse than none

SAMPLE_RATE_HZ = 100;    % MATLAB Mobile's default is 10 Hz (max 100); the phone may deliver less
FUSION_RATE_HZ = 30.0;   % same decimated update() rate as the .ino
PLOT_RATE_HZ   = 10.0;
PRINT_RATE_HZ  = 5.0;
TICK_PERIOD_S  = 0.01;   % live-loop timer period

here = fileparts(mfilename('fullpath'));
MAG_CAL_FILE = fullfile(here, 'phone_mag_cal.mat');
LIVE_STATE_FILE = fullfile(here, 'phone_live_state.txt');
runStamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
LOG_FILE = fullfile(here, sprintf('usque_phone_log_%s.txt', runStamp));
RAW_FILE = fullfile(here, sprintf('usque_phone_raw_%s.csv', runStamp));

logFid = fopen(LOG_FILE, 'w');
if logFid < 0
    warning('Could not open log file %s -- continuing without file logging.', LOG_FILE);
end
rawFid = -1;
if RECORD_RAW
    rawFid = fopen(RAW_FILE, 'w');
    if rawFid < 0
        warning('Could not open raw file %s -- continuing without raw recording.', RAW_FILE);
    end
end
% Shared with the nested timer callbacks below (tick / stopAll), which keep
% running after this function returns.
m = []; ax = []; tmr = []; running = false; magCal = [];
hBox = []; hNose = []; hArrows = [];
tUpdate = []; tPlot = []; tPrint = []; tLastData = [];
nGyroDone = 0; lastGyroT = []; lastGyro = []; gyroCount = 0; tGyroRate = []; gyroHz = NaN;
lastYawDeg = [];   % printed yaw from the previous line, held while near pitch +-90
rawSeg = 0;        % raw recording: gyro-log segment (timestamps restart after each discardlogs)
% gyro-stream gaps: at 100 Hz samples are 0.01 s apart. A gap means samples were late or lost on the
% way from the phone. Gaps up to 0.5 s are integrated across (trapezoid); longer ones are skipped,
% and the rotation during them is lost.
gapCount = 0; gapMax = 0; gapLost = 0; gapLostTime = 0;
dupCount = 0;      % gyro samples delivered a second time by the app (dropped)

logPrintf(logFid, '=== USQUE_AHRS phone live log -- started %s ===\n', char(datetime('now')));

% Setup (connect, calibrate, boot window, figure) runs synchronously; if any
% of it fails, the catch below releases the phone and closes the log. After
% setup, the live loop runs on a timer and this function RETURNS -- MATLAB
% Online only draws figures while MATLAB is idle, so a blocking while-loop
% here is what left the figure blank.
try

% ---------------------------------------------------------------- filter
f = USQUE_AHRS();
% stop here, before connecting and calibrating, if an older USQUE_AHRS.m is on the path
needed = {'getGyroBiasStd', 'beginWithHeading', 'setMagRef'};
missing = needed(~ismember(needed, methods(f)));
if ~isempty(missing)
    error('live_phone_usque:oldClass', ['%s is an older version (missing: %s). Upload the current ' ...
        'USQUE_AHRS.m next to this script, then run "clear classes" and start again.'], ...
        which('USQUE_AHRS'), strjoin(missing, ', '));
end
p = f.P;
p.mag_inclination_deg = MAG_INCLINATION_DEG;
p.mag_declination_deg = MAG_DECLINATION_DEG;
p.mag_ref_from_boot   = MAG_REF_FROM_BOOT;
p.use_mag_bias_state  = USE_MAG_BIAS_STATE;
p.mag_noise_var       = MAG_NOISE_VAR;
f.setParams(p);

% ---------------------------------------------------------------- phone
releaseStaleMobiledev(logFid);

logPrintf(logFid, 'Connecting to ''%s''...\n', DEVICE_NAME);
try
    m = mobiledev(DEVICE_NAME);
catch ME
    if contains(ME.message, 'already exists')
        error(['The phone is still held by a connection from an earlier run that this script ', ...
               'cannot reach (only a MATLAB restart frees it). Type  exit  in MATLAB Online, ', ...
               'reopen it, and run live_phone_usque again. From now on, close the figure window ', ...
               'to stop a run -- that releases the phone properly.']);
    end
    logPrintf(logFid, 'mobiledev(''%s'') failed: %s\nRetrying without a device name...\n', DEVICE_NAME, ME.message);
    m = mobiledev;
end

tConn = tic;
while ~m.Connected
    if toc(tConn) > 15
        error(['Phone did not connect within 15s (m.Connected stayed 0). Check: MATLAB Mobile ', ...
               'is open on its Sensors screen, same MathWorks account, phone online.']);
    end
    pause(0.2);
end
logPrintf(logFid, 'Connected to %s.\n', m.Device);

logPrintf(logFid, 'Enabling sensors...\n');
m.AccelerationSensorEnabled    = 1;
m.AngularVelocitySensorEnabled = 1;
m.MagneticSensorEnabled        = 1;
m.OrientationSensorEnabled     = 1;   % the phone's own orientation, logged next to the filter's for comparison
try
    m.SampleRate = SAMPLE_RATE_HZ;
catch ME
    logPrintf(logFid, 'Could not set SampleRate (%s) -- keeping %g Hz.\n', ME.message, m.SampleRate);
end
logPrintf(logFid, 'Starting stream at %g Hz...\n', m.SampleRate);
m.Logging = 1;
pause(1.0);   % let the connector actually start delivering samples

% ---------------------------------------------------------------- magnetometer calibration
if USE_SAVED_MAG_CAL && isfile(MAG_CAL_FILE)
    S = load(MAG_CAL_FILE, 'magCal');
    magCal = S.magCal;
    logPrintf(logFid, 'Using saved magnetometer calibration from %s (fitted %s).\n', MAG_CAL_FILE, magCal.when);
else
    if strcmpi(MAG_CAL_MODE, 'flat')
        magCal = runMagCalibrationFlat(m, MAG_CAL_SECONDS, logFid);
    else
        magCal = runMagCalibration(m, MAG_CAL_SECONDS, logFid);
    end
    if magCal.residualFrac <= MAG_CAL_MAX_RESIDUAL
        save(MAG_CAL_FILE, 'magCal');
        logPrintf(logFid, 'Saved to %s (set USE_SAVED_MAG_CAL = true to reuse it).\n', MAG_CAL_FILE);
    else
        logPrintf(logFid, ['REJECTED: fit leaves |mag| varying %.1f%% (limit %.0f%%) -- not saved, ', ...
            'NOT applied; using the uncorrected magnetometer. Re-run it holding the phone out ', ...
            'in front of you, rotating in place, away from the laptop and desk.\n'], ...
            100 * magCal.residualFrac, 100 * MAG_CAL_MAX_RESIDUAL);
        magCal.offset = [0 0 0]; magCal.scale = [1 1 1]; magCal.kind = 'none (rejected)';
    end
end
logPrintf(logFid, ['mag calibration: offset %6.1f %6.1f %6.1f uT, scale %5.3f %5.3f %5.3f, ', ...
    'field radius %.1f uT (WMM %.1f uT), fit residual %.1f%%\n'], ...
    magCal.offset, magCal.scale, magCal.radius, MAG_FIELD_UT, 100 * magCal.residualFrac);

% ---------------------------------------------------------------- boot still window
% The figure is created only AFTER calibration -- the one version that ran
% for you did it in this order; the versions that opened the figure before
% or during connection/calibration hung in MATLAB Online.
logPrintf(logFid, 'Put the phone down FLAT and STILL...\n');
waitForStill(m, logFid);
logPrintf(logFid, 'Still -- capturing boot window...\n');
NCAL = 150;
for bootTry = 1:6
accB = zeros(NCAL,3); gyB = zeros(NCAL,3); magB = zeros(NCAL,3);
k = 0;
tCal = tic; tCalWarn = tic;
while k < NCAL
    if toc(tCal) > 30
        error(['No sensor data received for 30s (Connected=%d, Logging=%d). Check the sensor ', ...
               'toggles in MATLAB Mobile and keep the app in the foreground.'], m.Connected, m.Logging);
    end
    if toc(tCalWarn) > 3
        tCalWarn = tic;
        logPrintf(logFid, '...waiting for sensor data (%d/%d samples)\n', k, NCAL);
    end
    a = m.Acceleration; g = m.AngularVelocity; mg = m.MagneticField;
    if isempty(a) || isempty(g) || isempty(mg), pause(0.02); continue; end
    k = k + 1;
    accB(k,:) = remapDeviceToFRD(a);
    gyB(k,:)  = remapDeviceToFRD(g);
    magB(k,:) = applyMagCal(remapDeviceToFRD(mg), magCal);
    pause(0.02);
end
accAvg = mean(accB,1); gyAvg = mean(gyB,1); magAvg = mean(magB,1);
% The window must be REALLY still: its gyro mean becomes the bias seed and
% its mag/accel the heading reference. waitForStill only checks < 0.05 rad/s
% per sample, which a phone still settling after the calibration spin passes
% (12:21 log: seed -1.05 deg/s on z, true bias ~0.005 deg/s -- the filter
% then spent 100 s unlearning it while yaw drifted 7 deg).
gyStd = max(std(gyB, 0, 1));
halvesDiff = max(abs(mean(gyB(1:floor(NCAL/2),:), 1) - mean(gyB(floor(NCAL/2)+1:end,:), 1)));
if (gyStd <= BOOT_MAX_GYRO_STD && norm(gyAvg) <= BOOT_MAX_GYRO_MEAN && halvesDiff <= BOOT_MAX_HALVES_DIFF) || bootTry == 6
    if bootTry == 6
        logPrintf(logFid, 'WARNING: boot window never became still -- continuing with a doubtful bias seed.\n');
    end
    break;
end
logPrintf(logFid, ['Boot window not still (gyro mean %.4f rad/s, max std %.4f rad/s, halves differ ', ...
    '%.4f rad/s) -- leave the phone untouched; retrying...\n'], norm(gyAvg), gyStd, halvesDiff);
pause(1.5);
end
logPrintf(logFid, 'accel FRD avg: %7.3f %7.3f %7.3f  (want ~0 0 -9.81; |a|=%.3f)\n', accAvg, norm(accAvg));
logPrintf(logFid, 'gyro  FRD avg (bias seed): %7.4f %7.4f %7.4f rad/s\n', gyAvg);
logPrintf(logFid, 'mag   FRD avg: %7.1f %7.1f %7.1f  (|mag|=%.1f uT)\n', magAvg, norm(magAvg));

% bias prior: the seed's own standard error (sample std / sqrt(N)),
% floored at GYRO_BIAS_SIGMA0_DPS
sigSeed = max(std(gyB, 0, 1)) / sqrt(NCAL);
sigBias0 = max(sigSeed, deg2rad(GYRO_BIAS_SIGMA0_DPS));
p = f.P; p.P0_bg = sigBias0^2; f.setParams(p);

initHeadingDeg = NaN;
if INIT_YAW_FROM_APP
    ori0 = m.Orientation;
    if ~isempty(ori0) && all(isfinite(ori0))
        initHeadingDeg = ori0(1) + MAG_DECLINATION_DEG;   % app azimuth is from magnetic north
    else
        logPrintf(logFid, 'No app orientation available -- falling back to the magnetometer heading.\n');
    end
end
if isfinite(initHeadingDeg)
    f.beginWithHeading(accAvg, magAvg, initHeadingDeg);
    % magnetic reference = the boot field expressed in the nav frame of this
    % seeded attitude, so the magnetometer agrees with the seed at t0
    f.setMagRef(quat2rotm_local(f.getQuaternion()) * (magAvg(:) / norm(magAvg)));
    logPrintf(logFid, 'start heading seeded from the app: %.2f deg (azimuth %.2f + declination %.2f)\n', ...
        initHeadingDeg, ori0(1), MAG_DECLINATION_DEG);
else
    f.begin(accAvg, magAvg);
end
f.setGyroBias(gyAvg);
logPrintf(logFid, 'gyro bias prior: 1-sigma %.4f deg/s (seed standard error %.4f deg/s)\n', ...
    rad2deg(sigBias0), rad2deg(sigSeed));
logPrintf(logFid, 'mag inclination: WMM %.2f deg, measured at boot %.2f deg -> filter uses %.2f deg\n', ...
    MAG_INCLINATION_DEG, measuredInclDeg(accAvg, magAvg), f.getMagInclRefDeg());
if abs(measuredInclDeg(accAvg, magAvg) - MAG_INCLINATION_DEG) > BOOT_MAX_INCL_ERR_DEG
    logPrintf(logFid, ['WARNING: the magnetic field at this spot is badly disturbed (inclination %.0f deg off ', ...
        'the expected value). The heading reference taken here will be wrong as soon as the phone ', ...
        'moves, and the magnetometer gate will then reject it. Move away from the laptop / metal / ', ...
        'magnets (case, charger, speakers) and restart.\n'], ...
        abs(measuredInclDeg(accAvg, magAvg) - MAG_INCLINATION_DEG));
end

if rawFid > 0
    % Header: everything a replay needs to rebuild the same filter start.
    % Data rows (device frame: x=right, y=top, z=out of screen; the replay
    % applies remapDeviceToFRD and the mag calibration itself):
    %   G,seg,t,wx,wy,wz              one per gyro sample (phone timestamp, rad/s);
    %                                  seg increments at each discardlogs (timestamps restart)
    %   U,seg,t,ax,ay,az,mx,my,mz,az_app,p_app,r_app
    %                                  one per filter update; t = last integrated gyro
    %                                  timestamp, accel m/s^2, mag uT (uncalibrated),
    %                                  app orientation deg
    fprintf(rawFid, '# usque_phone_raw v1, started %s, device %s\n', char(datetime('now')), DEVICE_NAME);
    fprintf(rawFid, '# mag_cal_offset_frd,%.6f,%.6f,%.6f\n', magCal.offset);
    fprintf(rawFid, '# mag_cal_scale_frd,%.6f,%.6f,%.6f\n', magCal.scale);
    fprintf(rawFid, '# boot_accel_frd,%.6f,%.6f,%.6f\n', accAvg);
    fprintf(rawFid, '# boot_gyro_frd,%.8f,%.8f,%.8f\n', gyAvg);
    fprintf(rawFid, '# boot_mag_frd_calibrated,%.6f,%.6f,%.6f\n', magAvg);
    fprintf(rawFid, '# P0_bg,%.6e\n', sigBias0^2);
    fprintf(rawFid, '# mag_inclination_deg,%.4f\n# mag_declination_deg,%.4f\n# mag_ref_from_boot,%d\n', ...
        MAG_INCLINATION_DEG, MAG_DECLINATION_DEG, MAG_REF_FROM_BOOT);
    fprintf(rawFid, '# use_mag_bias_state,%d\n# fusion_rate_hz,%.2f\n', USE_MAG_BIAS_STATE, FUSION_RATE_HZ);
    fprintf(rawFid, '# mag_noise_var,%.4f\n# init_heading_deg,%.4f\n', MAG_NOISE_VAR, initHeadingDeg);
    fprintf(rawFid, 'type,seg,t,c1,c2,c3,c4,c5,c6,c7,c8,c9\n');
end

logPrintf(logFid, 'Opening 3D view...\n');
[hBox, hNose, hArrows, ax] = setupBoardPlot();

catch ME
    releasePhone(m);
    if rawFid > 0, fclose(rawFid); end
    closeLog(logFid, LOG_FILE);
    rethrow(ME);
end

% ---------------------------------------------------------------- live loop (timer)
tUpdate = tic; tPlot = tic; tPrint = tic; tLastData = tic;
discardlogs(m);          % start the gyro log fresh: timestamps restart from here
nGyroDone = 0;           % rows of the current gyro log already integrated
lastGyroT = []; lastGyro = [];
gyroCount = 0; tGyroRate = tic; gyroHz = NaN;
running = true;
tmr = timer('Name', 'live_phone_usque', 'ExecutionMode', 'fixedSpacing', ...
    'Period', TICK_PERIOD_S, 'BusyMode', 'drop', ...
    'TimerFcn', @(~,~) tick(), ...
    'ErrorFcn', @(~,evt) stopAll(sprintf('Live loop error: %s', evt.Data.message)), ...
    'StopFcn',  @(t,~) onTimerStopped(t));
ax.fig.DeleteFcn = @(~,~) stopAll('Figure closed, stopping.');
start(tmr);
logPrintf(logFid, ['\nRunning in the background -- the Command Window stays free. ', ...
    'Close the figure window to stop.\n\n']);
% (returns here; tick() keeps running on the timer)

    % ---- one live-loop iteration, called by the timer every TICK_PERIOD_S
    function tick()
        if ~running, return; end
        if isempty(ax) || ~ishandle(ax.fig)
            stopAll('Figure closed, stopping.');
            return;
        end
        % ---- gyro: integrate EVERY logged sample using the phone's own
        % timestamps. (Reading m.AngularVelocity only gives the latest
        % sample; multiplying it by MATLAB's wall-clock time between ticks
        % loses or double-counts rotation whenever samples arrive late or in
        % bursts -- which they do over MATLAB Online. That lost rotation was
        % the heading error the magnetometer then slowly pulled back.)
        [av, tg] = angvellog(m);
        n = size(av, 1);
        for i = nGyroDone+1:n
            if rawFid > 0
                fprintf(rawFid, 'G,%d,%.6f,%.8f,%.8f,%.8f\n', rawSeg, tg(i), av(i,1), av(i,2), av(i,3));
            end
            gi = remapDeviceToFRD(av(i,:))';
            if ~isempty(lastGyroT) && tg(i) <= lastGyroT
                % MATLAB Mobile now and then delivers a block of samples twice (the
                % time stamps step back by one block). Integrating it again adds the
                % rotation of that block a second time -- 11 deg in one 90 deg turn on
                % 5 Oct 2026 -- so a sample that is not newer than the last one is dropped.
                dupCount = dupCount + 1;
                continue;
            end
            if ~isempty(lastGyroT)
                dtg = tg(i) - lastGyroT;
                if dtg > 0.05
                    gapCount = gapCount + 1; gapMax = max(gapMax, dtg);
                end
                if dtg > 0 && dtg < 0.5
                    % trapezoidal rate over the interval; predict() ignores steps longer than
                    % 0.2 s, so a longer gap is integrated in equal sub-steps
                    nSub = ceil(dtg / 0.2);
                    for kSub = 1:nSub, f.predict(0.5 * (gi + lastGyro), dtg / nSub); end
                elseif dtg >= 0.5
                    gapLost = gapLost + 1; gapLostTime = gapLostTime + dtg;
                end
            end
            lastGyroT = tg(i); lastGyro = gi;
        end
        if n > nGyroDone
            gyroCount = gyroCount + (n - nGyroDone);
            nGyroDone = n;
            tLastData = tic;
        end
        if n > 2000
            % keep the log (re-read in full every tick) short. Timestamps
            % restart after discardlogs, so the first interval after it is
            % skipped -- at most one sample interval of rotation.
            discardlogs(m);
            nGyroDone = 0; lastGyroT = [];
            rawSeg = rawSeg + 1;
        end
        if toc(tGyroRate) >= 2
            gyroHz = gyroCount / toc(tGyroRate);
            gyroCount = 0; tGyroRate = tic;
        end

        a = m.Acceleration; mg = m.MagneticField;
        if isempty(a) || isempty(mg) || toc(tLastData) > 5
            if toc(tLastData) > 5
                tLastData = tic;
                logPrintf(logFid, '...no gyro data for 5s (Connected=%d) -- is MATLAB Mobile still in the foreground?\n', m.Connected);
            end
            return;
        end
        accel = remapDeviceToFRD(a)';
        mag   = applyMagCal(remapDeviceToFRD(mg), magCal)';

        if toc(tUpdate) >= 1/FUSION_RATE_HZ
            tUpdate = tic;
            f.update(accel, mag, 0);   % throttlePct=0: no motor/vibration source on a handheld phone
            if rawFid > 0
                ori = m.Orientation;
                if isempty(ori), ori = [NaN NaN NaN]; end
                tU = NaN; if ~isempty(lastGyroT), tU = lastGyroT; end
                fprintf(rawFid, 'U,%d,%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f\n', ...
                    rawSeg, tU, a(1), a(2), a(3), mg(1), mg(2), mg(3), ori(1), ori(2), ori(3));
            end
        end

        if toc(tPlot) >= 1/PLOT_RATE_HZ
            tPlot = tic;
            updateBoardPlot(hBox, hNose, hArrows, f.getQuaternion(), ax.boxV, ax.noseV);
            writeLiveStateFile(LIVE_STATE_FILE, f);
            drawnow limitrate;
        end

        if toc(tPrint) >= 1/PRINT_RATE_HZ
            tPrint = tic;
            q = f.getQuaternion();
            [roll, pitch, yaw, nearGL] = eulerDegHeld(q, lastYawDeg, GIMBAL_BAND_DEG);
            lastYawDeg = yaw;
            glTag = '  '; if nearGL, glTag = 'GL'; end
            sig = f.getAttitudeStdDeg();
            magNormRef = f.getMagNormRef();
            magNorm = norm(mag);
            fracDevPct = 100 * abs(magNorm - magNormRef) / max(magNormRef, 1e-9);
            bgDps = rad2deg(f.getGyroBias());
            sbgDps = rad2deg(f.getGyroBiasStd());
            % the phone's own fused orientation (Android convention:
            % azimuth 0-360 clockwise from magnetic north, pitch positive
            % top-edge DOWN, roll positive left-edge DOWN) -- compare
            % azimuth with yaw360 below; pitch/roll are opposite-sign by
            % convention, not by error.
            ori = m.Orientation;
            if isempty(ori), ori = [NaN NaN NaN]; end
            logPrintf(logFid, ['roll:%7.2f  pitch:%7.2f  yaw:%7.2f %s |  sigRPY:%5.2f %5.2f %5.2f  |  ', ...
                'psiMag:%6.2f (sig %5.2f)  |  bgZ:%7.4f dps  bgXY:%7.4f %7.4f  sigBg:%6.4f %6.4f %6.4f  |  ', ...
                'accelInfl:%5.2f  magInfl:%5.2f  gate:%d  |  ', ...
                '|mag|:%6.1f  ref:%6.1f  dev:%5.1f%%  incl:%6.2f  |  gyroHz:%5.1f  gaps:%d  |  ', ...
                'yaw360:%6.1f  app az:%6.1f p:%6.1f r:%6.1f  |  q:%8.5f %8.5f %8.5f %8.5f\n'], ...
                roll, pitch, yaw, glTag, sig(1), sig(2), sig(3), f.getPsiMagDeg(), f.getPsiMagStdDeg(), bgDps(3), ...
                bgDps(1), bgDps(2), sbgDps(1), sbgDps(2), sbgDps(3), ...
                f.getLastAccelInflation(), f.getLastMagInflation(), f.getLastMagGateStatus(), ...
                magNorm, magNormRef, fracDevPct, measuredInclDeg(accel, mag), gyroHz, gapCount, ...
                mod(yaw, 360), ori(1), ori(2), ori(3), q(1), q(2), q(3), q(4));
        end
    end

    % ---- the timer was stopped from anywhere (figure close, error, a later
    % run's startup, or stop(timerfind(...)) by hand): always release the
    % phone, then delete the timer.
    function onTimerStopped(t)
        stopAll('Live loop stopped.');
        delete(t);
    end

    % ---- single shutdown path: figure closed, live-loop error, or timer stop
    function stopAll(msg)
        if ~running, return; end
        running = false;
        logPrintf(logFid, '%s\n', msg);
        logPrintf(logFid, ['gyro stream: %d gaps longer than 0.05 s (longest %.2f s); %d gaps of 0.5 s or more ', ...
            'were skipped (%.1f s of rotation not integrated); %d repeated samples dropped.\n'], ...
            gapCount, gapMax, gapLost, gapLostTime, dupCount);
        try
            if isvalid(tmr), stop(tmr); end   % StopFcn deletes it
        catch
        end
        releasePhone(m);
        closeRaw();
        closeLog(logFid, LOG_FILE);
    end

    function closeRaw()
        if rawFid > 0 && ~isempty(fopen(rawFid))
            fclose(rawFid);
            logPrintf(logFid, 'Raw sensor data saved to %s\n', RAW_FILE);
        end
        rawFid = -1;
    end
end


% ============================================================================
%  local functions
% ============================================================================

function releasePhone(m)
    try, m.Logging = 0; catch, end %#ok<*NOCOM>
    try, delete(m); catch, end
end

function closeLog(logFid, logFile)
    if logFid > 0 && ~isempty(fopen(logFid))
        fclose(logFid);
        fprintf('Log saved to %s\n', logFile);
    end
end

function magCal = runMagCalibration(m, seconds, logFid)
    % Collect magnetometer samples while the user tumbles the phone through
    % all orientations, then fit (see fitMagEllipsoid). Samples are taken in
    % the FRD frame, the same frame the calibration is applied in.
    logPrintf(logFid, ['\nMAGNETOMETER CALIBRATION (%d s): pick the phone up and slowly rotate it ', ...
        'through EVERY orientation -- face up, face down, each edge down, top up, top down -- in a ', ...
        'slow tumbling figure-8. Do it away from the laptop and any metal.\n'], seconds);
    for c = 3:-1:1
        logPrintf(logFid, 'Starting in %d...\n', c);
        pause(1);
    end
    logPrintf(logFid, 'ROTATE NOW.\n');
    M = zeros(ceil(seconds * 60) + 100, 3);
    N = 0; last = [NaN NaN NaN];
    t0 = tic; tProg = tic;
    while toc(t0) < seconds
        mg = m.MagneticField;
        if ~isempty(mg)
            v = remapDeviceToFRD(mg);
            if any(v ~= last)   % MagneticField returns the latest sample; skip repeats
                N = N + 1;
                if N > size(M,1), M(end+1000,:) = 0; end
                M(N,:) = v; last = v;
            end
        end
        if toc(tProg) > 5
            tProg = tic;
            logPrintf(logFid, '  ...%2.0f s left, %d samples, %s covered\n', ...
                seconds - toc(t0), N, coverageStr(M(1:N,:)));
        end
        pause(0.02);
    end
    M = M(1:N,:);
    logPrintf(logFid, 'Rotation done: %d samples, %s covered. Fitting...\n', N, coverageStr(M));
    magCal = fitMagEllipsoid(M, logFid);
end

function magCal = runMagCalibrationFlat(m, seconds, logFid)
    % Phone flat on the table, turned slowly through one full circle. The
    % horizontal field components then trace a circle whose centre is the
    % phone's own horizontal offset -- the part that bends the heading.
    logPrintf(logFid, ['\nMAGNETOMETER CALIBRATION, flat (%d s): lay the phone FLAT on the table and ', ...
        'slowly turn it through one full circle (360 deg, about one turn per %d s). Keep it flat and ', ...
        'in the same spot -- just spin it.\n'], seconds, seconds);
    for c = 3:-1:1
        logPrintf(logFid, 'Starting in %d...\n', c);
        pause(1);
    end
    logPrintf(logFid, 'TURN NOW.\n');
    M = zeros(ceil(seconds * 60) + 100, 3);
    N = 0; last = [NaN NaN NaN];
    t0 = tic; tProg = tic;
    while toc(t0) < seconds
        mg = m.MagneticField;
        if ~isempty(mg)
            v = remapDeviceToFRD(mg);
            if any(v ~= last)
                N = N + 1;
                if N > size(M,1), M(end+1000,:) = 0; end
                M(N,:) = v; last = v;
            end
        end
        if toc(tProg) > 5
            tProg = tic;
            logPrintf(logFid, '  ...%2.0f s left, %d samples, %s of the circle covered\n', ...
                seconds - toc(t0), N, headingCoverageStr(M(1:N,1:2)));
        end
        pause(0.02);
    end
    M = M(1:N,:);
    logPrintf(logFid, 'Turn done: %d samples, %s of the circle covered. Fitting...\n', ...
        N, headingCoverageStr(M(:,1:2)));
    magCal = fitMagCircle(M, logFid);
end

function magCal = fitMagCircle(M, logFid)
    % Horizontal (x,y) offset + per-axis scale; z untouched. Tries a circle
    % and an axis-aligned ellipse, keeps whichever leaves the horizontal
    % magnitude more constant.
    magCal = struct('offset', [0 0 0], 'scale', [1 1 1], 'radius', NaN, 'residualFrac', NaN, ...
                    'nSamples', size(M,1), 'kind', 'none', 'when', char(datetime('now')));
    if size(M,1) < 30
        logPrintf(logFid, 'WARNING: only %d samples -- calibration skipped.\n', size(M,1));
        return;
    end
    x = M(:,1); y = M(:,2);
    cands = struct('offset', {}, 'scale', {}, 'kind', {});
    w = [2*x, 2*y, ones(size(x))] \ (x.^2 + y.^2);
    cands(end+1) = struct('offset', [w(1) w(2) 0], 'scale', [1 1 1], 'kind', 'circle (horizontal offset)');
    v = [x.^2, y.^2, x, y] \ ones(size(x));
    if all(v(1:2) > 0)
        c = -v(3:4)' ./ (2 * v(1:2)');
        G = 1 + sum(v(1:2)' .* c.^2);
        if G > 0
            radii = sqrt(G ./ v(1:2)');
            cands(end+1) = struct('offset', [c 0], 'scale', [mean(radii) ./ radii 1], ...
                                  'kind', 'ellipse (horizontal offset + scale)');
        end
    end
    best = Inf;
    for i = 1:numel(cands)
        Mc = (M - cands(i).offset) .* cands(i).scale;
        r = sqrt(sum(Mc(:,1:2).^2, 2));
        frac = std(r) / mean(r);
        if frac < best
            best = frac;
            magCal.offset = cands(i).offset; magCal.scale = cands(i).scale; magCal.kind = cands(i).kind;
            magCal.radius = mean(r); magCal.residualFrac = frac;
        end
    end
    logPrintf(logFid, 'Fit: %s. Horizontal field %.1f uT after correction.\n', magCal.kind, magCal.radius);
end

function s = headingCoverageStr(H)
    % 30-degree heading sectors (around the min/max midpoint) that contain
    % samples. Aim for 12/12.
    if size(H,1) < 10, s = '0/12 sectors'; return; end
    d = H - (max(H,[],1) + min(H,[],1)) / 2;
    sec = floor(mod(atan2d(d(:,2), d(:,1)), 360) / 30);
    s = sprintf('%d/12 sectors', numel(unique(sec)));
end

function magCal = fitMagEllipsoid(M, logFid)
    % Hard iron (offset) + axis-aligned soft iron (per-axis scale). Tries an
    % axis-aligned ellipsoid  A x^2+B y^2+C z^2+D x+E y+F z = 1  and a sphere
    % |m-c|^2 = r^2, keeps whichever leaves the more constant |m| afterwards.
    magCal = struct('offset', [0 0 0], 'scale', [1 1 1], 'radius', NaN, 'residualFrac', NaN, ...
                    'nSamples', size(M,1), 'kind', 'none', 'when', char(datetime('now')));
    if size(M,1) < 50
        logPrintf(logFid, 'WARNING: only %d samples -- calibration skipped (no correction applied).\n', size(M,1));
        return;
    end
    x = M(:,1); y = M(:,2); z = M(:,3);
    cands = struct('offset', {}, 'scale', {}, 'kind', {});

    w = [2*x, 2*y, 2*z, ones(size(x))] \ (x.^2 + y.^2 + z.^2);
    cands(end+1) = struct('offset', w(1:3)', 'scale', [1 1 1], 'kind', 'sphere (offset only)');

    v = [x.^2, y.^2, z.^2, x, y, z] \ ones(size(x));
    if all(v(1:3) > 0)
        offset = -v(4:6)' ./ (2 * v(1:3)');
        G = 1 + sum(v(1:3)' .* offset.^2);
        if G > 0
            radii = sqrt(G ./ v(1:3)');
            cands(end+1) = struct('offset', offset, 'scale', mean(radii) ./ radii, ...
                                  'kind', 'ellipsoid (offset + per-axis scale)');
        end
    end

    best = Inf;
    for i = 1:numel(cands)
        r = sqrt(sum(((M - cands(i).offset) .* cands(i).scale).^2, 2));
        frac = std(r) / mean(r);
        if frac < best
            best = frac;
            magCal.offset = cands(i).offset; magCal.scale = cands(i).scale; magCal.kind = cands(i).kind;
            magCal.radius = mean(r); magCal.residualFrac = frac;
        end
    end
    logPrintf(logFid, 'Fit: %s.\n', magCal.kind);
    if magCal.residualFrac > 0.05
        logPrintf(logFid, ['WARNING: |mag| still varies %.1f%% after calibration -- the rotation probably ', ...
            'missed some orientations, or the phone moved through a disturbed field (laptop, metal). ', ...
            'Consider re-running.\n'], 100 * magCal.residualFrac);
    end
end

function s = coverageStr(M)
    % Octants around the min/max midpoint that contain samples -- a quick
    % proxy for "did the rotation see all directions". Aim for 8/8.
    if size(M,1) < 10, s = '0/8 octants'; return; end
    d = M - (max(M,[],1) + min(M,[],1)) / 2;
    oct = (d(:,1) > 0) + 2*(d(:,2) > 0) + 4*(d(:,3) > 0);
    s = sprintf('%d/8 octants', numel(unique(oct)));
end

function v = applyMagCal(v, magCal)
    v = (v - magCal.offset) .* magCal.scale;
end

function waitForStill(m, logFid)
    % Wait for 1.5 s of continuous stillness before the boot still window,
    % so the TRIAD init / gyro-bias seed isn't taken while the phone is
    % still being put down after the calibration rotation.
    tStill = []; t0 = tic; tMsg = tic;
    while true
        a = m.Acceleration; g = m.AngularVelocity;
        if ~isempty(a) && ~isempty(g) && norm(g) < 0.05 && abs(norm(a) - 9.81) < 0.4
            if isempty(tStill), tStill = tic; end
            if toc(tStill) > 1.5, return; end
        else
            tStill = [];
        end
        if toc(t0) > 60
            logPrintf(logFid, 'WARNING: phone was never still for 1.5 s -- continuing anyway.\n');
            return;
        end
        if toc(tMsg) > 5
            tMsg = tic;
            logPrintf(logFid, '...waiting for the phone to lie still\n');
        end
        pause(0.05);
    end
end

function releaseStaleMobiledev(logFid)
    % A previous run that is still going (its figure never closed) holds the
    % phone inside its timer, not in any workspace. Stopping its timer runs
    % its onTimerStopped -> releases that phone connection.
    old = timerfind('Name', 'live_phone_usque');
    if ~isempty(old)
        logPrintf(logFid, 'Stopping %d live loop(s) left running by an earlier run...\n', numel(old));
        try, stop(old); catch, end
        pause(0.5);
        old = old(isvalid(old));
        if ~isempty(old), delete(old); end
    end
    % mobiledev objects left in the base workspace by the old script
    % version still hold the phone connection -- close them too.
    vars = evalin('base', 'whos');
    for i = 1:numel(vars)
        if contains(vars(i).class, 'mobiledev')
            logPrintf(logFid, 'Releasing stale mobiledev ''%s'' left in the workspace...\n', vars(i).name);
            try
                evalin('base', sprintf('%s.Logging = 0; delete(%s); clear %s;', ...
                    vars(i).name, vars(i).name, vars(i).name));
            catch
            end
        end
    end
end

function [roll, pitch, yaw, nearGL] = eulerDegHeld(q, prevYawDeg, bandDeg)
    % Roll/pitch/yaw (ZYX, degrees) for PRINTING only -- the filter and the
    % 3D view use the quaternion and never go through Euler angles.
    % At pitch = +-90 deg the ZYX angles are degenerate: the rotation only
    % fixes (roll - yaw) (at +90) or (roll + yaw) (at -90), so the standard
    % atan2 formulas return an arbitrary split and tiny changes swing roll
    % and yaw by up to 180 deg in opposite directions. Inside the band we
    % hold yaw at its last printed value and extract roll/pitch from
    % Rz(-yaw)*R, whose middle row doesn't depend on pitch -- roll carries
    % the rotation smoothly instead of jumping. nearGL flags those lines.
    R = quat2rotm_local(q);
    pitch = asind(max(-1, min(1, -R(3,1))));
    nearGL = abs(pitch) > bandDeg && ~isempty(prevYawDeg);
    if ~nearGL
        roll = atan2d(R(3,2), R(3,3));
        yaw  = atan2d(R(2,1), R(1,1));
        return;
    end
    yaw = prevYawDeg;
    c = cosd(yaw); s = sind(yaw);
    Rp = [c s 0; -s c 0; 0 0 1] * R;   % Rz(-yaw) * R  ~=  Ry(pitch) * Rx(roll)
    roll  = atan2d(-Rp(2,3), Rp(2,2));
    pitch = atan2d(-Rp(3,1), Rp(1,1));
end

function d = measuredInclDeg(accel, mag)
    down = -accel(:) / max(norm(accel), 1e-9);
    d = asind(max(-1, min(1, (mag(:) / max(norm(mag), 1e-9))' * down)));
end

function logPrintf(logFid, fmt, varargin)
    % Command Window AND log file, so the saved log matches the console.
    fprintf(fmt, varargin{:});
    if logFid > 0
        fprintf(logFid, fmt, varargin{:});
    end
    drawnow limitrate;   % MATLAB Online only flushes output while the event queue is serviced
end

function writeLiveStateFile(path, f)
    % Overwrites (not appends) with a single line -- the browser re-reads
    % the whole small file each poll.
    [roll, pitch, yaw] = f.getEulerDeg();
    q = f.getQuaternion();
    sig = f.getAttitudeStdDeg();
    fid = fopen(path, 'w');
    if fid < 0, return; end   % locked by a reader mid-write this tick -- next tick retries
    fprintf(fid, ['roll_deg:%.2f,pitch_deg:%.2f,yaw_deg:%.2f,', ...
                  'qw:%.5f,qx:%.5f,qy:%.5f,qz:%.5f,', ...
                  'sigRoll_deg:%.2f,sigPitch_deg:%.2f,sigYaw_deg:%.2f,', ...
                  'psiMag_deg:%.2f,sigPsiMag_deg:%.2f,', ...
                  'accelInfl:%.2f,magInfl:%.2f\n'], ...
        roll, pitch, yaw, q(1), q(2), q(3), q(4), ...
        sig(1), sig(2), sig(3), f.getPsiMagDeg(), f.getPsiMagStdDeg(), ...
        f.getLastAccelInflation(), f.getLastMagInflation());
    fclose(fid);
end

function v_frd = remapDeviceToFRD(v)
    % Phone flat, screen up, top edge = forward. Device (x=right,
    % y=toward phone-top, z=out of screen) -> FRD (x=fwd, y=right, z=down).
    % det=+1 (proper rotation), applied identically to accel/gyro/mag.
    v_frd = [v(2), v(1), -v(3)];
end

function R = quat2rotm_local(q)
    % Hamilton (w,x,y,z), body->nav: v_nav = R * v_body.
    w=q(1); x=q(2); y=q(3); z=q(4);
    R = [1-2*(y^2+z^2),   2*(x*y-w*z),     2*(x*z+w*y);
         2*(x*y+w*z),     1-2*(x^2+z^2),   2*(y*z-w*x);
         2*(x*z-w*y),     2*(y*z+w*x),     1-2*(x^2+y^2)];
end

function [hBox, hNose, hArrows, ax] = setupBoardPlot()
    % Default (light) figure/axes colours and no camlight/gouraud lighting:
    % MATLAB Online renders figures in the browser, and a near-black
    % background plus scene lighting there can leave the whole view looking
    % blank. Flat shading with black edges is visible in every renderer.
    % Earlier runs' viewer windows (possibly half-initialised) -- close them
    % so only one live figure exists; their DeleteFcn also stops any old loop.
    delete(findall(0, 'Type', 'figure', 'Name', 'USQUE_AHRS -- phone live orientation'));
    fig = figure('Name', 'USQUE_AHRS -- phone live orientation', 'NumberTitle', 'off', 'Visible', 'on');
    axHandle = axes(fig);
    hold(axHandle, 'on'); grid(axHandle, 'on'); axis(axHandle, 'equal');
    xlim(axHandle, [-3 3]); ylim(axHandle, [-3 3]); zlim(axHandle, [-3 3]);
    xlabel(axHandle, 'X'); ylabel(axHandle, 'Y'); zlabel(axHandle, 'Z (up)');
    view(axHandle, 135, 25);

    % cube body; faces coloured so a cube's orientation is still readable:
    % screen side (body -Z, up when flat) blue, back (+Z) dark grey,
    % front (+X, phone top edge) orange, the other three light grey
    hx=0.8; hy=0.8; hz=0.8;
    V = [ -hx -hy -hz;  hx -hy -hz;  hx  hy -hz; -hx  hy -hz; ...
          -hx -hy  hz;  hx -hy  hz;  hx  hy  hz; -hx  hy  hz ];
    Fb = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];
    faceRGB = [0.227 0.431 0.647;   % -Z  screen / top
               0.35  0.35  0.35;    % +Z  back / bottom
               0.80  0.80  0.80;    % -Y  left
               1.00  0.60  0.00;    % +X  front
               0.80  0.80  0.80;    % +Y  right
               0.80  0.80  0.80];   % -X  rear
    hBox = patch(axHandle, 'Vertices', V, 'Faces', Fb, 'FaceColor', 'flat', ...
        'FaceVertexCData', faceRGB, 'EdgeColor', 'k', 'FaceAlpha', 0.85);

    % nose marker (small pyramid at +X) so front/back is unambiguous
    Vn = [hx 0 0; hx+0.5 0.2 0; hx+0.5 -0.1 0.17; hx+0.5 -0.1 -0.17];
    Fn = [1 2 3; 1 3 4; 1 4 2; 2 3 4];
    hNose = patch(axHandle, 'Vertices', Vn, 'Faces', Fn, 'FaceColor', [1 0.6 0], 'EdgeColor', 'k');

    % rotating body-axis arrows: X=red/roll, Y=green/pitch, Z=blue/yaw
    AXLEN = 2.4;
    hArrows(1) = line(axHandle, [0 AXLEN], [0 0], [0 0], 'Color', [0.85 0.1 0.1], 'LineWidth', 3);
    hArrows(2) = line(axHandle, [0 0], [0 AXLEN], [0 0], 'Color', [0.1 0.65 0.1], 'LineWidth', 3);
    hArrows(3) = line(axHandle, [0 0], [0 0], [0 AXLEN], 'Color', [0.1 0.3 0.9], 'LineWidth', 3);

    % fixed (non-rotating) world reference triad for context
    line(axHandle, [0 1.5], [0 0], [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':');
    line(axHandle, [0 0], [0 1.5], [0 0], 'Color', [0.5 0.5 0.5], 'LineStyle', ':');
    line(axHandle, [0 0], [0 0], [0 1.5], 'Color', [0.5 0.5 0.5], 'LineStyle', ':');

    updateBoardPlot(hBox, hNose, hArrows, [1;0;0;0], V, Vn);   % same display transform as live frames

    ax.fig = fig; ax.ax = axHandle;
    ax.boxV = V; ax.noseV = Vn; ax.AXLEN = AXLEN;
end

function updateBoardPlot(hBox, hNose, hArrows, q, V0, Vn0)
    % V0/Vn0 are the ORIGINAL body-frame vertices -- each frame rotates from
    % that fixed shape. QDISP (180 deg about X: negates Y and Z together)
    % shows NED-down as screen-up. It must be a proper rotation (det=+1):
    % mirroring a single axis flips handedness and makes every rotation,
    % yaw included, appear reversed on screen.
    QDISP = diag([1 -1 -1]);
    R = QDISP * quat2rotm_local(q);
    set(hBox,  'Vertices', (R * V0')');
    set(hNose, 'Vertices', (R * Vn0')');
    AXLEN = 2.4;
    ex = R*[AXLEN;0;0]; ey = R*[0;AXLEN;0]; ez = R*[0;0;AXLEN];
    set(hArrows(1), 'XData', [0 ex(1)], 'YData', [0 ex(2)], 'ZData', [0 ex(3)]);
    set(hArrows(2), 'XData', [0 ey(1)], 'YData', [0 ey(2)], 'ZData', [0 ey(3)]);
    set(hArrows(3), 'XData', [0 ez(1)], 'YData', [0 ez(2)], 'ZData', [0 ez(3)]);
end
