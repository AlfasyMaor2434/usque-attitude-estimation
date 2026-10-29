function out = replay_phone_usque(rawFile, overrides)
% REPLAY_PHONE_USQUE  Re-run USQUE_AHRS offline on a usque_phone_raw_<ts>.csv
% written by live_phone_usque (RECORD_RAW = true). The rows are replayed in
% the order the live loop executed them (same trapezoidal gyro integration,
% same update instants), so with no overrides the result matches the live
% run; pass a struct of filter-parameter overrides to try other settings.
% (Recordings made up to 5 Oct 2026 can differ from their live log: the live
% loop of that time integrated gyro samples that the app delivered twice;
% they are dropped here and in the live loop now.)
%
%   out = replay_phone_usque('usque_phone_raw_20261001_120000.csv');
%   out = replay_phone_usque(f, struct('P0_bg', 1e-6));          % old bias prior
%   out = replay_phone_usque(f, struct('mag_noise_var', 0.3));
%
% out: t (s), yaw/roll/pitch (deg), bgZ (deg/s), app azimuth (deg), and the
% header values. Also plots yaw vs the app azimuth and the z gyro bias.

if nargin < 2, overrides = struct(); end

% ---- header
hdr = struct();
fid = fopen(rawFile, 'r');
if fid < 0, error('Cannot open %s', rawFile); end
nHead = 0;
while true
    ln = fgetl(fid);
    nHead = nHead + 1;   % ends on the column-name line
    if ~ischar(ln) || ~startsWith(ln, '#'), break; end
    parts = strsplit(strtrim(ln(2:end)), ',');
    key = strtrim(parts{1});
    if numel(parts) > 1 && isvarname(key)
        hdr.(key) = str2double(parts(2:end));
    end
end
fclose(fid);

% ---- rows: type,seg,t,c1..c9 (G rows have 3 values, U rows 9)
names = {'type','seg','t','c1','c2','c3','c4','c5','c6','c7','c8','c9'};
opts = delimitedTextImportOptions('NumVariables', 12, 'Delimiter', ',', ...
    'DataLines', [nHead + 1, Inf], 'VariableNames', names, ...
    'VariableTypes', [{'string'}, repmat({'double'}, 1, 11)], ...
    'MissingRule', 'fill', 'ExtraColumnsRule', 'ignore');
T = readtable(rawFile, opts);
typ = string(T.type);
V = [T.c1 T.c2 T.c3 T.c4 T.c5 T.c6 T.c7 T.c8 T.c9];

% ---- filter, rebuilt as the live run built it
f = USQUE_AHRS();
p = f.P;
p.mag_inclination_deg = hdr.mag_inclination_deg;
p.mag_declination_deg = hdr.mag_declination_deg;
p.mag_ref_from_boot   = logical(hdr.mag_ref_from_boot);
p.use_mag_bias_state  = logical(hdr.use_mag_bias_state);
p.P0_bg               = hdr.P0_bg;
fn = fieldnames(overrides);
for i = 1:numel(fn), p.(fn{i}) = overrides.(fn{i}); end
if isfield(hdr, 'mag_noise_var') && ~isfield(overrides, 'mag_noise_var')
    p.mag_noise_var = hdr.mag_noise_var;
end
f.setParams(p);
if isfield(hdr, 'init_heading_deg') && isfinite(hdr.init_heading_deg)
    % same start as the live run: heading seeded once, magnetic reference
    % rebuilt from that attitude
    f.beginWithHeading(hdr.boot_accel_frd, hdr.boot_mag_frd_calibrated, hdr.init_heading_deg);
    q0 = f.getQuaternion(); w=q0(1); x=q0(2); y=q0(3); z=q0(4);
    R0 = [1-2*(y^2+z^2), 2*(x*y-w*z), 2*(x*z+w*y); 2*(x*y+w*z), 1-2*(x^2+z^2), 2*(y*z-w*x); ...
          2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x^2+y^2)];
    m0 = hdr.boot_mag_frd_calibrated(:);
    f.setMagRef(R0 * (m0 / norm(m0)));
else
    f.begin(hdr.boot_accel_frd, hdr.boot_mag_frd_calibrated);
end
f.setGyroBias(hdr.boot_gyro_frd);

off = hdr.mag_cal_offset_frd; sc = hdr.mag_cal_scale_frd;
frd = @(v) [v(2), v(1), -v(3)];   % same as remapDeviceToFRD

nU = sum(typ == "U");
out.t = nan(nU,1); out.roll = out.t; out.pitch = out.t; out.yaw = out.t; out.bgZ = out.t; out.appAz = out.t;
lastT = []; lastG = []; lastSeg = -1; k = 0; tOff = 0; tSeg0 = NaN; tPrevSegEnd = 0;
for r = 1:height(T)
    seg = T.seg(r);
    if typ(r) == "G"
        tg = T.t(r); gi = frd(V(r,1:3))';
        if seg ~= lastSeg
            % timestamps restart after each discardlogs (the live loop skips the
            % first interval of a new segment too); keep a continuous clock for plotting
            tOff = tPrevSegEnd; tSeg0 = tg;
            lastT = []; lastSeg = seg;
        end
        if ~isempty(lastT) && tg <= lastT
            continue;    % sample delivered a second time by the app: dropped, as the live loop does
        end
        if ~isempty(lastT)
            dtg = tg - lastT;
            if dtg > 0 && dtg < 0.5
                nSub = ceil(dtg / 0.2);      % predict() ignores steps longer than 0.2 s
                for kSub = 1:nSub, f.predict(0.5 * (gi + lastG), dtg / nSub); end
            end
        end
        lastT = tg; lastG = gi; tPrevSegEnd = tOff + tg - tSeg0;
    else
        accel = frd(V(r,1:3))';
        mag   = ((frd(V(r,4:6)) - off) .* sc)';
        f.update(accel, mag, 0);
        k = k + 1;
        [out.roll(k), out.pitch(k), out.yaw(k)] = f.getEulerDeg();
        b = rad2deg(f.getGyroBias()); out.bgZ(k) = b(3);
        out.t(k) = tPrevSegEnd; out.appAz(k) = V(r,7);
    end
end
out.hdr = hdr; out.params = p;

d = mod(out.yaw - out.appAz + 180, 360) - 180;
fprintf('replay: %d updates over %.0f s | final bgZ %.4f deg/s (max |bgZ| %.4f) | yaw-appAz median %.1f deg, spread (p95-p5) %.1f deg\n', ...
    k, out.t(k), out.bgZ(k), max(abs(out.bgZ)), median(d, 'omitnan'), prctile(d, 95) - prctile(d, 5));

figure('Name', 'replay_phone_usque');
subplot(3,1,1); plot(out.t, out.yaw, out.t, mod(out.appAz + 180, 360) - 180, '--'); grid on;
ylabel('deg'); legend('filter yaw', 'app azimuth'); title(rawFile, 'Interpreter', 'none');
subplot(3,1,2); plot(out.t, d); grid on; ylabel('yaw - app az (deg)');
subplot(3,1,3); plot(out.t, out.bgZ); grid on; ylabel('bgZ (deg/s)'); xlabel('t (s)');
end
