function verify_replay_dump(rawFile, outFile, mode)
% VERIFY_REPLAY_DUMP  Replay a usque_phone_raw_*.csv through USQUE_AHRS.m without any
% plotting and write, per update, [q(4) bg(3) psiMag sqrt(diag(P))(7)] to outFile (CSV).
% Used by verify_port.py to compare the MATLAB port with usque.py sample by sample.
%
% mode: 'live'      mag on, dpsi_mag off, mag noise 0.05 (the phone's original settings)
%       'psi'       mag on, dpsi_mag ON (tau 300 s, sigma 15 deg)  -- exercises the 7th state
%       'accel'     accelerometer-only updates                      -- exercises the gain projection
%       'seeded'    start heading given (beginWithHeading + setMagRef), mag noise 3.0

hdr = struct();
fid = fopen(rawFile, 'r'); nHead = 0;
while true
    ln = fgetl(fid); nHead = nHead + 1;
    if ~ischar(ln) || ~startsWith(ln, '#'), break; end
    parts = strsplit(strtrim(ln(2:end)), ',');
    key = strtrim(parts{1});
    if numel(parts) > 1 && isvarname(key), hdr.(key) = str2double(parts(2:end)); end
end
fclose(fid);
names = {'type','seg','t','c1','c2','c3','c4','c5','c6','c7','c8','c9'};
opts = delimitedTextImportOptions('NumVariables', 12, 'Delimiter', ',', 'DataLines', [nHead + 1, Inf], ...
    'VariableNames', names, 'VariableTypes', [{'string'}, repmat({'double'}, 1, 11)], ...
    'MissingRule', 'fill', 'ExtraColumnsRule', 'ignore');
T = readtable(rawFile, opts);
isG = T.type == "G"; seg = T.seg; tt = T.t;
V = [T.c1 T.c2 T.c3 T.c4 T.c5 T.c6 T.c7 T.c8 T.c9];

f = USQUE_AHRS();
p = f.P;
p.mag_inclination_deg = hdr.mag_inclination_deg;
p.mag_declination_deg = hdr.mag_declination_deg;
p.mag_ref_from_boot   = true;
p.P0_bg               = hdr.P0_bg;
p.use_mag_bias_state  = strcmp(mode, 'psi');
p.mag_noise_var       = 0.05;
if strcmp(mode, 'seeded'), p.mag_noise_var = 3.0; end
f.setParams(p);

a0 = hdr.boot_accel_frd; m0 = hdr.boot_mag_frd_calibrated;
if strcmp(mode, 'seeded')
    k = find(~isG & ~isnan(V(:,7)), 1);
    heading = V(k,7) + hdr.mag_declination_deg;
    f.beginWithHeading(a0, m0, heading);
    q0 = f.getQuaternion(); w=q0(1); x=q0(2); y=q0(3); z=q0(4);
    R0 = [1-2*(y^2+z^2), 2*(x*y-w*z), 2*(x*z+w*y); 2*(x*y+w*z), 1-2*(x^2+z^2), 2*(y*z-w*x); ...
          2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x^2+y^2)];
    f.setMagRef(R0 * (m0(:) / norm(m0)));
else
    f.begin(a0, m0);
end
f.setGyroBias(hdr.boot_gyro_frd);

off = hdr.mag_cal_offset_frd; sc = hdr.mag_cal_scale_frd;
frd = @(v) [v(2), v(1), -v(3)];
nU = sum(~isG); out = zeros(nU, 15); k = 0;
lastT = []; lastG = []; lastSeg = -1;
for r = 1:height(T)
    if isG(r)
        tg = tt(r); gi = frd(V(r,1:3))';
        if seg(r) ~= lastSeg, lastT = []; lastSeg = seg(r); end
        if ~isempty(lastT)
            dtg = tg - lastT;
            if dtg > 0 && dtg < 0.5
                nSub = ceil(dtg / 0.2);      % predict() ignores steps longer than 0.2 s
                for kSub = 1:nSub, f.predict(0.5 * (gi + lastG), dtg / nSub); end
            end
        end
        lastT = tg; lastG = gi;
    else
        accel = frd(V(r,1:3))';
        if strcmp(mode, 'accel')
            f.update(accel, [], 0);
        else
            f.update(accel, ((frd(V(r,4:6)) - off) .* sc)', 0);
        end
        k = k + 1;
        out(k,:) = [f.getQuaternion()', f.getGyroBias()', deg2rad(f.getPsiMagDeg()), ...
                    deg2rad(f.getAttitudeStdDeg())', f.getGyroBiasStd()', deg2rad(f.getPsiMagStdDeg())];
    end
end
writematrix(out, outFile, 'Delimiter', ',');
fprintf('%s: %d updates written to %s\n', mode, k, outFile);
end
