classdef USQUE_AHRS < handle
% USQUE_AHRS  MATLAB port of arduino/USQUE_AHRS/USQUE_AHRS.h, itself a
% line-by-line port of the PC project's usque.py. See that header comment
% for the full algorithm description (GRP error state, pure unscented
% propagation, joint accel+mag unscented update, dpsi_mag magnetic-heading-
% bias state, adaptive-R soft inflation).
%
% This is a faithful numerical port, not a re-derivation: every formula
% below matches USQUE_AHRS.h's C++ line for line. The only DELIBERATE
% differences are ones MATLAB's native linear algebra makes redundant:
%   - enforcePD uses eig() instead of the embedded file's hand-rolled
%     12-sweep Jacobi rotation (no float/no-malloc/no-STL constraint here).
%   - getSigmaPoints uses chol() instead of the hand-rolled
%     Cholesky-Banachiewicz.
%   - update()'s gain uses MATLAB's backslash/mrdivide instead of the
%     hand-rolled 3x3 cofactor inverse / Gauss-Jordan.
% All three are the same mathematical operation, just using MATLAB's
% (more robust, double-precision) built-ins instead of an embedded-safe
% hand implementation.
%
% Frame: NED world (x-North, y-East, z-Down), FRD body (x-Fwd, y-Right,
% z-Down). Quaternion (w,x,y,z), Hamilton, BODY -> NAV (R_nb).
%
% Usage:
%   f = USQUE_AHRS();
%   f.begin(accelStillAvg, magStillAvg);   % TRIAD init from a still window
%   f.setGyroBias(gyroStillAvg);
%   ...
%   f.predict(gyro, dt);                   % every IMU sample
%   f.update(accel, mag, throttlePct);     % at a decimated rate; mag=[] for accel-only
%   q = f.getQuaternion();                 % [w;x;y;z]

    properties (Constant)
        L = 7
        NSIGMA = 15
        MAG_GATE_OK              = 0
        MAG_GATE_NOT_INITIALISED = 1
        MAG_GATE_ZERO_READING    = 2
        MAG_GATE_TRIGGERED       = 3
        IAE_THRESH_ = 5.99     % chi2(0.95, 2 dof)
        VIB_N = 64
    end

    properties
        P   % Params struct, see USQUE_AHRS.defaultParams()
    end

    properties (Access = private)
        q_ = [1;0;0;0]
        bg_ = [0;0;0]
        psiMag_ = 0
        magRef_ = [1;0;0]
        magInclRef_ = 0
        magNormRef_ = -1
        Pmat_ = zeros(7,7)
        initialised_ = false
        lastMagGateStatus_ = 1
        lastAccelInfl_ = 1
        lastMagInfl_ = 1
        Wm_
        Wc_
        nLambda_
        vibBuf_ = zeros(1,64)
        vibCount_ = 0
        vibHead_ = 0
        vibLastPushMs_ = 0
        vibCachedFloor_ = 1e-3
        epoch_
    end

    methods
        function obj = USQUE_AHRS()
            obj.epoch_ = tic;
            obj.setParams(USQUE_AHRS.defaultParams());
            obj.reset();
        end

        function setParams(obj, p)
            obj.P = p;
            [obj.nLambda_, obj.Wm_, obj.Wc_] = USQUE_AHRS.computeUkfWeights(p.alpha_ukf, p.beta_ukf, p.kappa_ukf);
            obj.setMagRefFromInclDecl(p.mag_inclination_deg, p.mag_declination_deg);
            obj.vibReset();
        end

        function setMagRef(obj, ref)
            ref = ref(:);
            n = norm(ref);
            if n < 1e-9, return; end
            obj.magRef_ = ref / n;
            obj.magInclRef_ = asin(max(-1, min(1, obj.magRef_(3))));
        end

        function setMagRefFromInclDecl(obj, inclDeg, declDeg)
            inc = deg2rad(inclDeg); dec = deg2rad(declDeg);
            ref = [cos(inc)*cos(dec); cos(inc)*sin(dec); sin(inc)];
            obj.setMagRef(ref);
        end

        % TRIAD init from a still-window accel+mag average (RAW, unnormalised).
        function begin(obj, accel, mag)
            obj.reset();   % already sets P0 with yaw sigma = P.P0_att_yaw_deg
            if obj.P.mag_ref_from_boot
                obj.calibrateMagRefFromBoot(accel(:), mag(:));
            end
            obj.q_ = USQUE_AHRS.triadToQuat(accel(:), mag(:), obj.magRef_);
            obj.initialised_ = true;
        end

        % TRIAD roll/pitch, yaw overridden ONCE from an externally-verified heading.
        function beginWithHeading(obj, accel, mag, headingDeg)
            obj.reset();
            if obj.P.mag_ref_from_boot
                obj.calibrateMagRefFromBoot(accel(:), mag(:));
            end
            q0 = USQUE_AHRS.triadToQuat(accel(:), mag(:), obj.magRef_);
            [roll, pitch, ~] = USQUE_AHRS.quatToEuler(q0);
            obj.q_ = USQUE_AHRS.eulerToQuat(roll, pitch, deg2rad(headingDeg));
            obj.bg_ = [0;0;0];
            obj.psiMag_ = 0;
            obj.initP0(2.0);
            obj.initialised_ = true;
        end

        % Externally-verified heading re-calibration, callable any time after begin().
        function setHeadingDeg(obj, headingDeg)
            if ~obj.initialised_, return; end
            [roll, pitch, ~] = USQUE_AHRS.quatToEuler(obj.q_);
            obj.q_ = USQUE_AHRS.eulerToQuat(roll, pitch, deg2rad(headingDeg));
            obj.Pmat_(1:3,1:3) = diag(repmat(deg2rad(2.0)^2, 3, 1));
        end

        function reset(obj)
            obj.q_ = [1;0;0;0];
            obj.bg_ = [0;0;0];
            obj.psiMag_ = 0;
            obj.Pmat_ = zeros(obj.L);
            obj.initP0(obj.P.P0_att_yaw_deg);
            obj.magNormRef_ = -1;
            obj.lastMagGateStatus_ = obj.MAG_GATE_NOT_INITIALISED;
            obj.lastAccelInfl_ = 1; obj.lastMagInfl_ = 1;
            obj.vibReset();
            obj.initialised_ = false;
        end

        function tf = initialised(obj)
            tf = obj.initialised_;
        end

        % Seed nominal gyro bias, e.g. from a still-window average of raw
        % gyro (true rate ~=0 while still, so the average IS the bias).
        % Call AFTER begin()/beginWithHeading() -- those call reset(), which
        % zeroes bg_.
        function setGyroBias(obj, b)
            obj.bg_ = b(:);
            for i = 1:3
                obj.Pmat_(3+i, 3+i) = obj.P.P0_bg;
            end
        end

        % ---------- PREDICT: call every IMU sample. gyro FRD rad/s, dt seconds. ----------
        function predict(obj, gyro, dt)
            if ~(dt > 0) || dt > 0.2, return; end
            gyro = gyro(:);

            chi = obj.getSigmaPoints(obj.Pmat_);   % NSIGMA x L

            if obj.P.use_mag_bias_state && obj.P.mag_bias_tau_s > 0
                psiDecay = exp(-dt / obj.P.mag_bias_tau_s);
            else
                psiDecay = 1;
            end

            qProp = zeros(obj.NSIGMA, 4);
            bProp = zeros(obj.NSIGMA, 3);
            psiProp = zeros(obj.NSIGMA, 1);
            for s = 1:obj.NSIGMA
                dq = USQUE_AHRS.errorVecToQuat(chi(s,1:3)');
                qi = USQUE_AHRS.quatNorm(USQUE_AHRS.quatMul(obj.q_, dq));
                bgi = obj.bg_ + chi(s,4:6)';
                qProp(s,:) = USQUE_AHRS.propagateQuat(qi, bgi, gyro, dt)';
                bProp(s,:) = bgi';
                psiProp(s) = (obj.psiMag_ + chi(s,7)) * psiDecay;
            end

            % sigma point 0 has zero error state -> qProp(1,:) IS the
            % propagated nominal (use_sigma0_reference=true, always).
            qNomNew = qProp(1,:)';
            bNomNew = (obj.Wm_' * bProp)';
            psiNomNew = obj.Wm_' * psiProp;

            qNomNewInv = [qNomNew(1); -qNomNew(2:4)];
            errPred = zeros(obj.NSIGMA, obj.L);
            for s = 1:obj.NSIGMA
                dqNew = USQUE_AHRS.quatNorm(USQUE_AHRS.quatMul(qNomNewInv, qProp(s,:)'));
                pv = USQUE_AHRS.quatToErrorVec(dqNew);
                errPred(s,1:3) = pv';
                errPred(s,4:6) = bProp(s,:) - bNomNew';
                errPred(s,7)   = psiProp(s) - psiNomNew;
            end

            errMean = (obj.Wm_' * errPred)';
            dqMean = USQUE_AHRS.errorVecToQuat(errMean(1:3));
            obj.q_ = USQUE_AHRS.quatNorm(USQUE_AHRS.quatMul(qNomNew, dqMean));
            obj.bg_ = bNomNew + errMean(4:6);
            obj.psiMag_ = psiNomNew + errMean(7);

            errPred = errPred - repmat(errMean', obj.NSIGMA, 1);

            % Q: attitude white noise + bias random walk + Van Loan cross
            % term + dpsi_mag's exact OU discretisation noise.
            qAtt = obj.P.gyro_noise_var*dt + (obj.P.bias_rw_var*dt^3)/3;
            Q = zeros(obj.L);
            for i = 1:3
                Q(i,i) = qAtt;
                Q(3+i,3+i) = obj.P.bias_rw_var*dt;
            end
            crossv = -(obj.P.bias_rw_var*dt^2)/2;
            for i = 1:3
                Q(i,3+i) = crossv; Q(3+i,i) = crossv;
            end
            if obj.P.use_mag_bias_state
                Q(7,7) = deg2rad(obj.P.mag_bias_sigma_ss_deg)^2 * (1 - psiDecay^2);
            end

            Ppred = zeros(obj.L);
            for s = 1:obj.NSIGMA
                Ppred = Ppred + obj.Wc_(s) * (errPred(s,:)' * errPred(s,:));
            end
            Ppred = Ppred + Q;
            obj.Pmat_ = USQUE_AHRS.enforcePD(Ppred);
        end

        % ---------- UPDATE: joint accel(+mag) unscented measurement update.
        % Call at a decimated rate (e.g. ~30 Hz), not every predict().
        % mag = [] for an accel-only update. throttlePct in [0,1], pass 0
        % if you have no vibration proxy for this run. ----------
        function update(obj, accel, mag, throttlePct)
            accel = accel(:);
            hasMag = ~isempty(mag);
            measDim = 3; if hasMag, measDim = 6; end

            chi = obj.getSigmaPoints(obj.Pmat_);

            normA = norm(accel);
            if normA < 1e-9, return; end
            aUnit = accel / normA;

            mUnit = zeros(3,1); normM = 0;
            if hasMag
                mag = mag(:);
                normM = norm(mag);
                if normM < 1e-9
                    hasMag = false; measDim = 3;
                    obj.lastMagGateStatus_ = obj.MAG_GATE_ZERO_READING;
                else
                    mUnit = mag / normM;
                end
            end
            if ~obj.initialised_, return; end

            yFull = [aUnit; mUnit];

            Z = zeros(obj.NSIGMA, 6);
            for s = 1:obj.NSIGMA
                dq = USQUE_AHRS.errorVecToQuat(chi(s,1:3)');
                qi = USQUE_AHRS.quatNorm(USQUE_AHRS.quatMul(obj.q_, dq));
                Z(s,1:3) = USQUE_AHRS.hAccel(qi)';
                if hasMag
                    if obj.P.use_mag_bias_state
                        magRefI = USQUE_AHRS.rotateYawRef(obj.magRef_, obj.psiMag_ + chi(s,7));
                    else
                        magRefI = obj.magRef_;
                    end
                    Z(s,4:6) = USQUE_AHRS.hMag(qi, magRefI)';
                end
            end

            y = yFull(1:measDim);
            zbar = (obj.Wm_' * Z(:,1:measDim))';
            innov = y - zbar;

            PzzNoR = zeros(measDim); Pxz = zeros(obj.L, measDim);
            for s = 1:obj.NSIGMA
                dz = Z(s,1:measDim)' - zbar;
                PzzNoR = PzzNoR + obj.Wc_(s) * (dz*dz');
                Pxz = Pxz + obj.Wc_(s) * (chi(s,:)' * dz');
            end

            Rbase = zeros(measDim);
            for i = 1:3, Rbase(i,i) = obj.P.accel_noise_var; end
            if hasMag, for i = 4:6, Rbase(i,i) = obj.P.mag_noise_var; end, end

            accelInfl = 1; magInfl = 1;
            obj.lastMagGateStatus_ = obj.MAG_GATE_NOT_INITIALISED;  % reused as "n/a, accel-only", matching the .h
            if hasMag, obj.lastMagGateStatus_ = obj.MAG_GATE_OK; end

            % ---- IAE: per-channel innovation chi-square test ----
            if obj.P.use_iae
                SaccBlk = PzzNoR(1:3,1:3) + Rbase(1:3,1:3);
                if rcond(SaccBlk) > 1e-15
                    nb = innov(1:3)' / SaccBlk * innov(1:3);
                    if nb > obj.IAE_THRESH_
                        accelInfl = max(accelInfl, min(nb/obj.IAE_THRESH_, obj.P.max_iae_inflation));
                    end
                end
                if hasMag
                    SmagBlk = PzzNoR(4:6,4:6) + Rbase(4:6,4:6);
                    if rcond(SmagBlk) > 1e-15
                        nbM = innov(4:6)' / SmagBlk * innov(4:6);
                        if nbM > obj.IAE_THRESH_
                            magInfl = max(magInfl, min(nbM/obj.IAE_THRESH_, obj.P.max_iae_inflation));
                        end
                    end
                end
            end

            % ---- PHYSICAL: angle between measured and predicted specific force ----
            if obj.P.phys_dir_thresh_deg > 0
                cosang = max(-1, min(1, aUnit' * Z(1,1:3)'));
                angDeg = rad2deg(acos(cosang));
                if angDeg > obj.P.phys_dir_thresh_deg
                    r = angDeg / obj.P.phys_dir_thresh_deg;
                    accelInfl = max(accelInfl, min(r*r, obj.P.max_phys_inflation));
                end
            end

            % ---- MAG_GATE: field magnitude + inclination vs this run's own baseline ----
            if hasMag
                if obj.magNormRef_ <= 0
                    obj.magNormRef_ = normM;
                else
                    magFracDev = abs(normM - obj.magNormRef_) / obj.magNormRef_;
                    downEst = -Z(1,1:3)';
                    inclMeas = asin(max(-1, min(1, mUnit' * downEst)));
                    inclDev = abs(inclMeas - obj.magInclRef_);
                    if magFracDev > obj.P.mag_gate_mag_frac || rad2deg(inclDev) > obj.P.mag_gate_incl_deg
                        magInfl = max(magInfl, obj.P.max_mag_gate_inflation);
                        obj.lastMagGateStatus_ = obj.MAG_GATE_TRIGGERED;
                    end
                end
            end

            % ---- VIBRATION: accel high-frequency residual vs a causal rolling baseline ----
            if obj.P.use_vibration && throttlePct > 0
                vibAcc = min(1 + 10*throttlePct^2, obj.P.max_vib_inflation);
                accelInfl = max(accelInfl, vibAcc);
                if hasMag
                    vibMag = min(1 + 50*throttlePct, obj.P.max_vib_inflation_mag);
                    magInfl = max(magInfl, vibMag);
                end
            end

            obj.lastAccelInfl_ = accelInfl; obj.lastMagInfl_ = magInfl;

            Radaptive = Rbase;
            Radaptive(1:3,1:3) = Radaptive(1:3,1:3) * accelInfl;
            if hasMag, Radaptive(4:6,4:6) = Radaptive(4:6,4:6) * magInfl; end

            Pzz = PzzNoR + Radaptive;
            if rcond(Pzz) < 1e-15, return; end
            K = Pxz / Pzz;   % L x measDim == Pxz * inv(Pzz)

            % ACCEL-ONLY: project the unobservable rotation-about-gravity
            % component out of the attitude AND gyro-bias gain rows (NOT
            % dpsi_mag -- it's a scalar, no body-frame "direction" to
            % project against; see USQUE_AHRS.h's update() comment).
            if ~hasMag && obj.P.accel_only_projection
                d_b = -Z(1,1:3)';
                dn = norm(d_b);
                if dn > 1e-9
                    d_b = d_b / dn;
                    for blk = 0:1
                        r0 = blk*3;
                        for j = 1:measDim
                            dj = d_b' * K(r0+1:r0+3, j);
                            K(r0+1:r0+3, j) = K(r0+1:r0+3, j) - d_b*dj;
                        end
                    end
                end
            end

            dx = K * innov;
            dq = USQUE_AHRS.errorVecToQuat(dx(1:3));
            obj.q_ = USQUE_AHRS.quatNorm(USQUE_AHRS.quatMul(obj.q_, dq));
            obj.bg_ = obj.bg_ + dx(4:6);
            obj.psiMag_ = obj.psiMag_ + dx(7);

            % General covariance form (valid for any gain -- needed because
            % K may have been projected above, so it's not necessarily optimal):
            %   P = P - K*Pxz' - Pxz*K' + K*Pzz*K'
            KPxzT = K * Pxz';
            KPzzKT = K * Pzz * K';
            Pnew = obj.Pmat_ - KPxzT - KPxzT' + KPzzKT;
            obj.Pmat_ = USQUE_AHRS.enforcePD(Pnew);
        end

        % Call once per loop, before update(), with the RAW and low-pass-
        % filtered accel for this sample -- feeds VibBaseline. Returns the
        % throttlePct value to pass to update(). Returns 0 if use_vibration is off.
        function throttlePct = updateVibrationProxy(obj, accelRaw, accelLp)
            if ~obj.P.use_vibration, throttlePct = 0; return; end
            d = accelRaw(:) - accelLp(:);
            energy = norm(d);
            throttlePct = obj.vibPush(energy, obj.P.vib_baseline_pct, obj.P.vib_rel_span);
        end

        % ---------- accessors ----------
        function q = getQuaternion(obj), q = obj.q_; end
        function [roll, pitch, yaw] = getEulerDeg(obj)
            [roll, pitch, yaw] = USQUE_AHRS.quatToEuler(obj.q_);
            roll = rad2deg(roll); pitch = rad2deg(pitch); yaw = rad2deg(yaw);
        end
        function b = getGyroBias(obj), b = obj.bg_; end
        % 1-sigma of the gyro-bias estimate [rad/s], from the filter's own covariance
        function s = getGyroBiasStd(obj), s = sqrt(max(diag(obj.Pmat_(4:6,4:6)), 0)); end
        function d = getPsiMagDeg(obj), d = rad2deg(obj.psiMag_); end
        function s = getAttitudeStdDeg(obj)
            s = rad2deg(sqrt(max(diag(obj.Pmat_(1:3,1:3)), 0)));
        end
        function d = getPsiMagStdDeg(obj), d = rad2deg(sqrt(max(obj.Pmat_(7,7), 0))); end
        function s = getLastMagGateStatus(obj), s = obj.lastMagGateStatus_; end
        function v = getLastAccelInflation(obj), v = obj.lastAccelInfl_; end
        function v = getLastMagInflation(obj), v = obj.lastMagInfl_; end
        % this run's self-calibrated |mag| baseline MAG_GATE compares
        % against (<=0 = not yet set, i.e. no mag sample fused yet)
        function v = getMagNormRef(obj), v = obj.magNormRef_; end
        % inclination the filter is actually using as its reference (deg)
        function v = getMagInclRefDeg(obj), v = rad2deg(obj.magInclRef_); end
    end

    methods (Access = private)
        % Replace the model (WMM) inclination and the |mag| baseline with the
        % values actually measured in the boot still window. dpsi_mag can
        % only absorb a HEADING (horizontal) bias -- if the magnetometer's
        % real inclination differs from the model's (residual hard-iron,
        % indoor field distortion, phone mag calibration), the vertical part
        % of every mag innovation is permanently unexplainable: MAG_GATE/IAE
        % fire on every sample, and the persistent residual leaks through the
        % UT cross-covariances into yaw/dpsi_mag/gyro-bias as a steady drift.
        % Declination is kept from the params (a still window can't observe it).
        function calibrateMagRefFromBoot(obj, accel, mag)
            na = norm(accel); nm = norm(mag);
            if na < 1e-9 || nm < 1e-9, return; end
            down = -accel / na;   % specific force at rest points up
            incl = asin(max(-1, min(1, (mag / nm)' * down)));
            dec = deg2rad(obj.P.mag_declination_deg);
            obj.setMagRef([cos(incl)*cos(dec); cos(incl)*sin(dec); sin(incl)]);
            obj.magNormRef_ = nm;
        end

        function initP0(obj, yawDeg)
            obj.Pmat_ = zeros(obj.L);
            obj.Pmat_(1,1) = deg2rad(obj.P.P0_att_roll_deg)^2;
            obj.Pmat_(2,2) = deg2rad(obj.P.P0_att_pitch_deg)^2;
            obj.Pmat_(3,3) = deg2rad(yawDeg)^2;
            for i = 1:3
                obj.Pmat_(3+i, 3+i) = obj.P.P0_bg_no_static;
            end
            if obj.P.use_mag_bias_state
                obj.Pmat_(7,7) = deg2rad(obj.P.mag_bias_sigma_ss_deg)^2;
            else
                obj.Pmat_(7,7) = 1e-12;
            end
        end

        % ---- sigma points: Cholesky of nLambda_ * enforcePD(P) ----
        function chi = getSigmaPoints(obj, Pin)
            Psafe = USQUE_AHRS.enforcePD(Pin);
            scaled = obj.nLambda_ * Psafe;
            [Lm, failed] = chol(scaled, 'lower');
            if failed ~= 0
                Lm = diag(sqrt(max(diag(scaled), 1e-12)));
            end
            chi = zeros(obj.NSIGMA, obj.L);
            for i = 1:obj.L
                chi(1+i, :)          =  Lm(:,i)';
                chi(obj.L+1+i, :)    = -Lm(:,i)';
            end
        end

        % ---- Vibration relative-baseline: excursion above a causal rolling
        % percentile of recent accel high-frequency-residual energy. ----
        function vibReset(obj)
            obj.vibBuf_ = zeros(1, obj.VIB_N);
            obj.vibCount_ = 0; obj.vibHead_ = 0;
            obj.vibLastPushMs_ = 0; obj.vibCachedFloor_ = 1e-3;
        end

        function exc = vibPush(obj, energy, pct, span)
            nowMs = 1000 * toc(obj.epoch_);
            if (nowMs - obj.vibLastPushMs_) >= 1000 || obj.vibCount_ == 0
                obj.vibLastPushMs_ = nowMs;
                obj.vibBuf_(obj.vibHead_ + 1) = energy;
                obj.vibHead_ = mod(obj.vibHead_ + 1, obj.VIB_N);
                if obj.vibCount_ < obj.VIB_N, obj.vibCount_ = obj.vibCount_ + 1; end
                if obj.vibCount_ >= 4
                    sorted = sort(obj.vibBuf_(1:obj.vibCount_));
                    idx = round(pct * (obj.vibCount_ - 1)) + 1;
                    obj.vibCachedFloor_ = max(sorted(idx), 1e-3);
                end
            end
            if obj.vibCount_ < 4, exc = 0; return; end
            e = (energy / obj.vibCachedFloor_ - 1) / max(span, 1e-6);
            exc = max(0, min(1, e));
        end
    end

    methods (Static)
        function p = defaultParams()
            p = struct();
            p.gyro_noise_var  = 9.0e-5;
            p.bias_rw_var     = 1.0e-10;
            p.accel_noise_var = 1.0e-2;
            p.mag_noise_var   = 0.05;
            p.alpha_ukf = 1.0; p.beta_ukf = 2.0; p.kappa_ukf = 1.0;
            p.P0_att_roll_deg = 2.0; p.P0_att_pitch_deg = 2.0; p.P0_att_yaw_deg = 10.0;
            p.P0_bg = 1.0e-6; p.P0_bg_no_static = 1.0e-4;
            p.use_mag_bias_state = true;
            p.mag_bias_tau_s = 300.0; p.mag_bias_sigma_ss_deg = 15.0;
            p.mag_gate_mag_frac = 0.15; p.mag_gate_incl_deg = 10.0; p.max_mag_gate_inflation = 50.0;
            p.phys_dir_thresh_deg = 5.0; p.max_phys_inflation = 5.0;
            p.use_vibration = true; p.vib_rel_span = 1.0;
            p.max_vib_inflation = 3.0; p.max_vib_inflation_mag = 8.0;
            p.vib_hp_cutoff_hz = 15.0; p.vib_baseline_window_s = 60.0; p.vib_baseline_pct = 0.20;
            p.use_iae = true; p.max_iae_inflation = 15.0;
            p.accel_only_projection = true;
            p.mag_inclination_deg = 46.281; p.mag_declination_deg = 0.0;
            % false = usque.py behaviour (model inclination). true = take
            % inclination + |mag| baseline from begin()'s still window.
            p.mag_ref_from_boot = false;
            p.g = 9.81;
        end

        % ---- Measurement model (matches usque.py's _h_from_q exactly) ----
        function ha = hAccel(q)
            w=q(1); x=q(2); y=q(3); z=q(4);
            ha = [ -(2*(x*z - w*y));
                   -(2*(w*x + y*z));
                   -(w*w - x*x - y*y + z*z) ];
        end
        function hm = hMag(q, ref)
            w=q(1); x=q(2); y=q(3); z=q(4);
            N=ref(1); E=ref(2); D=ref(3);
            hm = zeros(3,1);
            hm(1) = N*(w*w+x*x-y*y-z*z) + 2*E*(x*y+w*z) + 2*D*(x*z-w*y);
            hm(2) = 2*N*(x*y-w*z) + E*(w*w-x*x+y*y-z*z) + 2*D*(y*z+w*x);
            hm(3) = 2*N*(x*z+w*y) + 2*E*(y*z-w*x) + D*(w*w-x*x-y*y+z*z);
        end
        function out = rotateYawRef(ref, psi)
            c = cos(psi); s = sin(psi);
            N=ref(1); E=ref(2); D=ref(3);
            out = [c*N - s*E; s*N + c*E; D];
        end

        % ---- TRIAD: body->nav quaternion from a specific-force sample and
        % a mag sample (two-vector TRIAD, gravity primary, mag secondary;
        % magRef's horizontal direction encodes declination). ----
        function q = triadToQuat(accel, mag, magRef)
            a = -accel(:); an = norm(a);
            if an < 1e-9, q = [1;0;0;0]; return; end
            a = a / an;
            m = mag(:); mn = norm(m);
            if mn < 1e-9, q = [1;0;0;0]; return; end
            m = m / mn;

            b1 = a;
            b2 = cross(b1, m); b2n = norm(b2);
            if b2n < 1e-9, q = [1;0;0;0]; return; end
            b2 = b2 / b2n;
            b3 = cross(b1, b2);

            n1 = [0;0;1];
            n2 = cross(n1, magRef(:)); n2n = norm(n2);
            if n2n < 1e-9, q = [1;0;0;0]; return; end
            n2 = n2 / n2n;
            n3 = cross(n1, n2);

            % R_nb (body->nav): R_nb*b_i = n_i  =>  R = sum n_i (x) b_i
            R = n1*b1' + n2*b2' + n3*b3';
            q = USQUE_AHRS.quatFromR(R);
        end

        function q = quatFromR(R)
            tr = R(1,1) + R(2,2) + R(3,3);
            if tr > 0
                S = sqrt(tr+1)*2;
                qw=0.25*S; qx=(R(3,2)-R(2,3))/S; qy=(R(1,3)-R(3,1))/S; qz=(R(2,1)-R(1,2))/S;
            elseif R(1,1) > R(2,2) && R(1,1) > R(3,3)
                S = sqrt(1+R(1,1)-R(2,2)-R(3,3))*2;
                qw=(R(3,2)-R(2,3))/S; qx=0.25*S; qy=(R(1,2)+R(2,1))/S; qz=(R(1,3)+R(3,1))/S;
            elseif R(2,2) > R(3,3)
                S = sqrt(1+R(2,2)-R(1,1)-R(3,3))*2;
                qw=(R(1,3)-R(3,1))/S; qx=(R(1,2)+R(2,1))/S; qy=0.25*S; qz=(R(2,3)+R(3,2))/S;
            else
                S = sqrt(1+R(3,3)-R(1,1)-R(2,2))*2;
                qw=(R(2,1)-R(1,2))/S; qx=(R(1,3)+R(3,1))/S; qy=(R(2,3)+R(3,2))/S; qz=0.25*S;
            end
            q = USQUE_AHRS.quatNorm([qw;qx;qy;qz]);
        end

        % ---- GRP (Gibbs/Rodrigues, a=1 f=4) error <-> quaternion ----
        function q = errorVecToQuat(p)
            a = 1; f = 4;
            p2 = p(1)^2 + p(2)^2 + p(3)^2;
            q0 = (-a*p2 + f*sqrt(f*f + (1-a*a)*p2)) / (f*f + p2);
            k = (a + q0) / f;
            q = [q0; k*p(1); k*p(2); k*p(3)];
        end
        function p = quatToErrorVec(dq)
            a = 1; f = 4;
            if dq(1) < 0, dq = -dq; end
            denom = a + dq(1);
            if denom < 1e-8
                p = f * dq(2:4);
            else
                k = f / denom;
                p = k * dq(2:4);
            end
        end

        % Symmetrise + real eigenvalue clip (matches usque.py's _enforce_pd).
        function Mout = enforcePD(Min)
            A = 0.5*(Min + Min');
            [V, D] = eig(A);
            d = diag(D);
            d(d < 1e-9) = 1e-9;
            Mout = V * diag(d) * V';
            Mout = 0.5*(Mout + Mout');
        end

        % usque.py's exact formula: lambda = alpha^2*(L+kappa) - L (NOT
        % Merwe's kappa=3-n convention). Returns [L+lambda, Wm, Wc].
        function [nLambda, Wm, Wc] = computeUkfWeights(alpha, beta, kappa)
            L_ = USQUE_AHRS.L; N_ = USQUE_AHRS.NSIGMA;
            lambda = alpha^2*(L_ + kappa) - L_;
            nLambda = L_ + lambda;
            if abs(nLambda) > 1e-9, w0m = lambda/nLambda; else, w0m = 0; end
            w0c = w0m + (1 - alpha^2 + beta);
            if nLambda > 1e-9, wi = 1/(2*nLambda); else, wi = 0; end
            Wm = [w0m; repmat(wi, N_-1, 1)];
            Wc = [w0c; repmat(wi, N_-1, 1)];
        end

        % ---------- quaternion (w,x,y,z), Hamilton, body->nav ----------
        function q = quatNorm(q)
            n = norm(q);
            if n < 1e-12, q = [1;0;0;0]; return; end
            q = q / n;
        end
        function o = quatMul(a, b)
            o = [ a(1)*b(1) - a(2)*b(2) - a(3)*b(3) - a(4)*b(4);
                  a(1)*b(2) + a(2)*b(1) + a(3)*b(4) - a(4)*b(3);
                  a(1)*b(3) - a(2)*b(4) + a(3)*b(1) + a(4)*b(2);
                  a(1)*b(4) + a(2)*b(3) - a(3)*b(2) + a(4)*b(1) ];
        end
        % exact exp-map: q_new = normalize(q (x) dq(gyro-bg, dt))
        function qout = propagateQuat(q, bg, gyro, dt)
            w = gyro(:) - bg(:);
            ang = norm(w) * dt;
            if ang > 1e-8
                half = ang*0.5;
                cosTerm = cos(half);
                sinTerm = dt * sin(half) / ang;
            else
                half = ang*0.5;
                cosTerm = 1 - half*half*0.5;
                sinTerm = dt*0.5*(1 - (ang*ang)/24);
            end
            dq = [cosTerm; w(1)*sinTerm; w(2)*sinTerm; w(3)*sinTerm];
            qout = USQUE_AHRS.quatNorm(USQUE_AHRS.quatMul(q, dq));
        end
        function q = eulerToQuat(roll, pitch, yaw)
            cr=cos(0.5*roll);  sr=sin(0.5*roll);
            cp=cos(0.5*pitch); sp=sin(0.5*pitch);
            cy=cos(0.5*yaw);   sy=sin(0.5*yaw);
            q = [ cr*cp*cy + sr*sp*sy;
                  sr*cp*cy - cr*sp*sy;
                  cr*sp*cy + sr*cp*sy;
                  cr*cp*sy - sr*sp*cy ];
            q = USQUE_AHRS.quatNorm(q);
        end
        function [roll, pitch, yaw] = quatToEuler(q)
            w=q(1); x=q(2); y=q(3); z=q(4);
            roll = atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y));
            sp = 2*(w*y - z*x); sp = max(-1, min(1, sp));
            pitch = asin(sp);
            yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z));
        end
    end
end
