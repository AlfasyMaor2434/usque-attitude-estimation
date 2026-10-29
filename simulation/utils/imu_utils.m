function varargout = imu_utils(op, varargin)
% IMU_UTILS  Central helper for various IMU-related utilities.
% ROBUST VERSION: Forces consistency between Euler and Quaternion math.

switch op
    case 'accmag_to_euler'
        [phi,theta,psi] = accmag_to_euler(varargin{:});
        varargout = {phi,theta,psi};
        
    case 'wrapToPi'
        y = wrapToPi(varargin{1});
        varargout = {y};
        
    case 'quat2R'
        % Returns Rotation Matrix from Body to NED
        R = quat2R(varargin{1});
        varargout = {R};
        
    case 'quat2eulerZYX'
        euler = quat2eulerZYX(varargin{1});
        varargout = {euler};
        
    case 'eulerZYX2quat'
        q = eulerZYX2quat_impl(varargin{1});
        varargout = {q};
        
    case 'eulerZYX_to_R'
        R = eulerZYX_to_R(varargin{:}); 
        varargout = {R};
        
    case 'numerical_jacobian'
        J = numerical_jacobian(varargin{:});
        varargout = {J};
        
    case 'f_euler'
        x_next = f_euler(varargin{:});
        varargout = {x_next};
        
    case 'h_euler'
        z = h_euler(varargin{:});
        varargout = {z};
        
    case 'f_quat'
        x_next = f_quat(varargin{:});
        varargout = {x_next};
        
    case 'h_quat'
        z = h_quat(varargin{:});
        varargout = {z};
        
    otherwise
        error('imu_utils:UnknownOp', 'Unknown operation "%s"', op);
end
end

%% ================= LOCAL IMPLEMENTATIONS =================

function [phi,theta,psi] = accmag_to_euler(acc, mag, ~)
% 1. Accelerometer -> Roll/Pitch
% acc measures -g (UP) in body frame when flat.
ax = -acc(1);
ay = -acc(2);
az = -acc(3);

phi   = atan2(ay, az);
theta = atan( -ax / sqrt(ay^2 + az^2) );

% 2. Magnetometer -> Yaw
mx = mag(1); my = mag(2); mz = mag(3);

% De-rotate mag vector to horizontal plane
cph = cos(phi);   sph = sin(phi);
cth = cos(theta); sth = sin(theta);

% Correct projection to horizontal plane
Bfy = my*cph - mz*sph;
Bfx = mx*cth + my*sph*sth + mz*cph*sth;

psi = atan2(-Bfy, Bfx);
end

function y = wrapToPi(x)
y = mod(x + pi, 2*pi) - pi;
end

function R = quat2R(q)
% Standard Body-to-NED DCM from Quaternion [w, x, y, z]
qw = q(1); qx = q(2); qy = q(3); qz = q(4);

R = zeros(3,3);

R(1,1) = 1 - 2*(qy^2 + qz^2);
R(1,2) = 2*(qx*qy - qw*qz);
R(1,3) = 2*(qx*qz + qw*qy);

R(2,1) = 2*(qx*qy + qw*qz);
R(2,2) = 1 - 2*(qx^2 + qz^2);
R(2,3) = 2*(qy*qz - qw*qx);

R(3,1) = 2*(qx*qz - qw*qy);
R(3,2) = 2*(qy*qz + qw*qx);
R(3,3) = 1 - 2*(qx^2 + qy^2);
end

function euler = quat2eulerZYX(q)
% Convert Quat to Euler ZYX
if size(q,2) == 1, q = reshape(q,4,1); end
N = size(q,2);
euler = zeros(3,N);

for k = 1:N
    R = quat2R(q(:,k));
    
    r31 = R(3,1);
    r32 = R(3,2);
    r33 = R(3,3);
    r21 = R(2,1);
    r11 = R(1,1);
    
    theta = asin(-r31);
    if abs(r31) > 1, theta = asin(-sign(r31)); end
    
    phi   = atan2(r32, r33);
    psi   = atan2(r21, r11);
    
    euler(:,k) = [phi; theta; psi];
end
end

function R_nb = eulerZYX_to_R(phi,theta,psi)
% Body-to-NED rotation matrix from Euler angles
% CRITICAL FIX: Instead of manual matrix (prone to typos), 
% we convert Euler->Quat->Matrix. This guarantees consistency 
% with the Quaternion filters.

q = eulerZYX2quat_impl([phi; theta; psi]);
R_nb = quat2R(q);

end

function J = numerical_jacobian(fun,x)
y0 = fun(x);
nx = numel(x);
ny = numel(y0);
J  = zeros(ny,nx);
eps_step = 1e-6;
for i = 1:nx
    xp = x;
    xp(i) = xp(i) + eps_step;
    y1 = fun(xp);
    J(:,i) = (y1 - y0)/eps_step;
end
end

%% --- FILTER DYNAMICS / MEASUREMENT ---

function x_next = f_euler(x,u,dt)
phi=x(1); theta=x(2); psi=x(3);
bg=x(4:6);
w = u - bg;
wx=w(1); wy=w(2); wz=w(3);

tt = tan(theta); ct = cos(theta);
% Check for singularity
if abs(ct) < 1e-4, ct = 1e-4; end

phi_dot   = wx + wy*sin(phi)*tt + wz*cos(phi)*tt;
theta_dot =      wy*cos(phi)    - wz*sin(phi);
psi_dot   =      wy*sin(phi)/ct + wz*cos(phi)/ct;

x_next = x;
x_next(1) = phi   + phi_dot*dt;
x_next(2) = theta + theta_dot*dt;
x_next(3) = psi   + psi_dot*dt;
end

function z = h_euler(x,p)
phi=x(1); theta=x(2); psi=x(3);
% R_nb: Body to NED
R_nb = eulerZYX_to_R(phi,theta,psi);

% Sensors are in Body. Vectors are in NED.
% v_body = R_nb' * v_ned
g_ned = [0;0;p.g];
acc_pred = R_nb' * (-g_ned); 

mag_ned  = p.mag_field_ned;
mag_pred = R_nb' * mag_ned;

z = [acc_pred; mag_pred];
end

function x_next = f_quat(x,u,dt)
q=x(1:4); bg=x(5:7);
w = u - bg;
Omega = [0 -w(1) -w(2) -w(3);
         w(1) 0 w(3) -w(2);
         w(2) -w(3) 0 w(1);
         w(3) w(2) -w(1) 0];
q_dot = 0.5 * Omega * q;
q_next = q + q_dot*dt;
q_next = q_next / norm(q_next);
x_next = [q_next; bg];
end

function z = h_quat(x,p)
q = x(1:4);
R_nb = quat2R(q); 

g_ned = [0;0;p.g];
acc_pred = R_nb' * (-g_ned);

mag_ned  = p.mag_field_ned;
mag_pred = R_nb' * mag_ned;

z = [acc_pred; mag_pred];
end

function q = eulerZYX2quat_impl(euler)
phi=euler(1); theta=euler(2); psi=euler(3);
cy = cos(psi/2);  sy = sin(psi/2);
cp = cos(theta/2); sp = sin(theta/2);
cr = cos(phi/2);  sr = sin(phi/2);
q0 = cy*cp*cr + sy*sp*sr;
q1 = cy*cp*sr - sy*sp*cr;
q2 = cy*sp*cr + sy*cp*sr;
q3 = sy*cp*cr - cy*sp*sr;
q = [q0; q1; q2; q3];
end