%% main.m
close all;
clc;

% Tracking two cats in a room
% Workspace : 4 x 6, height 4 [m]
% Camera pin hole location : (2,3,4)
% Total observation time
T0 = 0;
Tf = 50;
Nstep = 500;
delt = (Tf-T0)/Nstep;

% Target motion / F for first cat
xT = [2 2]’;
xTdot = [0.05, 0.1]’;

% Target motion / F for second cat
xT2 = [3 3]’; % initial position for second cat
xTdot2 = [-0.05, -0.1]’; % initial velocity for second cat

% Camera state
s = [2.3562, 2.5261, 0, 0]’;
u = [0, 0]’;
lambda = 4;
lambda_min = 2; % Define the minimum allowable lambda
lambda_max = 6; % Define the maximum allowable lambda
adjustment_rate = 0.1; % Define the rate of adjustment for lambda
error_threshold = 0.5; % Define a threshold beyond which the cat is considered to be too far

% Initialization
x_rec = [];
z_rec = [];
x_rec2 = []; % for the second cat
z_rec2 = []; % for the second cat
s_rec = [];
u_rec = [];
cam_rec = [];
for t=T0:delt:Tf
[z, z2] = measurement_cam(s, xT, xTdot, xT2, xTdot2, lambda); % Modify the function call % Adjust lambda based on measurements
if any(abs(z) > error_threshold)
if z(1) > error_threshold
lambda = min(lambda_max, lambda + adjustment_rate);
else
lambda = max(lambda_min, lambda - adjustment_rate);
end
end
u = controller(z, s, delt, u, lambda);
x_rec = [x_rec; xT’, xTdot’];
z_rec = [z_rec; z’];
x_rec2 = [x_rec2; xT2’, xTdot2’];
z_rec2 = [z_rec2; z2’];
s_rec = [s_rec; s’];
u_rec = [u_rec; u’];

% Record camera
psi = s(1,1);
phi = s(2,1);
R_phi = [1, 0, 0; 0, cos(phi), sin(phi); 0, -sin(phi), cos(phi)];
R_psi = [cos(psi), sin(psi), 0; -sin(psi), cos(psi), 0; 0, 0, 1];
C = [R_phi(3,:)*R_psi(:,1), R_phi(3,:)*R_psi(:,2), R_phi(3,:)*R_psi(:,3)];
P = [0, 0, lambda]’;
k = -lambda/(C*P);
cam = k*R_psi’*R_phi’*[0 0 lambda]’ - [0,0,-lambda]’;
cam_rec = [cam_rec; cam’];

% Update
s = kinematic_cam(s, u, delt);

% Random walk for first cat
theta = rand() * 2 * pi;
xTdot = 0.6 * [cos(theta), sin(theta)]’;
xT = xT + xTdot * delt;

% Random walk for second cat
theta2 = rand() * 2 * pi;
xTdot2 = 0.6 * [cos(theta2), sin(theta2)]’;
xT2 = xT2 + xTdot2 * delt;
end

% Plotting
figure;
plot(x_rec(:,1), x_rec(:,2), ’b’, ’LineWidth’, 2);
hold on;
plot(x_rec2(:,1), x_rec2(:,2), ’r’, ’LineWidth’, 2); % second cat
grid on;
title(’Target trajectory’);
xlabel(’$x_T$ (X in inertial frame)’, ’Interpreter’, ’latex’, ’FontSize’, 13);
ylabel(’$y_T$ (Y in inertial frame)’, ’Interpreter’, ’latex’, ’FontSize’, 13);
figure;
plot(z_rec(:,1), z_rec(:,2), ’b’, ’LineWidth’, 2);
hold on;
plot(z_rec2(:,1), z_rec2(:,2), ’r’, ’LineWidth’, 2); % second cat
grid on;
title(’Targets in camera frame’);
xlabel(’$p_x$’, ’Interpreter’, ’latex’, ’FontSize’, 13);
ylabel(’$p_y$’, ’Interpreter’, ’latex’, ’FontSize’, 13);
time = T0:delt:Tf;
figure();
subplot(2,1,1);
plot(time, s_rec(:,1)*180/pi);
title(’Camera angle’);
ylabel(’$\psi$’, ’Interpreter’, ’latex’,’FontSize’,13);
xlabel(’t (sec)’,’FontSize’,13);
grid on;
subplot(2,1,2);
plot(time, s_rec(:,2)*180/pi);
grid on;
ylabel(’$\phi$’, ’Interpreter’, ’latex’,’FontSize’,13);
xlabel(’t (sec)’,’FontSize’,13);
figure();
subplot(2,1,1);
plot(time, u_rec(:,1));
title(’Voltage input’);
ylabel(’$u_1$ (V)’, ’Interpreter’, ’latex’,’FontSize’,13);
xlabel(’t (sec)’,’FontSize’,13);
grid on;
subplot(2,1,2);
plot(time, u_rec(:,2));
ylabel(’$u_2$ (V)’, ’Interpreter’, ’latex’,’FontSize’,13);
xlabel(’t (sec)’,’FontSize’,13);
grid on;


%% controller.m
function u = controller(z,s,delt,u_prev,lambda)
b1 = 100 * pi/180; % given [rad/Vs^2]
b2 = 100 * pi/180; % given [rad/Vs^2]
%lambda = 4; % given [m]
PandT = desired_angle(z, s,lambda);
delpsi = PandT(1,1) - s(1,1);
delphi = PandT(2,1) - s(2,1);
u_now1 = (delpsi/delt - s(3,1))/b1;
u_now2 = (delphi/delt - s(4,1))/b2;
u_now = [u_now1 u_now2]’;
dudt = (u_now - u_prev)/delt;
kp1 = 0.3;
kp2 = 0.3;
kd1 = 0.03;
kd2 = 0.03;
u1 = kp1*u_now(1,1) + kd1*dudt(1,1);
u2 = kp2*u_now(2,1) + kd2*dudt(2,1);
% saturation
u1 = min(1, max(-1, u1));
u2 = min(1, max(-1, u2));
u = [u1 u2]’;
end


%% measurement_cam.m
function [z, z2] = measurement_cam(s, xT, xTdot, xT2, xTdot2, lambda)
% PT camera settings
% Camera position / F
xC = [0, 0, 4]’;
% Camera state
psi = s(1,1);
phi = s(2,1);
psiDot = s(3,1);
phiDot = s(4,1);
% Euler rotation matrices
R_phi = [1, 0, 0; 0, cos(phi), sin(phi); 0, -sin(phi), cos(phi)];
R_psi = [cos(psi), sin(psi), 0; -sin(psi), cos(psi), 0; 0, 0, 1];
% Target position / C for first cat
qT = R_phi * R_psi * ([xT’ 0]’ - xC);
qx = qT(1,1);
qy = qT(2,1);
qz = qT(3,1);
% Target position / C for second cat
qT2 = R_phi * R_psi * ([xT2’ 0]’ - xC);
qx2 = qT2(1,1);
qy2 = qT2(2,1);
qz2 = qT2(3,1);
% Target projection / VIP for first cat
pT = lambda * [qx/qz qy/qz]’;
px = pT(1,1);
py = pT(2,1);
% Target projection / VIP for second cat
pT2 = lambda * [qx2/qz2 qy2/qz2]’;
px2 = pT2(1,1);
py2 = pT2(2,1);
% Image Jacobian matrix for first cat
H = [-lambda/qz, 0, px/qz, px*py/lambda, -(lambda^2+px^2)/lambda, py;
0, -lambda/qz, py/qz, (lambda^2+px^2)/lambda, -px*py/lambda, -px];
% Image Jacobian matrix for second cat
H2 = [-lambda/qz2, 0, px2/qz2, px2*py2/lambda, -(lambda^2+px2^2)/lambda, py2;
0, -lambda/qz2, py2/qz2, (lambda^2+px2^2)/lambda, -px2*py2/lambda, -px2];
% Target projection speed / VIP for first cat
R_6 = [R_phi’*R_psi’, zeros(3,3); zeros(3,3), -R_phi’];
pTdot = H * R_6 * [xTdot’, 0, phiDot, 0, psiDot]’;
% Target projection speed / VIP for second cat
pTdot2 = H2 * R_6 * [xTdot2’, 0, phiDot, 0, psiDot]’;
z = [pT(:); pTdot(:)];
z2 = [pT2(:); pTdot2(:)];
end


%% desired_angle.m
function PandT = desired_angle(z, s, lambda)
% Find the coordinates in inertial frame
px = z(1,1);
py = z(2,1);
psi = s(1,1);
phi = s(2,1);

R_phi = [1, 0, 0; 0, cos(phi), sin(phi); 0, -sin(phi), cos(phi)];
R_psi = [cos(psi), sin(psi), 0; -sin(psi), cos(psi), 0; 0, 0, 1];

C = [R_phi(3,:)*R_psi(:,1), R_phi(3,:)*R_psi(:,2), R_phi(3,:)*R_psi(:,3)];
P = [px, py, lambda]’;
k = -lambda/(C*P);
xT_recon = k*R_psi’*R_phi’*[px py lambda]’ - [0,0,-lambda]’;

% Calcuate desired pan and tilt angle
pan = pi/2 + atan2(xT_recon(2,1),xT_recon(1,1));
tilt = pi/2 + atan2(4,sqrt(xT_recon(1,1)^2+xT_recon(2,1)^2));
PandT = [pan, tilt]’;
% Repeat for second target
px2 = z(3,1);
py2 = z(4,1);
xT2_recon = k * R_psi’ * R_phi’ * [px2 py2 lambda]’ - [0,0,-lambda]’;
pan2 = pi/2 + atan2(xT2_recon(2,1), xT2_recon(1,1));
tilt2 = pi/2 + atan2(4, sqrt(xT2_recon(1,1)^2 + xT2_recon(2,1)^2));
% Here you need a strategy. For simplicity, let’s average the desired angles.
avg_pan = (pan + pan2) / 2;
avg_tilt = (tilt + tilt2) / 2;
PandT = [avg_pan, avg_tilt]’;
end



%% kinematic_cam.m
function s_next = kinematic_cam(s_k, u_k, delt)
% Given constraint in the problem
psiDot_m = 100 * pi/180; % given [rad/s]
phiDot_m = 100 * pi/180; % given [rad/s]
b1 = 100 * pi/180; % given [rad/Vs^2]
b2 = 100 * pi/180; % given [rad/Vs^2]
% state-space form
A = [1,0,delt,0; 0,1,0,delt; 0,0,1,0; 0,0,0,1];
B = [0,0; 0,0; b1,0; 0,b2];
s_next_raw = A*s_k + B*u_k;
% constraint vectors
b1_vec = [0, pi/2, -psiDot_m, -phiDot_m]’;
b2_vec = [2*pi, pi, psiDot_m, phiDot_m]’;
% saturation
s_next = min(b2_vec, max(b1_vec, s_next_raw));
end
