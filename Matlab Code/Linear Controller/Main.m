%Clear workspace
clear; close all; clc;

%Define Orbital Constants
mu = 398600; %km^3/s^2
r_0 = 6678; %km

%Set Initial Condition (small perturbations about nominal orbit)
%x = [delta r; delta r_dot; delta theta; delta theta_dot]
delta_r0 = 0.1; %0.1 km = 100 m radial deviation
delta_rdot0 = 0.0; %km/s
delta_theta0 = deg2rad(0.5); %rad (0.5 degrees)
delta_thetad0 = 0.0; %rad/s

x_nom = [delta_r0; delta_rdot0; delta_theta0; delta_thetad0];

%Define A, B, C, D matrices (At nominal Orbit)
A = [0           1                    0  0;
     3*mu/r_0^3  0                    0  2*sqrt(mu/r_0);
     0           0                    0  1;
     0           -2*sqrt(mu/(r_0^5))  0  0];

B = [0 0;
     1 0;
     0 0;
     0 1/r_0];

C = [1 0 0 0;
     0 0 1 0];

D = zeros(2);
%% Part 1, Steps 1-3 - Linear System Analysis
%Find eigenvalues of A matrix (open-loop poles)
lambda = eig(A);
disp("Open Loop A Matrix Eigenvalues: ");
disp(lambda);

%Determine Reachability
P = ctrb(A, B);
rp = rank(P); %Full Rank == Fully Reachable
disp("Reachability Rank: " + rp + " (Should be 4)");

%Determine Observability
O = obsv(A, C);
ro = rank(O); %Full Rank == Fully Observable
disp("Observability Rank: " + ro + " (Should be 4)");

%Set up Linear System
sys = ss(A, B, C, D);

%Determine time vector for 2 orbits
T_orbit = 2*pi/sqrt(mu/r_0^3); %orbital period [s]
numOrbits = 5;
t = linspace(0, numOrbits*T_orbit, 1000); %Simulation time vector [s]

%Simulate zero-input response from perturbation initial condition
[y, tOut] = initial(sys, x_nom, t);


%% Part 2, Step 4 - Closed Loop Control
%Desired reference values
rhist = zeros(length(t),2);

%Define augmented matrices
Aaug = [ A zeros(4,2);
 -C zeros(2,2)];

Baug = [ B; zeros(2,2)];

Faug = [zeros(size(B)); eye(2)];

Caug = [ C zeros(2)];

Daug = zeros(size(Caug,1),size(Baug,2));

%Choose poles to satisfy requirements
poles = [-0.00123, -0.001, -0.0011, -0.00115, -0.00018, -0.0001];
Kaug = place(Aaug, Baug, poles);

%Define augmented closed loop system
AaugCL = Aaug - Baug*Kaug;
BaugCL = Faug;
sysCL = ss(AaugCL, BaugCL, Caug, Daug);

%Get response from reference profile
[yCLaug,~,xCLaug] = lsim(sysCL, rhist, t, [x_nom; zeros(2,1)]);

%Compute actuator efforts in each case, where uclaug = -K * xclaug
uCLaug = -(Kaug * xCLaug'); %units: km/s^2

%Convert control accelerations from km/s^2 to g's
g0 = 9.80665; %m/s^2
kmps2_to_g = 1000 / g0; %(km/s^2) * (1000 m/km) / (9.80665 m/s^2)
uCLaug_g = uCLaug * kmps2_to_g; %now in units of g

%% Part 2, Step 5 - Luenberger Observer
%Select observer poles
poles_luen = 5 * real(poles(1:4)); 

%Compute Observer Gain
L = place(A', C', poles_luen)';

%Extract feedback part from augmented gain
Kx = Kaug(:, 1:4); %Feedback gains
Ki = Kaug(:, 5:6); %Integrator gains

%Form Augmented Matrices
Acl_aug = [A - B*Kx, B*Kx    ;...
           zeros(4), A - L*C];
Bcl_aug = [B*Ki; zeros(4, size(Ki, 2))];
Ccl_aug = [C, zeros(size(C, 1), 4)];
Dcl_aug = zeros(size(Ccl_aug, 1), size(Bcl_aug, 2));

%Case 1 - Zero initial observer error
z0_case1 = [x_nom; zeros(4, 1)];

%Case 2: Non-zero observer error
percentError = 0.5;
z0_case2 = [x_nom; x_nom*percentError];

%Simulate Case 1
sysObs1 = ss(Acl_aug, Bcl_aug, Ccl_aug, Dcl_aug);
[~, ~, zObs1] = lsim(sysObs1, rhist, t, z0_case1);
x_true_case1 = zObs1(:,1:4);
e_hist_case1 = zObs1(:,5:8);
x_hat_case1 = x_true_case1 - e_hist_case1;
est_err_case1 = x_true_case1 - x_hat_case1;

%Simulate Case 2
sysObs2 = ss(Acl_aug, Bcl_aug, Ccl_aug, Dcl_aug);
[~, ~, zObs2] = lsim(sysObs2, rhist, t, z0_case2);
x_true_case2 = zObs2(:,1:4);
e_hist_case2 = zObs2(:,5:8);
x_hat_case2 = x_true_case2 - e_hist_case2;
est_err_case2 = x_true_case2 - x_hat_case2;

%Display Observer Poles
eig_obs = eig(A - L*C);
disp("Observer poles (A - L*C):");
disp(eig_obs);

%% Part 2, Step 6 - Infinite-Horizon Cost Function
%Define Max Allowable Deviations
xmax_delta_r = 0.20 * delta_r0; %km (20% of initial radial offset)
xmax_delta_r_dot = 1.0; %km/s
xmax_delta_theta = 0.20 * delta_theta0; %rad (20% of initial angle offset)
xmax_delta_theta_dot = 1e0; %rad/s

xmax_vec = [xmax_delta_r; xmax_delta_r_dot; xmax_delta_theta; xmax_delta_theta_dot];

%Define Max allowable control accelerations
umax_g = 0.01; %Max acceleration = 0.01 g
umax_kmps2 = umax_g * g0 / 1000; %Convert to km/s^2
umax_vec = umax_kmps2 * [1; 1];

%Define relative importance of each state (sum to 1)
alpha_states = [0.10; 0.05; 0.80; 0.05];

%Define relative importance of each input (sum to 1)
beta_inputs  = [0.5; 0.5]; %Equal weight thrusters

%Define overall tradeoff between state deviations and control effort
rho = 80;

%Build Q and R via Bryson's Rule
Q_lqr = diag((alpha_states ./ xmax_vec).^2);
R_lqr = rho * diag((beta_inputs  ./ umax_vec).^2);

%Calculate optimal gains
[K_lqr, S_lqr, ~] = lqr(A, B, Q_lqr, R_lqr);

%Compare closed-loop poles with manual design
eig_manual = eig(A - B*Kx);
eig_lqr = eig(A - B*K_lqr);

disp("Manual closed-loop poles (A - B*Kx):");
disp(eig_manual);

disp("LQR closed-loop poles (A - B*K_lqr):");
disp(eig_lqr);

%Create Luenberger Observer Augmented Dynamics
Acl_lqr_obs = [A - B*K_lqr, B * K_lqr;...
               zeros(4),    A - L*C];
Bcl_lqr_obs = zeros(8, size(rhist, 2));
Ccl_lqr_obs = eye(8);
Dcl_lqr_obs = zeros(8, size(rhist,2));

%Create State Space Model Object
sys_lqr_obs = ss(Acl_lqr_obs, Bcl_lqr_obs, Ccl_lqr_obs, Dcl_lqr_obs);

%Create 0% initial state error in observer for comparing response
percentError = 0.0;
z0_lqr = [x_nom; x_nom * percentError];

%Simulate LQR and observer closed loop
[~, ~, z_lqr] = lsim(sys_lqr_obs, rhist, t, z0_lqr);

%Pull out true and estimated states
x_true_lqr = z_lqr(:, 1:4); %True States
e_hist_lqr = z_lqr(:, 5:8); %Error States
x_hat_lqr  = x_true_lqr - e_hist_lqr; %Estimated = True - Error

%Compute LQR control forcing based on estimated state [Acceleration = km/s^2]
u_lqr = -(K_lqr * x_hat_lqr')';

%Covert to Gs of acceleration
u_lqr_g = u_lqr * kmps2_to_g;

%Manual controller (From Part 4) control based on estimated state (case 2 from part 5)
u_manual = uCLaug';
u_manual_g = u_manual * kmps2_to_g;

%Numerically integrate actuation acceleration to get "Total Energy"-ish comparison
E_manual = trapz(t, sum(u_manual.^2, 2)); %Manual state feedback
E_lqr = trapz(t, sum(u_lqr.^2,    2)); %LQR state feedback

disp("Manual controller energy: " + E_manual);
disp("LQR controller energy: " + E_lqr);

%Run LQR with 50% initial observer error
percentError = 0.5;
z0_lqr_err = [x_nom; x_nom*percentError];
[~,~,z_lqr_err] = lsim(sys_lqr_obs,rhist,t,z0_lqr_err);
x_true_lqr_err = z_lqr_err(:,1:4);


%% Produce Open Loop Response Figures
figure('Color','w');
tiles = tiledlayout('flow');

%Overall title for the tiled layout
axisFontSize = 16;
plotTitleFontSize = 18;
figureTitleFontSize = 22;
title(tiles, 'Open-Loop Perturbation Response Over Two Orbits', "Interpreter", "Latex", "FontSize", figureTitleFontSize);

%---RADIUS DEVIATION VERSUS TIME---
nexttile;
plot(tOut(tOut<2*T_orbit), y((tOut<2*T_orbit),1), 'LineWidth', 2);
grid on;
title('Radius Deviation vs Time', "Interpreter","Latex", "FontSize", plotTitleFontSize);
ylabel('$\delta r$ (km)', "Interpreter","Latex", "FontSize", axisFontSize);
xlabel('Time (s)', "Interpreter","Latex", "FontSize", axisFontSize);
axis padded;
xlim([0, 2*T_orbit]);

%---ANGLE DEVIATION VERSUS TIME---
nexttile;
plot(tOut(tOut<2*T_orbit), y((tOut<2*T_orbit),2), 'LineWidth', 2);
grid on;
title('Angle Deviation vs Time', "Interpreter","Latex", "FontSize", plotTitleFontSize);
ylabel('$\delta \theta$ (rad)', "Interpreter","Latex", "FontSize", axisFontSize);
xlabel('Time (s)', "Interpreter","Latex", "FontSize", axisFontSize);
axis padded;
xlim([0, 2*T_orbit]);

%% Produce Response Figure
figure('Color','w');
tiles = tiledlayout('flow');

%Overall title for the tiled layout
axisFontSize = 16;
plotTitleFontSize = 18;
figureTitleFontSize = 22;
title(tiles, 'Close-Loop Perturbation Response Over Two Orbits', "Interpreter", "Latex", "FontSize", figureTitleFontSize);

%-- RADIUS DEVIATION VERUS TIME PLOT --
nexttile;

%Add in Response Boundaries
radialStep = delta_r0;
plotColors = orderedcolors("gem");
patch([0, T_orbit, T_orbit, 0], [0, 0, radialStep * 0.05, radialStep*0.05], plotColors(3, :), "FaceAlpha", 0.2, "EdgeColor", "none");
hold on; box on;
patch([0, tOut(end), tOut(end), T_orbit, T_orbit, 0], [-radialStep*0.2, -radialStep*0.2, radialStep*0.2, radialStep*0.2, 0, 0], plotColors(4, :), "FaceAlpha", 0.2, "EdgeColor", "none")
yline(0, '--', 'LineWidth', 1.5, "FontSize", 14);
xline(T_orbit, '-', "1x Orbital Period", "LabelHorizontalAlignment","center", "LabelVerticalAlignment","top", "Interpreter", "latex", "LineWidth", 1, "FontSize", 14);
%xline(2*T_orbit, '-', "2x Orbital Period", "LabelHorizontalAlignment","left", "LabelVerticalAlignment","top", "Interpreter", "latex", "LineWidth", 1, "FontSize", 14);
yline(radialStep * 0.05, '-', "$5\%$ Settling Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","top", "Interpreter", "latex", "LineWidth", 1, "FontSize", 14);
yline(radialStep * 0.20, '-', "$20\%$ Undershoot Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","top", "Interpreter", "latex", "LineWidth", 1, "FontSize", 14);
yline(radialStep *-0.20, '-', "$20\%$ Overshoot Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","bottom", "Interpreter", "latex", "LineWidth", 1, "FontSize", 14);

%Plot Response Line
plot(tOut, yCLaug(:,1), 'LineWidth', 2);

%Plot Styling
grid on;
title('Radius Deviation vs Time', "Interpreter","Latex", "FontSize", plotTitleFontSize);
ylabel('$\delta r$ (km)', "Interpreter","Latex", "FontSize", axisFontSize);
xlabel('Time (s)', "Interpreter","Latex", "FontSize", axisFontSize);
xlim([tOut(1), tOut(end)]);
ylim([radialStep *-0.30, 1.1*abs(radialStep)]);

%-- ANGLE DEVIATION VERSUS TIME PLOT --
nexttile;

%Add in Response Boundaries
angularStep = delta_theta0;
plotColors = orderedcolors("gem");
patch([0, T_orbit, T_orbit, 0], [0, 0, angularStep * 0.05, angularStep*0.05], plotColors(3, :), "FaceAlpha", 0.2, "EdgeColor", "none");
hold on; box on;
patch([0, tOut(end), tOut(end), T_orbit, T_orbit, 0], [-angularStep*0.2, -angularStep*0.2, angularStep*0.2, angularStep*0.2, 0, 0], plotColors(4, :), "FaceAlpha", 0.2, "EdgeColor", "none")
yline(0, '--', 'LineWidth', 1.5, "FontSize", 14);
xline(T_orbit, '-', "1x Orbital Period", "LabelHorizontalAlignment","center", "LabelVerticalAlignment","top", "Interpreter", "latex", "LineWidth", 1, "FontSize", 14);
%xline(2*T_orbit, '-', "2x Orbital Period", "LabelHorizontalAlignment","left", "LabelVerticalAlignment","top", "Interpreter", "latex", "LineWidth", 1, "FontSize", 14);
yline(angularStep * 0.05, '-', "$5\%$ Settling Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","top", "Interpreter", "latex", "LineWidth", 1, "FontSize", 14);
yline(angularStep * 0.20, '-', "$20\%$ Undershoot Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","top", "Interpreter", "latex", "LineWidth", 1, "FontSize", 14);
yline(angularStep *-0.20, '-', "$20\%$ Overshoot Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","bottom", "Interpreter", "latex", "LineWidth", 1, "FontSize", 14);

%Plot Response Line
plot(tOut, yCLaug(:,2), 'LineWidth', 2);

%Plot Styling
grid on;
title('Angle Deviation vs Time', "Interpreter","Latex", "FontSize", plotTitleFontSize);
ylabel('$\delta \theta$ (radians)', "Interpreter","Latex", "FontSize", axisFontSize);
xlabel('Time (s)', "Interpreter","Latex", "FontSize", axisFontSize);
xlim([tOut(1), tOut(end)]);
ylim([angularStep *-0.30, 1.1*abs(angularStep)]);

%% Produce Output Force Figure
figure('Color','w');
tiles = tiledlayout('flow');

%Overall title for the tiled layout
axisFontSize = 14;
plotTitleFontSize = 16;
figureTitleFontSize = 22;
title(tiles, 'Close-Loop Thrust Acceleration Response Over Two Orbits', "Interpreter", "Latex", "FontSize", figureTitleFontSize);

%---RADIAL FORCING ACCELERATION VERSUS TIME---
nexttile;
patch([0, tOut(end), tOut(end), 0], [-0.01 -0.01 0.01 0.01], plotColors(5, :), 'FaceAlpha', 0.1, 'EdgeColor', 'none'); 
hold on;
plot(tOut, uCLaug_g(1,:), 'LineWidth', 2.0); hold on
yline(0.01);
yline(-0.01);
ylim([-0.02 0.02]);

%Plot Styling and Labels
grid on;
title('Radial Thrust-Generated Acceleration vs Time', "Interpreter","Latex", "FontSize", plotTitleFontSize);
ylabel('Radial Thrust $\delta u_{1}$ (g)', "Interpreter","Latex", "FontSize", axisFontSize);
xlabel('Time (s)', "Interpreter","Latex", "FontSize", axisFontSize);
xlim([tOut(1), tOut(end)]);

%---IN-TRACK FORCING ACCLERATION VERSUS TIME---
nexttile;
patch([0, tOut(end), tOut(end), 0], [-0.01 -0.01 0.01 0.01], plotColors(5, :), 'FaceAlpha', 0.1, 'EdgeColor', 'none'); 
hold on;
plot(tOut, uCLaug_g(2,:), 'LineWidth', 2.0); hold on
yline(0.01);
yline(-0.01);
ylim([-0.02 0.02]);

%Plot Styling and Labels
grid on;
title('In-Track Thrust-Generated Acceleration vs Time', "Interpreter","Latex", "FontSize", plotTitleFontSize);
ylabel('In-Track Thrust $\delta u_{2}$ (g)', "Interpreter","Latex", "FontSize", axisFontSize);
xlabel('Time (s)', "Interpreter","Latex", "FontSize", axisFontSize);
xlim([tOut(1), tOut(end)]);

%---TOTAL FORCING ACCELERATION VERSUS TIME---
nexttile(tiles, "south", [1, 2]);
u_mag_g = sqrt(uCLaug_g(1,:).^2 + uCLaug_g(2,:).^2); %Forcing magnitude in g

%Plot Bounding Box
plotColors = orderedcolors("gem");
patch([0, tOut(end), tOut(end), 0], [0 0 0.01 0.01], plotColors(5, :), 'FaceAlpha', 0.1, 'EdgeColor', 'none'); 
hold on;

%Plot Line
plot(t, u_mag_g, 'LineWidth', 2.0);

%Add in Boundaries
yline(0.01, '-', "Maximum Acceleration = 0.01G", "Interpreter", "Latex", 'LabelHorizontalAlignment','left', 'LabelVerticalAlignment','top');
yline(0, '-');

%Plot Styling
grid on;
title('Total Thrust Acceleration Magnitude vs Time', "Interpreter","Latex", "FontSize", plotTitleFontSize);
ylabel('Total Thrust $\|\mathbf{u}\|$ (g)', "Interpreter","Latex", "FontSize", axisFontSize);
xlabel('Time (s)', "Interpreter","Latex", "FontSize", axisFontSize);
ylim([-0.001, 0.02]);
xlim([tOut(1), tOut(end)]);

%% Observer Performance Plots
figure('Color','w');
tiles = tiledlayout('flow', 'TileSpacing','compact','Padding','compact');
axisFontSize = 16;
plotTitleFontSize = 18;
figureTitleFontSize = 22;
timeEnd = T_orbit * 2;

%Set Figure Window Title
title(tiles, 'Observer Convergence from Non-zero Initial Error', "Interpreter","Latex","FontSize",figureTitleFontSize);
plotColors = orderedcolors("gem");

%---Delta R Plot---
nexttile;

%Plot the True State
plot(t, x_true_case2(:, 1), '-', 'Color', plotColors(1, :), 'LineWidth', 2.0); hold on;

%Plot Estimated Case
plot(t, x_hat_case2(:, 1), '-', 'Color', plotColors(2, :), 'LineWidth', 2.0);

%Plot Styling
ylabel("$\delta r$ (km)", "Interpreter","Latex","FontSize",axisFontSize);
xlabel('Time (sec)', "Interpreter","Latex","FontSize",axisFontSize);
title('Radial Deviation State', "Interpreter","Latex", "FontSize",plotTitleFontSize);
box on; grid on; axis padded;
xlim([0, timeEnd]);

%---Delta R_dot Plot---
nexttile;

%Plot the True State
plot(t, x_true_case2(:, 2), '-', 'Color', plotColors(1, :), 'LineWidth', 2.0); hold on;

%Plot Estimated Case
plot(t, x_hat_case2(:, 2), '-', 'Color', plotColors(2, :), 'LineWidth', 2.0);

%Plot Styling
ylabel("$\delta \dot r$ (km/s)", "Interpreter","Latex","FontSize",axisFontSize);
xlabel('Time (sec)', "Interpreter","Latex","FontSize",axisFontSize);
title('Radial Velocity State', "Interpreter","Latex", "FontSize",plotTitleFontSize);
box on; grid on; axis padded;
xlim([0, timeEnd]);

%---Delta Theta Plot---
nexttile;

%Plot the True State
plot(t, x_true_case2(:, 3), '-', 'Color', plotColors(1, :), 'LineWidth', 2.0); hold on;

%Plot Estimated Case
plot(t, x_hat_case2(:, 3), '-', 'Color', plotColors(2, :), 'LineWidth', 2.0);

%Plot Styling
ylabel("$\delta \theta$ (rad)", "Interpreter","Latex","FontSize",axisFontSize);
xlabel('Time (sec)', "Interpreter","Latex","FontSize",axisFontSize);
title('Angular Position State', "Interpreter","Latex", "FontSize",plotTitleFontSize);
box on; grid on; axis padded;
xlim([0, timeEnd]);

%---Delta Theta_dot Plot---
nexttile;

%Plot the True State
plot(t, x_true_case2(:, 4), '-', 'Color', plotColors(1, :), 'LineWidth', 2.0); hold on;

%Plot Estimated Case
plot(t, x_hat_case2(:, 4), '-', 'Color', plotColors(2, :), 'LineWidth', 2.0);

%Plot Styling
ylabel("$\delta \dot \theta$ (rad/s)", "Interpreter","Latex","FontSize",axisFontSize);
xlabel('Time (sec)', "Interpreter","Latex","FontSize",axisFontSize);
title('Angular Velocity State', "Interpreter","Latex", "FontSize",plotTitleFontSize);
box on; grid on; axis padded;
xlim([0, timeEnd]);
legend(["True State","Estimated State"],"Interpreter","Latex",'Location','best', "FontSize", axisFontSize);

%% Observer Performance Plots (Zero Initial Error)
figure('Color','w');
tiles = tiledlayout('flow', 'TileSpacing','compact','Padding','compact');
axisFontSize  = 16;
plotTitleFontSize  = 18;
figureTitleFontSize = 22;
timeEnd = T_orbit * 2;

title(tiles, 'Observer Convergence from Zero Initial Error', "Interpreter","Latex","FontSize",figureTitleFontSize);
plotColors = orderedcolors("gem");

%---Delta R Plot---
nexttile;
plot(t, x_true_case1(:, 1), '-', 'Color', plotColors(1, :), 'LineWidth', 2.0); hold on;
plot(t, x_hat_case1(:, 1),  '-', 'Color', plotColors(2, :), 'LineWidth', 2.0);
ylabel("$\delta r$ (km)", "Interpreter","Latex","FontSize",axisFontSize);
xlabel('Time (sec)',   "Interpreter","Latex","FontSize",axisFontSize);
title('Radial Deviation State', "Interpreter","Latex","FontSize",plotTitleFontSize);
box on; grid on; axis padded;
xlim([0, timeEnd]);

%---Delta r_dot Plot---
nexttile;
plot(t, x_true_case1(:, 2), '-', 'Color', plotColors(1, :), 'LineWidth', 2.0); hold on;
plot(t, x_hat_case1(:, 2),  '-', 'Color', plotColors(2, :), 'LineWidth', 2.0);
ylabel("$\delta \dot r$ (km/s)", "Interpreter","Latex","FontSize",axisFontSize);
xlabel('Time (sec)',        "Interpreter","Latex","FontSize",axisFontSize);
title('Radial Velocity State', "Interpreter","Latex","FontSize",plotTitleFontSize);
box on; grid on; axis padded;
xlim([0, timeEnd]);

%---Delta theta Plot---
nexttile;
plot(t, x_true_case1(:, 3), '-', 'Color', plotColors(1, :), 'LineWidth', 2.0); hold on;
plot(t, x_hat_case1(:, 3),  '-', 'Color', plotColors(2, :), 'LineWidth', 2.0);
ylabel("$\delta \theta$ (rad)", "Interpreter","Latex","FontSize",axisFontSize);
xlabel('Time (sec)',        "Interpreter","Latex","FontSize",axisFontSize);
title('Angular Position State', "Interpreter","Latex","FontSize",plotTitleFontSize);
box on; grid on; axis padded;
xlim([0, timeEnd]);

%---Delta theta_dot Plot---
nexttile;
plot(t, x_true_case1(:, 4), '-', 'Color', plotColors(1, :), 'LineWidth', 2.0); hold on;
plot(t, x_hat_case1(:, 4),  '-', 'Color', plotColors(2, :), 'LineWidth', 2.0);
ylabel("$\delta \dot \theta$ (rad/s)", "Interpreter","Latex","FontSize",axisFontSize);
xlabel('Time (sec)',            "Interpreter","Latex","FontSize",axisFontSize);
title('Angular Velocity State', "Interpreter","Latex","FontSize",plotTitleFontSize);
box on; grid on; axis padded;
xlim([0, timeEnd]);
legend(["True State","Estimated State"], "Interpreter","Latex",'Location','best', "FontSize", axisFontSize);


%% Estimation Error Norms Plots
e_norm_case1 = vecnorm(est_err_case1,2,2); %Case 1 Total Error Magnitude
e_norm_case2 = vecnorm(est_err_case2,2,2); %Case 2 Total Error Magnitude

figure('Color','w');

axisFontSize = 16;
plotTitleFontSize = 18;
figureTitleFontSize = 22;
title("State Estimation Error Norms", "Interpreter","latex", "FontSize", figureTitleFontSize);

%Add Error Lines
plot(t, e_norm_case1, "-", "Color", plotColors(1,:), "LineWidth", 2.0); hold on;
plot(t, e_norm_case2, "-", "Color", plotColors(2,:), "LineWidth", 2.0);

%Plot Styling
xlabel("Time (seconds)", "Interpreter","Latex","FontSize",axisFontSize);
ylabel("$\|e(t)\|_2$", "Interpreter","Latex","FontSize",axisFontSize);
title("Observer Error Norm (Zero vs Non-zero Initial Error)", "Interpreter","Latex","FontSize",plotTitleFontSize);
legend(["Zero Initial Error", "50\%Initial Error"], "Interpreter","Latex",'Location','NorthEast');
box on; grid on;

%% Manual vs LQR Closed-Loop Response (Radial & Angular Positions) Plots
figure('Color','w');
tiles = tiledlayout('flow');

axisFontSize = 16;
plotTitleFontSize = 18;
figureTitleFontSize = 22;
title(tiles, 'Manual vs LQR Closed-Loop Response', "Interpreter","Latex","FontSize",figureTitleFontSize);

plotColors = orderedcolors("gem");

%---------------- RADIAL DEVIATION ----------------%
nexttile;

radialStep = delta_r0;

%Settling / overshoot bands
patch([0, T_orbit, T_orbit, 0], [0, 0, radialStep * 0.05, radialStep * 0.05], plotColors(3, :), "FaceAlpha", 0.2, "EdgeColor", "none");
hold on; box on;
patch([0, t(end), t(end), T_orbit, T_orbit, 0], [-radialStep * 0.2, -radialStep * 0.2, radialStep * 0.2, radialStep * 0.2, 0, 0], plotColors(4, :), "FaceAlpha", 0.2, "EdgeColor", "none");

yline(0, '--', 'LineWidth', 1.5, "FontSize", 14);
xline(T_orbit, '-', "1x Orbital Period", "LabelHorizontalAlignment","center", "LabelVerticalAlignment","top", "Interpreter","latex", "LineWidth",1, "FontSize",14);
yline(radialStep * 0.05, '-', "$5\%$ Settling Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","top", "Interpreter","latex", "LineWidth",1, "FontSize",14);
yline(radialStep * 0.20, '-', "$20\%$ Undershoot Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","top", "Interpreter","latex", "LineWidth",1, "FontSize",14);
yline(-radialStep * 0.20, '-', "$20\%$ Overshoot Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","bottom", "Interpreter","latex", "LineWidth",1, "FontSize",14);

%Plot Response Lines
plot(t, yCLaug(:,1), 'LineWidth', 2.0, 'Color', plotColors(1,:)); hold on;
plot(t, x_true_lqr(:,1), '-', 'LineWidth', 2.0, 'Color', plotColors(2,:));

%Plot Styling
grid on;
title('Radial Deviation vs Time', "Interpreter","Latex", "FontSize",plotTitleFontSize);
ylabel('$\delta r$ (km)', "Interpreter","Latex", "FontSize",axisFontSize);
xlabel('Time (s)', "Interpreter","Latex", "FontSize",axisFontSize);
xlim([t(1), t(end)]);
ylim([radialStep * -0.30, 1.1 * abs(radialStep)]);

%---------------- ANGULAR POSITION ----------------%
nexttile;

angularStep = delta_theta0;

%Settling / overshoot bands (same as previous plots)
patch([0, T_orbit, T_orbit, 0], [0, 0, angularStep * 0.05, angularStep * 0.05], plotColors(3, :), "FaceAlpha", 0.2, "EdgeColor", "none");
hold on; box on;
patch([0, t(end), t(end), T_orbit, T_orbit, 0], [-angularStep * 0.2, -angularStep * 0.2, angularStep * 0.2, angularStep * 0.2, 0, 0], plotColors(4, :), "FaceAlpha", 0.2, "EdgeColor", "none");

yline(0, '--', 'LineWidth', 1.5, "FontSize",14);
xline(T_orbit, '-', "1x Orbital Period", "LabelHorizontalAlignment","center", "LabelVerticalAlignment","top", "Interpreter","latex", "LineWidth",1, "FontSize",14);
yline(angularStep * 0.05, '-', "$5\%$ Settling Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","top", "Interpreter","latex", "LineWidth",1, "FontSize",14);
yline(angularStep * 0.20, '-', "$20\%$ Undershoot Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","top", "Interpreter","latex", "LineWidth",1, "FontSize",14);
yline(-angularStep * 0.20, '-', "$20\%$ Overshoot Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","bottom", "Interpreter","latex", "LineWidth",1, "FontSize",14);

%Plot Response Lines
h(1) = plot(t, yCLaug(:,2), 'LineWidth', 2.0, 'Color', plotColors(1,:)); hold on; %Manual Placement
h(2) = plot(t, x_true_lqr(:,3), '-', 'LineWidth', 2.0, 'Color', plotColors(2,:)); %LQR Placement

%PLot Styling
grid on;
title('Angular Position vs Time', "Interpreter","Latex", "FontSize",plotTitleFontSize);
ylabel('$\delta \theta$ (rad)', "Interpreter","Latex", "FontSize",axisFontSize);
xlabel('Time (s)', "Interpreter","Latex", "FontSize",axisFontSize);
xlim([t(1), t(end)]);
ylim([angularStep * -0.30, 1.1 * abs(angularStep)]);
legend(h, ["Manual (Part 4)", "LQR (Part 6)"], "Interpreter","Latex",'Location','NorthEast', "FontSize", axisFontSize);

%% Manual vs LQR Closed-Loop Acceleration Plots
figure('Color','w');
tiles = tiledlayout('flow');
axisFontSize = 14;
plotTitleFontSize = 16;
figureTitleFontSize = 22;

title(tiles, 'Manual vs LQR Thrust Acceleration Response', "Interpreter","Latex", "FontSize", figureTitleFontSize);
plotColors = orderedcolors("gem");

%---------------- RADIAL THRUST ACCELERATION ----------------%
nexttile;

%Shaded allowable band +/- 0.01 g
patch([0, t(end), t(end), 0], [-0.01, -0.01, 0.01, 0.01], plotColors(5, :), 'FaceAlpha', 0.1, 'EdgeColor', 'none'); 
hold on;

%Plot Lines
plot(t, u_manual_g(:,1), 'LineWidth', 2.0, 'Color', plotColors(1,:)); %Manual Pole Placement Response
plot(t, u_lqr_g(:,1),    'LineWidth', 2.0, 'Color', plotColors(2,:)); %LQR Pole PLacement Response

%Add in Bounding Lines / Limits
yline(0.01, '-', "0.01 g Limit", "Interpreter","Latex", "LabelHorizontalAlignment","left", "LabelVerticalAlignment","top");
yline(-0.01, '-', "Interpreter","Latex");
yline(0, '-', "LineWidth", 1.0);

%Plot Styling
ylim([-0.02, 0.02]);
xlim([t(1), t(end)]);
grid on; box on;
title('Radial Thrust Acceleration', "Interpreter","Latex", "FontSize", plotTitleFontSize);
ylabel('$\delta u_{1}$ (g)', "Interpreter","Latex", "FontSize", axisFontSize);
xlabel('Time (s)', "Interpreter","Latex", "FontSize", axisFontSize);

%---------------- IN-TRACK THRUST ACCELERATION ----------------%
nexttile;

%Shaded allowable band +/- 0.01 g
patch([0, t(end), t(end), 0], [-0.01, -0.01, 0.01, 0.01], plotColors(5, :), 'FaceAlpha', 0.1, 'EdgeColor', 'none'); 
hold on;

%Plot Lines
h(1) = plot(t, u_manual_g(:,2), 'LineWidth', 2.0, 'Color', plotColors(1,:));
h(2) = plot(t, u_lqr_g(:,2),    'LineWidth', 2.0, 'Color', plotColors(2,:));

%Add in Bounding Lines / Limits
yline(0.01, '-', "0.01 g Limit", "Interpreter","Latex", "LabelHorizontalAlignment","left", "LabelVerticalAlignment","top");
yline(-0.01, '-', "Interpreter","Latex");
yline(0, '-', "LineWidth", 1.0);

%Plot Styling
ylim([-0.02, 0.02]);
xlim([t(1), t(end)]);
grid on; box on;
title('In-Track Thrust Acceleration', "Interpreter","Latex", "FontSize", plotTitleFontSize);
ylabel('$\delta u_{2}$ (g)', "Interpreter","Latex", "FontSize", axisFontSize);
xlabel('Time (s)', "Interpreter","Latex", "FontSize", axisFontSize);
legend(h, ["Manual (Part 4)", "LQR (Part 6)"], "Interpreter","Latex", 'Location','SouthEast', "FontSize", axisFontSize);

%---------------- TOTAL THRUST ACCELERATION MAGNITUDE ----------------%
nexttile(tiles, "south", [1, 2]);

%Calculate Total Forcing Mangitude
u_mag_manual_g = sqrt(u_manual_g(:,1).^2 + u_manual_g(:,2).^2);
u_mag_lqr_g    = sqrt(u_lqr_g(:,1).^2    + u_lqr_g(:,2).^2);

%Shaded allowable band +/- 0.01 g
patch([0, t(end), t(end), 0], [0, 0, 0.01, 0.01], plotColors(5, :), 'FaceAlpha', 0.1, 'EdgeColor', 'none'); 
hold on;

%Plot Lines
plot(t, u_mag_manual_g, 'LineWidth', 2.0, 'Color', plotColors(1,:));
plot(t, u_mag_lqr_g,    'LineWidth', 2.0, 'Color', plotColors(2,:));

%Add in Bounding Lines / Limits
yline(0.01, '-', "0.01 g Limit", "Interpreter","Latex", "LabelHorizontalAlignment","left", "LabelVerticalAlignment","top");
yline(0, '-', "LineWidth", 1.0);

%Plot Styling
ylim([-0.001, 0.02]);
xlim([t(1), t(end)]);
grid on; box on;
title('Total Thrust Acceleration Magnitude', "Interpreter","Latex", "FontSize", plotTitleFontSize);
ylabel('$\|\mathbf{u}\|$ (g)', "Interpreter","Latex", "FontSize", axisFontSize);
xlabel('Time (s)', "Interpreter","Latex", "FontSize", axisFontSize);

%% Manual Controller: Full-State vs Manual+Observer Closed-Loop Response
figure('Color','w');
tiles = tiledlayout('flow');
axisFontSize = 16;
plotTitleFontSize = 18;
figureTitleFontSize = 22;
title(tiles,'Manual Full-State vs Manual+Observer Closed-Loop Response', "Interpreter","Latex","FontSize",figureTitleFontSize);
plotColors = orderedcolors("gem");

%---RADIAL DEVIATION---%
nexttile;
radialStep = delta_r0;

%Add in shaded allowance band
patch([0,T_orbit,T_orbit,0],[0,0,radialStep*0.05,radialStep*0.05], plotColors(3,:),"FaceAlpha",0.2,"EdgeColor","none");
hold on;
patch([0,t(end),t(end),T_orbit,T_orbit,0], [-radialStep*0.2,-radialStep*0.2,radialStep*0.2,radialStep*0.2,0,0], plotColors(4,:),"FaceAlpha",0.2,"EdgeColor","none");
yline(0,'--','LineWidth',1.5,"FontSize",14);
xline(T_orbit,'-',"1x Orbital Period", "LabelHorizontalAlignment","center", "LabelVerticalAlignment","top", "Interpreter","latex","LineWidth",1,"FontSize",14);
yline(radialStep*0.05,'-',"$5\%$ Settling Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","top", "Interpreter","latex","LineWidth",1,"FontSize",14);
yline(radialStep*0.20,'-',"$20\%$ Undershoot Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","top", "Interpreter","latex","LineWidth",1,"FontSize",14);
yline(-radialStep*0.20,'-',"$20\%$ Overshoot Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","bottom", "Interpreter","latex","LineWidth",1,"FontSize",14);

%Plot Lines
plot(t,yCLaug(:,1),'LineWidth',2.0,"Color",plotColors(1,:));
plot(t,x_true_case1(:,1),'LineWidth',2.0,"Color",plotColors(2,:));

%Plot Styling
grid on; box on;
title('Radial Deviation vs Time',"Interpreter","Latex","FontSize",plotTitleFontSize);
ylabel('$\delta r$ (km)',"Interpreter","Latex","FontSize",axisFontSize);
xlabel('Time (s)',"Interpreter","Latex","FontSize",axisFontSize);
xlim([t(1),t(end)]);
ylim([radialStep*-0.30,1.1*abs(radialStep)]);

%---ANGULAR POSITION---%
nexttile;
angularStep = delta_theta0;

%Add in shaded allowance band
patch([0,T_orbit,T_orbit,0],[0,0,angularStep*0.05,angularStep*0.05], plotColors(3,:),"FaceAlpha",0.2,"EdgeColor","none");
hold on;
patch([0,t(end),t(end),T_orbit,T_orbit,0], [-angularStep*0.2,-angularStep*0.2,angularStep*0.2,angularStep*0.2,0,0], plotColors(4,:),"FaceAlpha",0.2,"EdgeColor","none");
yline(0,'--','LineWidth',1.5,"FontSize",14);
xline(T_orbit,'-',"1x Orbital Period", "LabelHorizontalAlignment","center", "LabelVerticalAlignment","top", "Interpreter","latex","LineWidth",1,"FontSize",14);
yline(angularStep*0.05,'-',"$5\%$ Settling Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","top", "Interpreter","latex","LineWidth",1,"FontSize",14);
yline(angularStep*0.20,'-',"$20\%$ Undershoot Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","top", "Interpreter","latex","LineWidth",1,"FontSize",14);
yline(-angularStep*0.20,'-',"$20\%$ Overshoot Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","bottom", "Interpreter","latex","LineWidth",1,"FontSize",14);

%Plot Lines
h(1) = plot(t,yCLaug(:,2),'LineWidth',2.0,"Color",plotColors(1,:));
h(2) = plot(t,x_true_case1(:,3),'LineWidth',2.0,"Color",plotColors(2,:));

%Plot Styling
grid on; box on;
title('Angular Position vs Time',"Interpreter","Latex","FontSize",plotTitleFontSize);
ylabel('$\delta \theta$ (rad)',"Interpreter","Latex","FontSize",axisFontSize);
xlabel('Time (s)',"Interpreter","Latex","FontSize",axisFontSize);
xlim([t(1),t(end)]);
ylim([angularStep*-0.30,1.1*abs(angularStep)]);
legend(h,["Manual (Full-State)","Manual+Observer (Zero Error)"], "Interpreter","Latex","Location","NorthEast","FontSize",axisFontSize);


%% LQR Closed-Loop Response with Observer Error (0% vs 50% Initial Error)
figure('Color','w');
tiles = tiledlayout('flow');
axisFontSize = 16;
plotTitleFontSize = 18;
figureTitleFontSize = 22;
title(tiles,'LQR Closed-Loop Response with Observer Error', "Interpreter","Latex","FontSize",figureTitleFontSize);
plotColors = orderedcolors("gem");

%---RADIAL DEVIATION---%
nexttile;
radialStep = delta_r0;

%Add in Allowable Tolerance
patch([0,T_orbit,T_orbit,0],[0,0,radialStep*0.05,radialStep*0.05], plotColors(3,:),"FaceAlpha",0.2,"EdgeColor","none");
hold on; 
patch([0,t(end),t(end),T_orbit,T_orbit,0], [-radialStep*0.2,-radialStep*0.2,radialStep*0.2,radialStep*0.2,0,0], plotColors(4,:),"FaceAlpha",0.2,"EdgeColor","none");
yline(0,'--','LineWidth',1.5,"FontSize",14);
xline(T_orbit,'-',"1x Orbital Period", "LabelHorizontalAlignment","center", "LabelVerticalAlignment","top", "Interpreter","latex","LineWidth",1,"FontSize",14);
yline(radialStep*0.05,'-',"$5\%$ Settling Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","top", "Interpreter","latex","LineWidth",1,"FontSize",14);
yline(radialStep*0.20,'-',"$20\%$ Undershoot Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","top", "Interpreter","latex","LineWidth",1,"FontSize",14);
yline(-radialStep*0.20,'-',"$20\%$ Overshoot Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","bottom", "Interpreter","latex","LineWidth",1,"FontSize",14);

%Plot Lines
plot(t,x_true_lqr(:,1),'LineWidth',2.0,"Color",plotColors(1,:));
plot(t,x_true_lqr_err(:,1),'LineWidth',2.0,"Color",plotColors(2,:));

%Plot Styling
grid on; box on;
title('Radial Deviation vs Time (LQR)',"Interpreter","Latex","FontSize",plotTitleFontSize);
ylabel('$\delta r$ (km)',"Interpreter","Latex","FontSize",axisFontSize);
xlabel('Time (s)',"Interpreter","Latex","FontSize",axisFontSize);
xlim([t(1),t(end)]);
ylim([radialStep*-0.30,1.1*abs(radialStep)]);

%---ANGULAR POSITION---%
nexttile;
angularStep = delta_theta0;

%Add in shaded allowable bands
patch([0,T_orbit,T_orbit,0],[0,0,angularStep*0.05,angularStep*0.05], plotColors(3,:),"FaceAlpha",0.2,"EdgeColor","none");
hold on; 
patch([0,t(end),t(end),T_orbit,T_orbit,0], [-angularStep*0.2,-angularStep*0.2,angularStep*0.2,angularStep*0.2,0,0], plotColors(4,:),"FaceAlpha",0.2,"EdgeColor","none");
yline(0,'--','LineWidth',1.5,"FontSize",14);
xline(T_orbit,'-',"1x Orbital Period", "LabelHorizontalAlignment","center", "LabelVerticalAlignment","top", "Interpreter","latex","LineWidth",1,"FontSize",14);
yline(angularStep*0.05,'-',"$5\%$ Settling Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","top", "Interpreter","latex","LineWidth",1,"FontSize",14);
yline(angularStep*0.20,'-',"$20\%$ Undershoot Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","top", "Interpreter","latex","LineWidth",1,"FontSize",14);
yline(-angularStep*0.20,'-',"$20\%$ Overshoot Limit", "LabelHorizontalAlignment","right", "LabelVerticalAlignment","bottom", "Interpreter","latex","LineWidth",1,"FontSize",14);
h(1) = plot(t,x_true_lqr(:,3),'LineWidth',2.0,"Color",plotColors(1,:));
h(2) = plot(t,x_true_lqr_err(:,3),'LineWidth',2.0,"Color",plotColors(2,:));

%Plot Styling
grid on; box on;
title('Angular Position vs Time (LQR)',"Interpreter","Latex","FontSize",plotTitleFontSize);
ylabel('$\delta \theta$ (rad)',"Interpreter","Latex","FontSize",axisFontSize);
xlabel('Time (s)',"Interpreter","Latex","FontSize",axisFontSize);
xlim([t(1),t(end)]);
ylim([angularStep*-0.30,1.1*abs(angularStep)]);
legend(h,["LQR (0\% Initial Error)","LQR (50\% Initial Error)"], "Interpreter","Latex","Location","NorthEast","FontSize",axisFontSize);
