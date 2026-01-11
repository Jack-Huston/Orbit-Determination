%Clear workspace
clear; clc; close all;

%Add in data paths
addpath("EKF Functions\", "Plotting\", "Visibility Logic\");

%Load in Matlab data file
load('orbitdeterm_finalproj_KFdata.mat');

%Set Random Seed
rng(100);

%Define orbital constants
mu      = 398600;        %km^3/s^2 (Earth gravitational parameter)
R_e     = 6378;          %km (Earth radius)
omega_E = (2*pi)/86400;  %rad/s (Earth rotational rate)

dT   = 10;               %Sampling interval (seconds)
K    = 1400;             %Number of time steps
tvec = 0:dT:K*dT;        %Data simulation time vector (length K+1)

%Define nominal initial state
r0     = 6678;                  %300 km altitude orbit
x_nom0 = [r0; 0; 0; r0*sqrt(mu/r0^3)]; %[X_nom, X_dot_nom, Y_nom, Y_dot_nom]

%Define initial desired perturbation
desiredPerturbation = [0.0; 0.0; 0.0; 0.0]; %[delta_X, delta_X_dot, delta_Y, delta_Y_dot]
x_initial = x_nom0 + desiredPerturbation; %[X, X_dot, Y, Y_dot]

%Eulerized DT process-noise covariance
Gamma = [0 0;
         1 0;
         0 0;
         0 1];

%Apply first-order noise
Q_DT = Gamma * Qtrue * Gamma' * dT;

%Apply jitter to process noise to ensure positive-definite matrix
Q_DT = Q_DT + 1e-12 * eye(4);

%Find Cholesky factors for process / measurement noise
S_w = chol(Q_DT,  'lower'); %Process noise (4x4)
S_v = chol(Rtrue, 'lower'); %Single-station measurement noise (3x3)

%Simulate noisy nonlinear "truth" and convert to y-data format
sim_outputs = NonlinearSimulationNoisy(x_initial,0,K*dT,dT,S_w,S_v);
sim_yData   = convertOutput2Ydata(sim_outputs);

%Form yData cell array aligned with tvec (index 1 corresponds to t = 0)
yData      = cell(1,K+1);
yData{1}   = [];               %No measurement at t = 0
yData(2:end) = sim_yData(:);   %Measurements from t = dT to K*dT

%EKF initial conditions and noise guesses
sigma_x0 = 1e-6;    %Position STD in km
sigma_v0 = 1e-5;  %Velocity STD in km/s
x0_hat = x_nom0 + [sigma_x0; sigma_v0; sigma_x0; sigma_v0];
P_0 = diag([sigma_x0, sigma_v0, sigma_x0, sigma_v0]);
Q_KF = 1e-10*eye(2); %Process-noise guess for EKF

%Run Extended Kalman Filter
[xhat_hist,P_hist,innov_cell,S_cell,visible_ids_cell,yhat_cell] = ...
    EKF_Orbit(yData,tvec,x0_hat,P_0,Q_KF,Rtrue,mu,R_e,omega_E);

%Pre-compute true states and 1-sigma standard deviations
N          = numel(tvec);
x_true     = sim_outputs.x.';          %4xN
sigma_hist = zeros(4,N);
for k = 1:N
    sigma_hist(:,k) = sqrt(diag(P_hist(:,:,k)));
end
state_labels = {'X (km)','X\_dot (km/s)','Y (km)','Y\_dot (km/s)'};

%% Plot 1: Ground truth vs EKF estimated state (with ±2σ bounds)
figure('Color','w');
tiles = tiledlayout('flow','TileSpacing','compact','Padding','compact');

figureTitleFontSize = 22;
plotTitleFontSize = 18;
axisFontSize = 16;

title(tiles,'Ground Truth vs EKF Estimated State','FontSize',figureTitleFontSize, 'Interpreter', 'latex');

%X position
nexttile; hold on;
plot(tvec,x_true(1,:),'k','LineWidth',1.2);
plot(tvec,xhat_hist(1,:),'b','LineWidth',1.2);
plot(tvec,xhat_hist(1,:) + 2*sigma_hist(1,:),'r--','LineWidth',1);
plot(tvec,xhat_hist(1,:) - 2*sigma_hist(1,:),'r--','LineWidth',1);
ylabel('$X \, (km)$','Interpreter','latex','FontSize',axisFontSize);
title("$X$ Position", 'Interpreter', 'latex', 'fontSize', plotTitleFontSize);
legend('True','Estimated','+2\sigma','-2\sigma','Location','Northeast','Interpreter','latex');
grid on;

%X velocity
nexttile; hold on;
plot(tvec,x_true(2,:),'k','LineWidth',1.2);
plot(tvec,xhat_hist(2,:),'b','LineWidth',1.2);
plot(tvec,xhat_hist(2,:) + 2*sigma_hist(2,:),'r--','LineWidth',1);
plot(tvec,xhat_hist(2,:) - 2*sigma_hist(2,:),'r--','LineWidth',1);
ylabel('$\dot{X} \, (km/s)$','Interpreter','latex','FontSize',axisFontSize);
title("$X$ Velocity", 'Interpreter', 'latex', 'fontSize', plotTitleFontSize);
legend('True','Estimated','+2\sigma','-2\sigma','Location','Northeast','Interpreter','latex');
grid on;

%Y position
nexttile; hold on;
plot(tvec,x_true(3,:),'k','LineWidth',1.2);
plot(tvec,xhat_hist(3,:),'b','LineWidth',1.2);
plot(tvec,xhat_hist(3,:) + 2*sigma_hist(3,:),'r--','LineWidth',1);
plot(tvec,xhat_hist(3,:) - 2*sigma_hist(3,:),'r--','LineWidth',1);
ylabel('$Y \, (km)$','Interpreter','latex','FontSize',axisFontSize);
title("$Y$ Position", 'Interpreter', 'latex', 'fontSize', plotTitleFontSize);
legend('True','Estimated','+2\sigma','-2\sigma','Location','Northeast','Interpreter','latex');
grid on;

%Y velocity
nexttile; hold on;
plot(tvec,x_true(4,:),'k','LineWidth',1.2);
plot(tvec,xhat_hist(4,:),'b','LineWidth',1.2);
plot(tvec,xhat_hist(4,:) + 2*sigma_hist(4,:),'r--','LineWidth',1);
plot(tvec,xhat_hist(4,:) - 2*sigma_hist(4,:),'r--','LineWidth',1);
ylabel('$\dot{Y} \, (km/s)$','Interpreter','latex','FontSize',axisFontSize);
xlabel(tiles, 'Time (s)', 'Interpreter', 'latex', 'FontSize', axisFontSize);
title("$Y$ Velocity", 'Interpreter', 'latex', 'fontSize', plotTitleFontSize);
legend('True','Estimated','+2\sigma','-2\sigma','Location','Northeast','Interpreter','latex');
grid on;

%Set up for additional comparison plots
n = 4;
Dt_loc = tvec(2) - tvec(1);
T = tvec(end);

%Use times k = 1..K (skip k = 0 to match LKF style)
t_plot = tvec(2:end);
x_true_plot = sim_outputs.x(2:end,:);
xhat_plot = xhat_hist(:,2:end)';

%Get Posterior 1-sigma from P_hist
K_steps = numel(t_plot);
sigma_x_hat_ekf = zeros(K_steps,n);
for k = 1:K_steps
    sigma_x_hat_ekf(k,:) = sqrt(diag(P_hist(:,:,k+1)))';
end

%Estimation error (estimate - truth)
e_x_hat_ekf = xhat_plot - x_true_plot;

%% Plot 2: EKF state estimation error with ±2σ bounds
figure('Color','w');
tiles = tiledlayout(4,1,'TileSpacing','compact','Padding','compact');

figureTitleFontSize = 22;
plotTitleFontSize = 18;
axisFontSize = 16;

title(tiles,'EKF State Estimation Error with $\pm$ $2\sigma$ Bounds', 'interpreter', 'latex', 'fontSize', figureTitleFontSize);

%X position
nexttile; hold on;
plot(t_plot,e_x_hat_ekf(:,1),'b','LineWidth',1.2);
plot(t_plot, 2*sigma_x_hat_ekf(:,1),'r--','LineWidth',1);
plot(t_plot,-2*sigma_x_hat_ekf(:,1),'r--','LineWidth',1);
ylabel('$X \, (km)$','Interpreter','latex','FontSize',axisFontSize);
title("$X$ Position", 'Interpreter', 'latex', 'fontSize', plotTitleFontSize);
legend('Error','+2\sigma','-2\sigma','Location','Northeast','Interpreter','latex');
grid on;

%X velocity
nexttile; hold on;
plot(t_plot,e_x_hat_ekf(:,2),'b','LineWidth',1.2);
plot(t_plot, 2*sigma_x_hat_ekf(:,2),'r--','LineWidth',1);
plot(t_plot,-2*sigma_x_hat_ekf(:,2),'r--','LineWidth',1);
ylabel('$\dot{X} \, (km/s)$','Interpreter','latex','FontSize',axisFontSize);
title("$X$ Velocity", 'Interpreter', 'latex', 'fontSize', plotTitleFontSize);
legend('Error','+2\sigma','-2\sigma','Location','Northeast','Interpreter','latex');
grid on;

%Y position
nexttile; hold on;
plot(t_plot,e_x_hat_ekf(:,3),'b','LineWidth',1.2);
plot(t_plot, 2*sigma_x_hat_ekf(:,3),'r--','LineWidth',1);
plot(t_plot,-2*sigma_x_hat_ekf(:,3),'r--','LineWidth',1);
ylabel('$Y \, (km)$','Interpreter','latex','FontSize',axisFontSize);
title("$Y$ Position", 'Interpreter', 'latex', 'fontSize', plotTitleFontSize);
legend('Error','+2\sigma','-2\sigma','Location','Northeast','Interpreter','latex');
grid on;

%Y velocity
nexttile; hold on;
plot(t_plot,e_x_hat_ekf(:,4),'b','LineWidth',1.2);
plot(t_plot, 2*sigma_x_hat_ekf(:,4),'r--','LineWidth',1);
plot(t_plot,-2*sigma_x_hat_ekf(:,4),'r--','LineWidth',1);
ylabel('$\dot{Y} \, (km/s)$','Interpreter','latex','FontSize',axisFontSize);
xlabel(tiles, 'Time (s)', 'Interpreter', 'latex', 'FontSize', axisFontSize);
title("$Y$ Velocity", 'Interpreter', 'latex', 'fontSize', plotTitleFontSize);
legend('Error','+2\sigma','-2\sigma','Location','Northeast','Interpreter','latex');
grid on;


%% Plot 3: Nonlinear simulated measurements vs time (EKF case)
figure('Color','w');
tiles = tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

figureTitleFontSize = 22;
plotTitleFontSize = 18;
axisFontSize = 16;

title(tiles,'Nonlinear Simulated Measurements vs Time (EKF)', 'Interpreter', 'latex', 'fontSize', figureTitleFontSize);

t_meas    = [];
rho_meas  = [];
rhod_meas = [];
phi_meas  = [];

for k = 1:numel(yData)
    yk = yData{k};
    if isempty(yk)
        continue;
    end
    n_vis = size(yk,2);
    t_meas    = [t_meas,    repmat(tvec(k),1,n_vis)];
    rho_meas  = [rho_meas,  yk(1,:)];
    rhod_meas = [rhod_meas, yk(2,:)];
    phi_meas  = [phi_meas,  yk(3,:)];
end

%Plot H_matrix Y Output values
range_vec = [];
range_rate_vec = [];
angle_vec = [];
time_vec = [];
times = [0 t_plot];

%For each measurement, pull out estimator guess of estimated state
for i = 1:numel(yhat_cell)
    %Pull out iteration state
    iterationValue = yhat_cell{i};

    %Initialize Empty Holder
    range = []; range_rate = []; angle = [];

    %If a value exists, then assign
    if(~isempty(iterationValue))
        iterationValue = reshape(iterationValue, 3, []);
        
        %Pull out value from iteration value
        range = iterationValue(1, :);
        range_rate = iterationValue(2, :);
        angle = iterationValue(3, :);
    end

    %Append to vector
    range_vec = [range_vec range(:)'];
    range_rate_vec = [range_rate_vec range_rate(:)'];
    angle_vec = [angle_vec angle(:)'];
    time_vec = [time_vec times(i) * ones(1, numel(range))];
end

%Plot Results
nexttile; hold on;
scatter(t_meas,rho_meas,10,'filled', 'MarkerFaceColor', [0, 0, 1]);
scatter(time_vec, range_vec, 10, 'filled', 'MarkerFaceColor',[1, 0, 0]);
ylabel('$\rho$ (km)', 'interpreter', 'latex', 'fontSize', axisFontSize);
title("Range ($\rho$) Values", 'Interpreter', 'latex', 'fontSize', plotTitleFontSize);
legend('Measurement Value', 'Estimator Guess','Location','Northeast','Interpreter','latex');
grid on;

nexttile; hold on;
scatter(t_meas,rhod_meas,10,'filled', 'MarkerFaceColor', [0, 0, 1]);
scatter(time_vec, range_rate_vec, 10, 'filled', 'MarkerFaceColor',[1, 0, 0]);
title("Range Rate ($\dot{\rho}$) Values", 'Interpreter', 'latex', 'fontSize', plotTitleFontSize);
ylabel('$\dot{\rho}$ (km/s)', 'interpreter', 'latex', 'fontSize', axisFontSize);
legend('Measurement Value', 'Estimator Guess','Location','Northeast','Interpreter','latex');
grid on;

nexttile; hold on;
scatter(t_meas,phi_meas,10,'filled', 'MarkerFaceColor', [0, 0, 1]);
scatter(time_vec, angle_vec, 10, 'filled', 'MarkerFaceColor',[1, 0, 0]);
title("Angle ($\phi$) Values", 'Interpreter', 'latex', 'fontSize', plotTitleFontSize);
ylabel('$\phi$ (rad)', 'interpreter', 'latex', 'fontSize', axisFontSize);
xlabel('Time (s)', 'interpreter', 'latex', 'fontSize', axisFontSize);
legend('Measurement Value', 'Estimator Guess','Location','Northeast','Interpreter','latex');
grid on;

%% Plot 4: True vs EKF estimated trajectory in Cartesian coordinates
figure('Color','w');
hold on;
plotSimulationStatesCartesian(t_plot,x_true_plot);
plotSimulationStatesCartesian(t_plot,xhat_plot);
title('EKF: True vs Estimated States (Cartesian)', 'Interpreter', 'latex', 'fontSize', 22);
xlabel("X Position (km)", 'Interpreter', 'latex', 'fontSize', 16);
ylabel("Y Position (km)", 'Interpreter', 'latex', 'fontSize', 16);
grid on;
axis equal padded;

% Add a filled circle representing the Earth
R_e = 6371; % Radius of Earth in kilometers
theta = linspace(0, 2*pi, 100); % Parameter for circle
x_circle = R_e * cos(theta); % X coordinates
y_circle = R_e * sin(theta); % Y coordinates
fill(x_circle, y_circle, 'g', 'FaceAlpha', 0.5); % Draw filled circle

legend(["True Position", "Estimated Position", "Earth"], 'Interpreter', 'latex');
