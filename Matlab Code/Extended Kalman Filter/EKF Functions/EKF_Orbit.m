function [xhat_hist, P_hist, innov_cell, S_cell, visible_ids_cell, yhat_cell] = EKF_Orbit(ydata, tvec, x0_hat, P0, Q_matrix, R_matrix, mu, R_E, omega_E)
%Runs an Extended Kalman Filter for a single measurement sequence
%
%Inputs
%  ydata    - 1xN cell array of measurements. ydata{i} = m_i x 1 measurement 
%             associated with the measurement at time tvec(i). m_i = 3 * (# visible stations)
%  tvec     - 1xN double time vector associated with the time of each measurement [seconds]
%  x0_hat   - 4x1 initial state estimate
%  P0       - 4x4 initial covariance
%  Qtrue    - 2x2 process noise covariance matrix applied to acceleration disturbances
%  Rtrue    - 3x3 measurement noise covariance matrix for each station
%  mu       - Earth gravitational parameter [km^3/s^2]
%  R_E      - Earth radius [km]
%  omega_E  - Earth rotation rate [rad/s]
%
%Outputs
% xhat_hist        - 4xN double matrix of posterior state estimates
% P_hist           - 4x4xN double matrix of posterior covariance matrices
% innov_cell       - 1xN cell array of innovation vectors v_k = y_k - yhat_k
% S_cell           - 1xN cell array of innovation covariance matrices S_k
% visible_ids_cell - 1xN cell array of visible station indices at each time step
% yhat_cell        - 1xN cell array of predicted measurement vectors h(xhat_minus)

%Get measurement length
N = length(tvec);
numStates = 4;

%Preallocate output variables
xhat_hist = zeros(numStates,N);
P_hist = zeros(numStates,numStates,N);
innov_cell = cell(1,N);
S_cell = cell(1,N);
visible_ids_cell = cell(1,N);
yhat_cell = cell(1,N);

%Initialize state estimate and covariance for time tvec(1)
xhat_plus = x0_hat;
P_plus = P0;

%Add to output arrays
xhat_hist(:,1) = xhat_plus;
P_hist(:,:,1) = P_plus;
innov_cell{1} = [];
S_cell{1} = [];
visible_ids_cell{1} = [];
yhat_cell{1} = [];

%Define Gamma (constant)
gamma = [0 0;
         1 0;
         0 0;
         0 1];

%Assume input zero at all times
u_k = [0; 0];

%Extended Kalman Filter loop
for k = 1:(N-1)
    %Time step
    dT = tvec(k+1) - tvec(k);
   
    % Now propagate using nonlinear dynamics
    xhat_minus = propagate_orbit(xhat_plus, u_k, dT, mu);

    %Time update / prediction
    A_tilde_k = A_jacobian(xhat_plus, mu);
    F_tilde_k = eye(numStates) + dT*A_tilde_k;
    Omega_tilde_k = dT * gamma;

    P_minus = F_tilde_k*P_plus*F_tilde_k' + Omega_tilde_k * Q_matrix * Omega_tilde_k';

    %Measurement data at time t_{k+1}
    measurement = ydata{k+1};

    %If no measurements, prediction only
    if isempty(measurement)
        xhat_plus = xhat_minus;
        P_plus = P_minus;
        xhat_hist(:,k+1) = xhat_plus;
        P_hist(:,:,k+1) = P_plus;
        innov_cell{k+1} = [];
        S_cell{k+1} = [];
        visible_ids_cell{k+1} = [];
        yhat_cell{k+1} = [];
        continue
    end

    %Extract station IDs and number of visible stations
    station_ids = measurement(4,:)';
    n_vis = numel(station_ids);

    %Build measurement vector y_meas by stacking [rho; rho_dot; angle] columns
    y_meas = reshape(measurement(1:3,:),3*n_vis,1);

    %Measurement update / correction step
    t_k1 = tvec(k+1);
    [yhat_minus, H_tilde_k, ~] = H_jacobian(xhat_minus, t_k1, station_ids, R_E, omega_E);

    %Force predicted measurement to column
    yhat_minus = yhat_minus(:);

    %Store predicted measurement and station IDs
    yhat_cell{k+1} = yhat_minus;
    visible_ids_cell{k+1} = station_ids;

    %Form measurement noise covariance for n_vis stations
    R_k = R_matrix;
    for j = 2:n_vis
        R_k = blkdiag(R_k,R_matrix);
    end

    if(k == 21 || k == 22)
        abc = 1;
    end

    %Innovation
    e_tilde_k = y_meas - yhat_minus;

    %Wrap innovations to [-pi, pi]
    for j = 1:n_vis
        angle_idx = 3*j;  % Every 3rd element is an angle
        e_tilde_k(angle_idx) = wrapToPi(e_tilde_k(angle_idx));
    end

    %Innovation covariance
    S_k = H_tilde_k*P_minus*H_tilde_k' + R_k;

    %Kalman gain
    K_tilde_k = P_minus*H_tilde_k'/S_k;

    %State update
    xhat_plus = xhat_minus + K_tilde_k*e_tilde_k;

    %Covariance update
    P_plus = (eye(numStates) - K_tilde_k*H_tilde_k)*P_minus;

    %Store results
    xhat_hist(:,k+1) = xhat_plus;
    P_hist(:,:,k+1) = P_plus;
    innov_cell{k+1} = e_tilde_k;
    S_cell{k+1} = S_k;
end

end