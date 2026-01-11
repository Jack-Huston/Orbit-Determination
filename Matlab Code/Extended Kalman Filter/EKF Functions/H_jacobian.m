function [y, H, visible_ids] = H_jacobian(x, t, visible_ids, R_E, omega_E)
%Compute measurements and Jacobian for all visible stations.
%Inputs:
% x          - 4x1 spacecraft state [X; Xdot; Y; Ydot]
% t          - current time [s]
% visible_ids- List of visible station IDs from actual time step measurements
% R_E        - Earth radius [km]
% omega_E    - Earth rotation rate [rad/s]
%
%Outputs:
% y           - stacked measurement vector for visible stations (3*number_visible x 1)
% H           - stacked measurement Jacobian (3*number_visible x 4)
% visible_ids - station IDs used (returned as a column vector)

%Pull Out Current State
x1 = x(1); x2 = x(2); x3 = x(3); x4 = x(4);

%Ensure visible_ids is a column vector
visible_ids = visible_ids(:);

%Get Number of Measurements
number_visible = numel(visible_ids);

%Compute Current Position of Each Ground Tracking Station
X_i = zeros(1,12);
X_dot_i = zeros(1,12);
Y_i = zeros(1,12);
Y_dot_i = zeros(1,12);

for i = 1:12
    theta_i_0 = (i - 1)*pi/6;
    X_i(i) = R_E*cos(omega_E*t + theta_i_0);
    Y_i(i) = R_E*sin(omega_E*t + theta_i_0);
    X_dot_i(i) = -omega_E*R_E*sin(omega_E*t + theta_i_0);
    Y_dot_i(i) = omega_E*R_E*cos(omega_E*t + theta_i_0);
end

%Preallocate output arrays
H = zeros(3*number_visible,4);
y = zeros(3*number_visible,1);

%Loop through specified visible station IDs
for k = 1:number_visible
    i = visible_ids(k);
    
    %Solve for Estimated Position Using Non-Linear CT Functions
    delta_X = x1 - X_i(i);
    delta_X_dot = x2 - X_dot_i(i);
    delta_Y = x3 - Y_i(i);
    delta_Y_dot = x4 - Y_dot_i(i);
    
    phi_i = atan2(x3 - Y_i(i), x1 - X_i(i));
    rho_i = sqrt(delta_X^2 + delta_Y^2);
    rho_dot_i = (delta_X*delta_X_dot + delta_Y*delta_Y_dot)/rho_i;
    
    y_i = [rho_i; rho_dot_i; phi_i];
    
    %Calculate DT H Matrix Linearized at Estimation Point
    H_i = zeros(3,4);
    a = delta_X*delta_X_dot + delta_Y*delta_Y_dot;

    H_i(1,1) = delta_X/rho_i;
    H_i(1,3) = delta_Y/rho_i;

    H_i(2,1) = (delta_X_dot/rho_i) - (a*delta_X/rho_i^3);
    H_i(2,3) = (delta_Y_dot/rho_i) - (a*delta_Y/rho_i^3);
    H_i(2,2) = delta_X/rho_i;
    H_i(2,4) = delta_Y/rho_i;

    H_i(3,1) = -delta_Y/(rho_i^2);
    H_i(3,3) = delta_X/(rho_i^2);

    %Add to Output Matrix
    idx_start = 3*(k - 1) + 1;
    y(idx_start:idx_start+2) = y_i;
    H(idx_start:idx_start+2,:) = H_i;
end

end
