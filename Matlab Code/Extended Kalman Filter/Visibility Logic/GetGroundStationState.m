% i is the station index (1 to 12)
function [X_i, Y_i, X_dot_i, Y_dot_i, theta_i] = GetGroundStationState(i, t)
    % TODO: Check if these should be global
    R_E = 6378; % Earth radius in km
    omega_E = 2*pi/86400; % Earth rotation speed in rad/s
    theta_i0 = (i - 1) * pi/6; % Initial location angle of ground station i
    
    X_i = R_E * cos(omega_E * t + theta_i0);
    Y_i = R_E * sin(omega_E * t + theta_i0);
    X_dot_i = -omega_E * R_E * sin(omega_E * t + theta_i0);
    Y_dot_i = omega_E * R_E * cos(omega_E * t + theta_i0);
    theta_i = atan2(Y_i,X_i);
end