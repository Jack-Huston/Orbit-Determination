function [y_i, is_visible] = GetStationMeasurement(x, t, i)
    [X_i, Y_i, X_dot_i, Y_dot_i, theta_i] = GetGroundStationState(i, t);

    X = x(1);
    X_dot = x(2);
    Y = x(3);
    Y_dot = x(4);
    phi_i = atan2((Y - Y_i),(X - X_i));

    phi_min = -pi/2 + theta_i;
    phi_max = pi/2 + theta_i;

    % Check if the LOS angle is within the station's "sky-side" view cone
    is_visible = (phi_i >= phi_min) && (phi_i <= phi_max);
    angle_diff_raw = phi_i - theta_i; % GEN_AI
    true_angle_diff_rad = mod(angle_diff_raw + pi, 2*pi) - pi; % GEN_AI
    angle_rad = abs(true_angle_diff_rad); % GEN_AI
    is_visible = angle_rad < pi/2; % GEN_AI
    rho_i = sqrt((X - X_i)^2 + (Y - Y_i)^2);
    X_diff = X - X_i;
    X_dot_diff = X_dot - X_dot_i;
    Y_diff = Y - Y_i;
    Y_dot_diff = Y_dot - Y_dot_i;
    rho_dot_i = ((X_diff * X_dot_diff) + (Y_diff * Y_dot_diff))/rho_i;
    y_i = [rho_i; rho_dot_i; phi_i];
end