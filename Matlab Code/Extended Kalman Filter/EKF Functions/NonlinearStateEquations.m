function dx = NonlinearStateEquations(~, x)
    % TODO: Check if we want that globally
    mu = 398600; % Standard gravitational parameter
    r = sqrt(x(1)^2 + x(3)^2);
    x1_dot = x(2);
    x2_dot = -mu*x(1)/r^3;
    x3_dot = x(4);
    x4_dot = -mu*x(3)/r^3;
    dx = [x1_dot; x2_dot; x3_dot; x4_dot];
end