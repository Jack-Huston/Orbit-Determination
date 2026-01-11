function dx = orbit_dynamics(~, x, u, mu)
%Calculates continuous-time spacecraft dynamics derivatives.
%Inputs:
%  ~  - Placeholder for used time variable, kept for ODE45
%  x  - 4x1 state vector [X; Xdot; Y; Ydot] in km and km/s
%  u  - 2x1 control input [u1; u2] (accelerations in km/s^2)
%  mu - gravitational parameter [km^3/s^2]
%
%Output:
%  dx - 4x1 time derivative of the state

%Pull Out States
x1 = x(1); x2 = x(2);
x3 = x(3); x4 = x(4);

%Calculate Derivatives
x2_dot = -mu * x1 / (sqrt(x1^2 + x3^2))^3 + u(1);
x4_dot = -mu * x3 / (sqrt(x1^2 + x3^2))^3 + u(2);

%Assemble Output Derivative
dx = [x2; x2_dot; x4; x4_dot];

end
