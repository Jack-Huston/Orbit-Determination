function x_next = propagate_orbit(x_curr, u_curr, dt, mu)
%Propagates the nonlinear dynamics one time step at a time (dt)
%Inputs:
%  x_curr - 4x1 current state at t_k
%  u_curr - 2x1 control input assumed ZOH (constant over [t_k, t_k+1])
%  dt     - time step [seconds]
%  mu     - gravitational parameter [km^3/s^2]
%
%Output:
%  x_next - 4x1 state at t_k+1 after integrating dynamics

% Anonymous handle for ODE45 (U_curr and Mu are held constant)
dynfun = @(t, x) orbit_dynamics(t, x, u_curr, mu);

% Integrate from t=0 to t=dt
tspan = [0, dt];
opts  = odeset('RelTol',1e-9,'AbsTol',1e-9);
[~, x_traj] = ode45(dynfun, tspan, x_curr, opts);

% Take final state
x_next = x_traj(end, :)';
end
