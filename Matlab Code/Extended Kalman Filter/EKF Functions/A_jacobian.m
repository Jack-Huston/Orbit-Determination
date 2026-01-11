function A = A_jacobian(x, mu)
%A_JACOBIAN  Continuous-time dynamics Jacobian A(x) = df/dx.
%Inputs:
%  x  - 4x1 state [X; Xdot; Y; Ydot]
%  mu - gravitational parameter

%Output:
%  A  - 4x4 Jacobian matrix evaluated at x

%Pull out and calculate commonly used variables
x1 = x(1);
x3 = x(3);
r = sqrt(x1^2 + x3^2);
r5 = r^5;

%Calculate Matrix Inputs
A_11 =  mu * (2 * x1^2 - x3^2) / r5;
A_13 =  3 * mu * x1 * x3 / r5;
A_31 =  3 * mu * x1 * x3 / r5;
A_33 =  mu * (2 * x3^2 - x1^2) / r5;

%Form Output Matrix
A = [0     1    0     0;
     A_11  0    A_13  0;
     0     0    0     1;
     A_31  0    A_33   0];
end
