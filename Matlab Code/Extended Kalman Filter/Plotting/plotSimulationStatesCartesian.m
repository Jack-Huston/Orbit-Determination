function plotSimulationStatesCartesian(t,x)
%% Plot

[~, n]= size(x);
plot(x(:,1),x(:,3));
axis equal;
grid on;
% i_x_A = x_A_indices(subplot_i);
% scatter(t_window,x_A(i_x_A,k_window+1));

end