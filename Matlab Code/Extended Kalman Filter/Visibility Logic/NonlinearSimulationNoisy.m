function outputs = NonlinearSimulationNoisy(x_0,t_0,T,Dt,S_w,S_v)
    t = t_0:Dt:T;
    % 1. Simulate the states
    options = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);
    [~,x] = ode45(@NonlinearStateEquations, t, x_0, options);
    outputs.x = x;
    if numel(t) == 2
        outputs.x = x(end,:);
        x = x(end,:);
    end
    n = size(x,2);
    N_x = size(x,1);
    q_k = randn(N_x, n)';
    w_k = S_w * q_k; % Process noise
    x = x' + w_k;
    x = x';
    % 2. Calculate the Measurements at Each Time Step
    N_steps = (T-t_0)/Dt;
    y = cell(N_steps, 1); % Use a cell array for variable-length vectors
    visible_stations = cell(N_steps,1);
    y_stacked = cell(N_steps,1);
    
    for k = 1:N_steps
        current_t = t(k+1);
        current_x = x(k+1, :)';
        [y_k,visible_stations{k},y_stacked{k}] = GetAllStationMeasurements(current_x, current_t);
        p = size(y_k,1);
        q_k = randn(p, 1);
        N_visible_stations_k = numel(visible_stations{k});
        S_v_diag = mat2cell(repmat(S_v,1,N_visible_stations_k),3,3*ones(1,N_visible_stations_k));
        S_v_blk = blkdiag(S_v_diag{:});
        v_k = S_v_blk * q_k; % Measurement noise
        y_k = y_k + v_k;
        
        y_stacked{k} = reshape(y_stacked{k},4,[]) + [reshape(v_k,3,[]);zeros(1,N_visible_stations_k)];
        y{k} = y_k';
    end
    outputs.y = y;
    outputs.visible_stations = visible_stations;
    outputs.y_stacked = y_stacked;
end