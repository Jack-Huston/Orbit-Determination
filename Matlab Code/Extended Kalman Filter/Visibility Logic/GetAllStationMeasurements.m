function [y_k, visible_stations_k, y_k_stacked] = GetAllStationMeasurements(x, t)
    % TODO: Check if this should be global
    N_stations = 12;

    y_k = [];
    visible_stations_k = [];
    y_k_stacked = [];
    for i = 1:N_stations
        [y_i, is_visible] = GetStationMeasurement(x, t, i);
        if is_visible
            % Add measurement vector y^i(t) to stacked measurement vector
            y_k = [y_k; y_i]; 
            stationID = i;
            visible_stations_k = [visible_stations_k stationID];
            identified_measurements = [y_i;stationID];
            y_k_stacked = [y_k_stacked identified_measurements];
        end
    end
end