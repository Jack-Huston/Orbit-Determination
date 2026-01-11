function ydata_simulated = convertOutput2Ydata(outputs)
ydata_simulated = cell(size(outputs.y));
for i_y = 1:numel(outputs.y)
    y_k = outputs.y{i_y};
    visible_stations_k = outputs.visible_stations{i_y};
    ydata_simulated_k = [reshape(y_k,3,[]); visible_stations_k];
    ydata_simulated{i_y} = ydata_simulated_k;
end
end