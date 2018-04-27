function wp_grid = vehicle2grid( wp_veh, range )
%VEHICLE2GRID Summary of this function goes here
%   Detailed explanation goes here

wp_grid = [wp_veh(:, 2).*801/range + 801, 801 - wp_veh(:, 1).*801/range];
end

