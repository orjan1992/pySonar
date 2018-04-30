function [ north, east ] = grid2ned( x_grid, y_grid, range, N_veh, E_veh, yaw )
%GRID2NED Summary of this function goes here
%   Detailed explanation goes here
[x_veh, y_veh] = grid2vehicle(x_grid, y_grid, range);
[north, east] = vehicle2ned(x_veh, y_veh, N_veh, E_veh, yaw);
end

