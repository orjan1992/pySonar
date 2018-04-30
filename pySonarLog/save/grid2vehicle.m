function [ x_veh, y_veh ] = grid2vehicle( x_grid, y_grid, range )
%GRID2VEHICLE Summary of this function goes here
%   Detailed explanation goes here
x_veh = (801 - y_grid) * range / 801.0;
y_veh = (x_grid - 801) * range / 801.0;
end

