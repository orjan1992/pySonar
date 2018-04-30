function [ north, east ] = vehicle2ned( x_veh, y_veh, N_veh, E_veh, yaw )
%VEHICLE2NED Summary of this function goes here
%   Detailed explanation goes here
r = (x_veh.^2 + y_veh.^2).^0.5;
alpha = atan2(y_veh, x_veh);
north = N_veh + r .* cos(alpha + yaw);
east = E_veh + r .* sin(alpha + yaw);

end

