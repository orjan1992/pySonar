function wp_grid = ned2grid(wp, pos, range )
%NED2GRID Summary of this function goes here
%   Detailed explanation goes here

wp_veh = ned2vehicle(wp, pos);
wp_grid = vehicle2grid(wp_veh, range);
end

