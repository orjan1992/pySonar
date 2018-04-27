function wp_veh = ned2vehicle( wp, pos )
%NED2VEHICLE Summary of this function goes here
%   Detailed explanation goes here
xy = wp - pos(:, 1:2);
r = sqrt(xy(:, 1).^2 + xy(:, 2).^2);
alpha = atan2(xy(:, 2), xy(:, 1));

wp_veh = [r.*cos(alpha), r.*sin(alpha)];
end

