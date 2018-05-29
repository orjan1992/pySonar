function xyz = ed50towgs84(E, N, Z)
%ED50TOWGS84 Summary of this function goes here
%   Detailed explanation goes here
T = [-116.641; -56.931; -110.559];

r_x =	0.893;
r_y = 0.921;
r_z = -0.917;
s_d = -3.52;

R = [1, -r_z, r_y; r_z, 1, -r_x; -r_y, r_x, 1];

if ~exist('Z', 'var')
    Z = zeros(size(E, 1));
end
xyz = zeros(size(E, 1), 3);
for i = 1:size(E, 1)
    tmp = T + (1+s_d*10^-6)*R*[E(i); N(i); Z(i)];
    xyz(i, :) = [tmp(2), tmp(1), tmp(3)];
end

