function plotGridBinary(fig, file_number, rov_size)
figure(fig);

load(sprintf('logs/bin_grid_%i.mat', file_number), 'bin_grid', 'xmax', 'ymax', 'head', 'long', 'lat');
head = pi/2-head;

% % Grid
cell_size = 2*xmax/size(bin_grid, 2);
grid_x = -xmax+cell_size/2:cell_size:xmax-cell_size/2;
grid_y = double(ymax)-cell_size/2:-cell_size:0;
[X, Y] = meshgrid(grid_y, grid_x);
co = cos(head);
si = sin(head);
tmp = X*co-Y*si;
Y = si*X+co*Y;
X = tmp;
X = X+long;
Y = Y+lat;
contourf(X, Y, fliplr(bin_grid)')
hold on;
color_map = [1, 1, 1; 1, 0, 0];
colormap(color_map)

% ROV
% hold on
shape = fliplr([-.5, -.5; .5, -.5; .5, .5; 0, 1; -.5, .5; -.5, -.5]);
ROV = rov_size*shape;
ROV = [ROV(:, 1)*co - ROV(:, 2)*si, ROV(:, 1)*si + ROV(:, 2)*co] + [long lat];
plot(ROV(:, 1), ROV(:, 2), 'g', 'Linewidth', 1.5);

grid_end_points = [X(1, 1), Y(1, 1); X(1, end), Y(1, end); X(end, end), Y(end, end); X(end, 1), Y(end, 1); X(1, 1), Y(1, 1)];

plot(grid_end_points(:, 1), grid_end_points(:, 2))
axis equal
xlim([-25 35]);
ylim([-30 30]);
ylim manual
xlim manual
end

