clear, close all
save_figs = false;

load('collision_info.mat')
load('data.mat')
old_wps = old_wps(:, 1:2);
voronoi_ridge_vertices = voronoi_ridge_vertices + 1;
voronoi_ridge_points = voronoi_ridge_points + 1;
% voronoi_regions = voronoi_regions + 1;
voronoi_point_region = voronoi_point_region + 1;
orig_voronoi_wp = double(orig_voronoi_wp);
voronoi_indices = double(voronoi_indices);
pos = double(pos);
range_scale = double(range_scale);
startwp = [0, 0];
endwp = [40, 0];

%% Fig 0
f0 = figure();
colorOrder = [get(gca, 'ColorOrder'); get(gca, 'ColorOrder')];
hold on
axis equal
grid on
set(gca,'Ydir','Normal')

% contours
s = size(contours_convex);
for i = 1:s(2)
    s_obs = size(contours_convex{i});
    cur_obs = double(reshape(contours_convex{i}, [s_obs(1), s_obs(3)]));
    [x, y] = grid2vehicle(cur_obs(:, 1), cur_obs(:, 2), 30);
%     lines(1) = plot([y; y(1)], [x; x(1)], 'LineWidth', 2, 'Color', colorOrder(1, :));
    lines(1) = fill(y, x, colorOrder(1, :));
end

% Points
[n, e] = grid2ned(voronoi_points(:, 1), voronoi_points(:, 2), 30, 0, 0, 0);
lines(2) = plot(e, n, 'o', 'LineWidth', 2, 'Color', colorOrder(10, :));

% Voronoi ridges
[n, e] = grid2ned(voronoi_vertices(:, 1), voronoi_vertices(:, 2), 30, 0, 0, 0);
ne = [e n];
end_ind = length(voronoi_vertices);
start_ind = end_ind - 1;

for i = 1:length(voronoi_ridge_vertices)
    if any(voronoi_ridge_vertices(i, :) == 0)
        continue
    end
    if (voronoi_ridge_vertices(i, 1) == start_ind || voronoi_ridge_vertices(i, 1) == start_ind) || ...
        (voronoi_ridge_vertices(i, 1) == end_ind || voronoi_ridge_vertices(i, 1) == end_ind)
        continue
    end
    v1 = ne(voronoi_ridge_vertices(i, 1), :);
    v2 = ne(voronoi_ridge_vertices(i, 2), :);
    plot(v1(1), v1(2), 'o', 'LineWidth', 2, 'Color', colorOrder(5, :));
    lines(2) = plot([v1(1), v2(1)], [v1(2), v2(2)], 'Color', colorOrder(5, :), 'LineWidth', 1.5);
end

lines(3) = plot(startwp(1), startwp(2), '*', 'Color', colorOrder(10, :), 'MarkerSize', 10, 'LineWidth', 2);
lines(4) = plot(endwp(2), endwp(1), 'd', 'Color', colorOrder(7, :), 'MarkerSize', 10, 'LineWidth', 2);

xlim([-30, 30]);
ylim([-5, 60]);
xlabel('East [m]');
ylabel('North [m]');
legend(lines, {'Obstacles', 'Voronoi ridges', 'Start waypoint', 'End waypoint'})
if save_figs
    set(gcf, 'PaperUnits', 'normalized')
    set(gcf, 'PaperPosition', [0 0 1 1])
    print -depsc col0
end
clear lines