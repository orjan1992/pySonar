clear, close all
save_figs = true;

load('collision_info20180623-121724')
% load('data.mat')
old_wps = old_wps(:, 1:2);
voronoi_ridge_vertices = voronoi_ridge_vertices + 1;
voronoi_ridge_points = voronoi_ridge_points + 1;
% voronoi_regions = voronoi_regions + 1;
voronoi_point_region = voronoi_point_region + 1;
orig_voronoi_wp = ned2grid(double(old_wps), double(pos), double(range_scale));
voronoi_indices = double(voronoi_indices);
pos = double(pos);
range_scale = double(range_scale);
startwp = [0, 0];
endwp = [40, 0];
orig_list = orig_list +1;

%% Fig 0
f0 = figure();
colorOrder = [get(gca, 'ColorOrder'); get(gca, 'ColorOrder')];
hold on
axis equal
grid on
set(gca,'Ydir','Normal')

% contours
s = size(obstacles);
for i = 1:s(2)
    s_obs = size(obstacles{i});
    cur_obs = double(reshape(obstacles{i}, [s_obs(1), s_obs(3)]));
    [x, y] = grid2vehicle(cur_obs(:, 1), cur_obs(:, 2), 30);
%     lines(1) = plot([y; y(1)], [x; x(1)], 'LineWidth', 2, 'Color', colorOrder(1, :));
    lines(1) = fill(y, x, colorOrder(1, :));
end

% Points
% [n, e] = grid2ned(voronoi_points(:, 1), voronoi_points(:, 2), 30, 0, 0, 0);
% lines(2) = plot(e, n, 'o', 'LineWidth', 2, 'Color', colorOrder(2, :));

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
legend(lines, {'Obstacles', 'VD ridges', 'Start WP', 'End WP'})
% if save_figs
%     set(gcf, 'PaperUnits', 'normalized')
%     set(gcf, 'PaperPosition', [0 0 1 1])
%     print -depsc col0
% end
clear lines
%% Fig 1
f1 = figure();
colorOrder = [get(gca, 'ColorOrder'); get(gca, 'ColorOrder')];
hold on
axis equal
grid on
set(gca,'Ydir','Normal')

% contours
s = size(obstacles);
for i = 1:s(2)
    s_obs = size(obstacles{i});
    cur_obs = double(reshape(obstacles{i}, [s_obs(1), s_obs(3)]));
    [x, y] = grid2vehicle(cur_obs(:, 1), cur_obs(:, 2), 30);
%     lines(1) = plot([y; y(1)], [x; x(1)], 'LineWidth', 2, 'Color', colorOrder(1, :));
    lines(1) = fill(y, x, colorOrder(1, :));
end

% Points
% [n, e] = grid2ned(voronoi_points(:, 1), voronoi_points(:, 2), 30, 0, 0, 0);
% lines(2) = plot(e, n, 'o', 'LineWidth', 2, 'Color', colorOrder(2, :));

% Voronoi ridges
[n, e] = grid2ned(voronoi_vertices(:, 1), voronoi_vertices(:, 2), 30, 0, 0, 0);
ne = [e n];

for i = 1:length(voronoi_ridge_vertices)
    if any(voronoi_ridge_vertices(i, :) == 0)
        continue
    end
    valid = false;
    if connection(voronoi_ridge_vertices(i, 1), voronoi_ridge_vertices(i, 2)) ~= 0
        valid = true;
    end
    v1_valid = any(connection(voronoi_ridge_vertices(i, 1), :) ~= 0);
    v1 = ne(voronoi_ridge_vertices(i, 1), :);
    v2 = ne(voronoi_ridge_vertices(i, 2), :);
    if v1_valid
        plot(v1(1), v1(2), 'o', 'LineWidth', 2, 'Color', colorOrder(5, :));
    else
        plot(v1(1), v1(2), 'o', 'Color', colorOrder(4, :));
    end
    if valid
        lines(2) = plot([v1(1), v2(1)], [v1(2), v2(2)], 'Color', colorOrder(5, :), 'LineWidth', 1.5);
    else
        lines(3) = plot([v1(1), v2(1)], [v1(2), v2(2)], 'Color', colorOrder(4, :));
    end
end

lines(4) = plot(startwp(1), startwp(2), '*', 'Color', colorOrder(10, :), 'MarkerSize', 10, 'LineWidth', 2);
lines(5) = plot(endwp(2), endwp(1), 'd', 'Color', colorOrder(7, :), 'MarkerSize', 10, 'LineWidth', 2);

xlim([-30, 30]);
ylim([-5, 60]);
xlabel('East [m]');
ylabel('North [m]');
legend(lines, {'Obstacles', sprintf('Valid VD\nridges'), sprintf('Invalid VD\nridges'), 'Start WP', 'End WP'})
% if save_figs
%     set(gcf, 'PaperUnits', 'normalized')
%     set(gcf, 'PaperPosition', [0 0 1 1])
%     print -depsc col1
% end
clear lines

%% Fig 2
f2 = figure();
colorOrder = [get(gca, 'ColorOrder'); get(gca, 'ColorOrder')];
hold on
axis equal
grid on
set(gca,'Ydir','Normal')

% contours
s = size(obstacles);
for i = 1:s(2)
    s_obs = size(obstacles{i});
    cur_obs = double(reshape(obstacles{i}, [s_obs(1), s_obs(3)]));
    [x, y] = grid2vehicle(cur_obs(:, 1), cur_obs(:, 2), 30);
%     lines(1) = plot([y; y(1)], [x; x(1)], 'LineWidth', 2, 'Color', colorOrder(1, :));
    lines(1) = fill(y, x, colorOrder(1, :));
end

% Points
% [n, e] = grid2ned(voronoi_points(:, 1), voronoi_points(:, 2), 30, 0, 0, 0);
% lines(2) = plot(e, n, 'o', 'LineWidth', 2, 'Color', colorOrder(2, :));

% Voronoi ridges
[n, e] = grid2ned(voronoi_vertices(:, 1), voronoi_vertices(:, 2), 30, 0, 0, 0);
ne = [e n];
% plot(e, n, 'o', 'LineWidth', 2, 'Color', colorOrder(5, :));
for i = 1:length(voronoi_ridge_vertices)
    if any(voronoi_ridge_vertices(i, :) == 0)
        continue
    end
    valid = false;
    if connection(voronoi_ridge_vertices(i, 1), voronoi_ridge_vertices(i, 2)) ~= 0
        valid = true;
    end
    v1 = ne(voronoi_ridge_vertices(i, 1), :);
    v2 = ne(voronoi_ridge_vertices(i, 2), :);
    if valid
        lines(2) = plot([v1(1), v2(1)], [v1(2), v2(2)], 'Color', colorOrder(5, :), 'LineWidth', 1.5);
    end
%         lines(4) = plot([v1(1), v2(1)], [v1(2), v2(2)], 'Color', colorOrder(4, :));
%     end
end

% old path
lines(3) = plot(old_wps(:, 2), old_wps(:, 1), '-*r', 'LineWidth', 2);

% new path
dijkstra = voronoi_vertices(orig_list, :);
% [n, e] = grid2ned(orig_voronoi_wp(:, 1), orig_voronoi_wp(:, 2), 30, 0, 0, 0);
[dijkstra_n, dijkstra_e] = grid2ned(dijkstra(:, 1), dijkstra(:, 2), 30, 0, 0, 0);
lines(4) = plot(dijkstra_e, dijkstra_n, '-*', 'LineWidth', 2, 'Color', colorOrder(6, :));

lines(5) = plot(startwp(1), startwp(2), '*', 'Color', colorOrder(10, :), 'MarkerSize', 10, 'LineWidth', 2);
lines(6) = plot(endwp(2), endwp(1), 'd', 'Color', colorOrder(7, :), 'MarkerSize', 10, 'LineWidth', 2);

xlim([-30, 30]);
ylim([-5, 60]);
xlabel('East [m]');
ylabel('North [m]');
legend(lines, {'Obstacles', sprintf('Valid VD\nridges'), 'Old path', 'Djikstra path', 'Start WP', 'End WP'})
% if save_figs
%     set(gcf, 'PaperUnits', 'normalized')
%     set(gcf, 'PaperPosition', [0 0 1 1])
%     print -depsc col2
% end
clear lines
%% Fig 3
f3 = figure();
colorOrder = [get(gca, 'ColorOrder'); get(gca, 'ColorOrder')];
hold on
axis equal
grid on
set(gca,'Ydir','Normal')

% contours
s = size(obstacles);
for i = 1:s(2)
    s_obs = size(obstacles{i});
    cur_obs = double(reshape(obstacles{i}, [s_obs(1), s_obs(3)]));
    [x, y] = grid2vehicle(cur_obs(:, 1), cur_obs(:, 2), 30);
%     lines(1) = plot([y; y(1)], [x; x(1)], 'LineWidth', 2, 'Color', colorOrder(1, :));
    lines(1) = fill(y, x, colorOrder(1, :));
end


% new path
% [n, e] = grid2ned(orig_voronoi_wp(:, 1), orig_voronoi_wp(:, 2), 30, 0, 0, 0);
% lines(2) = plot(e, n, '-*', 'LineWidth', 2, 'Color', colorOrder(6, :));
lines(2) = plot(dijkstra_e, dijkstra_n, '-*', 'LineWidth', 2, 'Color', colorOrder(6, :));


% smooth
lines(3) = plot(new_wps(:, 2), new_wps(:, 1), '-*', 'LineWidth', 2, 'Color', colorOrder(3, :));

% obsolete removed
% [n, e] = grid2ned(voronoi_indices(:, 1), voronoi_indices(:, 2), 30, 0, 0, 0);
% lines(3) = plot(e, n, '-.*', 'LineWidth', 2, 'Color', colorOrder(2, :));
lines(4) = plot(non_smooth_path(:, 2), non_smooth_path(:, 1), '--*', 'LineWidth', 2, 'Color', colorOrder(2, :));

lines(5) = plot(startwp(1), startwp(2), '*', 'Color', colorOrder(10, :), 'MarkerSize', 10, 'LineWidth', 2);
lines(6) = plot(endwp(2), endwp(1), 'd', 'Color', colorOrder(7, :), 'MarkerSize', 10, 'LineWidth', 2);

xlim([-30, 30]);
ylim([-5, 60]);
xlabel('East [m]');
ylabel('North [m]');
legend(lines, {'Obstacles', 'Djikstra path', sprintf('Smoothed with\nFermat''s spiral'), sprintf('Obsolete WPs\n removed'), 'Start WP', 'End WP'})
% if save_figs
%     set(gcf, 'PaperUnits', 'normalized')
%     set(gcf, 'PaperPosition', [0 0 1 1])
%     print -depsc col3
% end
if save_figs
    save_fig(f0, 'col0', true, true);
    save_fig(f1, 'col1', true);
    save_fig(f2, 'col2', true);
    save_fig(f3, 'col3', true);
end
