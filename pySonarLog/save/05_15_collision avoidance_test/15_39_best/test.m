clear, close
wp = [801, 801; 667, 802; 622, 780; 467, 653; 392, 596; 481, 439; 519, 299; 810, -4213; 1620, 77];
[fig, leg_text, leg] = plot_collision_info('collision_info20180515-154125', 1, true);
fig.WindowState = 'maximize';
% ax = gca;
% gca.Mode = 
axis equal
hold on;
load('collision_info20180515-154125');
% wp_grid = ned2grid(wp, pos, range_scale);
% l = plot(wp_grid(:, 1), wp_grid(:, 2), 'k--o', 'LineWidth', 1);
% dijkstra = [801, 801;808, 791;801, 720;810, -4213;1620, 77];
dijkstra = [801, 801;667, 802;622, 780;467, 653;392, 596;481, 439;519, 299;541, 101;580, -413;810, -4213;1620, 77];
l(2)= plot(dijkstra(:, 1), dijkstra(:, 2), '-*k', 'LineWidth', 2);

smooth = [801, 801; 667, 802; 622, 780; 467, 653; 392, 596; 481, 439; 519, 299; 810, -4213; 1620, 77];
plot(smooth(:, 1), smooth(:, 2), '*r');
% legend([leg l], [leg_text, {'new method', 'orig'}]);
% t = [289    76   291     2   290];
% v = voronoi_vertices(t, :);
% plot(v(:, 1), v(:, 2), '-*k');


f = 2;
xlim([-1600, 3200]*f);
ylim([-1600, 3200]*f);

% [fig, leg_text, leg] = plot_collision_info('collision_info20180515-154112', 1, false);
% dijkstra = [801, 801;652, 762;370, 710;234, 684;209, 680;267, 574;376, 109;391, 43;816, -109];
% plot(dijkstra(:, 1), dijkstra(:, 2), '-*k', 'LineWidth', 2);