clear, close all
data = readtable('14-43-04-56 log.csv');
fig = figure();
hold on;
% plot_path_on_map(fig, 'Los_log_20180427-144507');
% plot_path_on_map(fig, 'Los_log_20180427-144526');
% plot_path_on_map(fig, 'Los_log_20180427-144526');
% plot_path_on_map(fig, 'Los_log_20180427-144535');
% plot_path_on_map(fig, 'Los_log_20180427-144537');
% plot_path_on_map(fig, 'Los_log_20180427-144551');
% plot_path_on_map(fig, 'Los_log_20180427-144555');
% plot(path(:, 2), path(:, 1));
plot(data.East, data.North);
axis equal