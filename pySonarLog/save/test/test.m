clear, close all
fig = plot_obs_on_map('obs', 'c');
fig = plot_map_ed50(fig);
hold on;
load('paths');
plot(pos(:, 2), pos(:, 1));
xlim([4.579057301990855e+05 4.582663738581134e+05]);
ylim([6.821559293768059e+06 6.821919937427086e+06]);