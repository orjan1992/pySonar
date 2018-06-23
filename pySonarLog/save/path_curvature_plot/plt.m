clear, close all
load('data');
fig = figure();
yyaxis left
plot(p, v);
hold on;
ylabel('[m/s]');
cum_dist = [0 cum_dist];
curvature = [0 curvature];
yyaxis right
ylabel('[rad/m]');
plot(cum_dist, curvature, '*-');
xlabel('Distance [m]')
legend('Velocity', 'Curvature');
save_fig(fig, 'curvature', true);


fig2 = figure();
plot(smooth(:, 1), smooth(:, 2));
hold on;
plot([0 5 8 1]', [0 0 6 5]', '--*');
h = legend('Smooth path', 'Waypoints');
h.Location = 'southeast';
xlabel('East [m]');
ylabel('North [m]');
ylim([-.2 6.2]);
xlim([-.2 8.2]);
grid on;
save_fig(fig2, 'curvature_path', true);