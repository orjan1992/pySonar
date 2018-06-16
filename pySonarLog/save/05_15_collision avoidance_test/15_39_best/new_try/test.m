clear, close
fname = 'collision_info20180515-154041';
[fig, leg_text, leg] = plot_collision_info([fname, '_2'], 1, false);
fig.WindowState = 'maximize';
axis equal
hold on;
load(fname);

f = 0.75;
xlim([-1600, 3200]*f);
ylim([-1600, 3200]*f);

load(fname);
new_wps = ned2grid(new_wps(:, 1:2), pos, range_scale);
plot(new_wps(:, 1), new_wps(:,2), 'r');