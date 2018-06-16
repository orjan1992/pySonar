clear, close all
a = load('pos1');
b = load('pos2');
c = load('pos3');
pos = [a.pos; b.pos; c.pos];
time = [a.time b.time c.time];
time = datetime(time, 'convertfrom','posixtime') + hours(2);
% pos(:, 1:2) = pos(:, 1:2) - [36.3113    8.1790];
pos(:, 1:2) = pos(:, 1:2) - [36.3113    7.8];
save('pos', 'pos', 'time')
load('pos');
f = plot_map();
hold on;
plot(pos(:, 2), pos(:, 1), 'b')
axis([4.579261871477406e+05 4.581098271216768e+05 6.821465520303811e+06 6.821649160277746e+06]);
f.WindowState = 'Maximized';