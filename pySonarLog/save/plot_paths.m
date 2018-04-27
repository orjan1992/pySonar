clear, close all
load('paths_20180427-102600.mat');
f = figure();
hold on
plot(pos(:, 1), pos(:, 2), 'r');
plot(pos(1, 1), pos(1, 2), 'r*');
plot(pos(end, 1), pos(end, 2), 'or');
for i = 1:length(paths)
    plot(paths{i}(:, 1), paths{i}(:, 2), '--')
end