clear, close all
[fig, leg_text, leg] = plot_map_ed50();
load('../05_14_from_garage_to_c_and_back_rerun/paths');
% plot(pos(:, 2), pos(:, 1), 'b');
axis([4.579953281194424e+05 4.581414368509383e+05 6.821668174986997e+06 6.821814283718495e+06]);
grid on
XTickLabel = get(gca,'XTick');
set(gca,'XTickLabel',num2str(XTickLabel'))
YTickLabel = get(gca,'YTick');
set(gca,'YTickLabel',num2str(YTickLabel'))

wp = pos(end, 1:2);
wp(1, :) = [6821801, 458050];
wp(2, :) = [6821790, 458030];
wp(3, :) = [6821760, 458060];
wp(4, :) = [6821740, 458080];
wp(5, :) = [6821740, 458110];
wp(end+1, :) = [6821750, 458125];
wp(end+1, :) = [6821730, 458135];
wp(end+1, :) = [6821707, 458065];
wp(end+1, :) = [6821680, 458035];
wp(end+1, :) = [6821676, 458080];
wp(end+1, :) = [6821750, 458100];
wp(end+1, :) = [6821795, 458060];
l1 = plot(wp(:, 2), wp(:, 1), '-*r');
save('wp3.mat', 'wp');
load('smooth3')
hold on
l2 = plot(wp(:, 2), wp(:, 1), '-*b');
legend([leg, l1, l2], [leg_text, 'Path', 'Smoothed path']);