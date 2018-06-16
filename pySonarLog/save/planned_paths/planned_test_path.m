clear, close all
[fig, leg_text, leg] = plot_map_ed50();
load('../05_14_from_garage_to_c_and_back_rerun/paths');
% plot(pos(:, 2), pos(:, 1), 'b');
grid on
XTickLabel = get(gca,'XTick');
set(gca,'XTickLabel',num2str(XTickLabel'))
YTickLabel = get(gca,'YTick');
set(gca,'YTickLabel',num2str(YTickLabel'))

wp = pos(end, 1:2);
wp(1, :) = [6821801, 458050];
wp(end+1, :) = [6821784.6786, 458049.2207];
wp(end+1, :) = [6821774.0293, 458063.592];
wp(end+1, :) = [6821752.3047, 458074.5416];
wp(end+1, :) = [6821755.2865, 458112.1809];
wp(end+1, :) = [6821723.3385, 458124.4992];
wp(end+1, :) = [6821708.8554, 458110.47];
wp(end+1, :) = [6821708.4294, 458082.0695];
wp(end+1, :) = [6821695.2243, 458071.1199];
wp(end+1, :) = [6821694.3723, 458061.8812];
wp(end+1, :) = [6821720.3567, 458057.7751];
wp(end+1, :) = [6821751.8787, 458059.8281];
wp(end+1, :) = [6821784.6786, 458060.1703];
wp(end+1, :) = [6821800.4396, 458055.0377];
l1 = plot(wp(:, 2), wp(:, 1), '-*r');
axis([4.579953281194424e+05 4.581414368509383e+05 6.821668174986997e+06 6.821814283718495e+06]);

load('smooth')
hold on
l2 = plot(wp(:, 2), wp(:, 1), '-*b');
legend([leg, l1, l2], [leg_text, 'Path', 'Smoothed path']);
% set (gcf, 'WindowButtonDownFcn', @mouseDown);