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
wp(end+1, :) = [6821785.9565, 458049.2207];
wp(end+1, :) = [6821786.3825, 458049.2207];
wp(end+1, :) = [6821778.715, 458058.1172] ;
wp(end+1, :) = [6821765.5098, 458060.8546];
wp(end+1, :) = [6821739.9514, 458058.4594];
wp(end+1, :) = [6821738.2475, 458065.9873];
wp(end+1, :) = [6821753.5826, 458068.7247];
wp(end+1, :) = [6821748.4709, 458119.0244];
wp(end+1, :) = [6821766.3618, 458115.6026];
wp(end+1, :) = [6821771.8994, 458106.3639];
wp(end+1, :) = [6821755.7124, 458094.3878];
wp(end+1, :) = [6821759.1202, 458077.279] ;
wp(end+1, :) = [6821776.5851, 458069.0668];
wp(end+1, :) = [6821779.1409, 458063.592] ;
wp(end+1, :) = [6821801.2915, 458056.7485];
l1 = plot(wp(:, 2), wp(:, 1), '-*r');
axis([4.579953281194424e+05 4.581414368509383e+05 6.821668174986997e+06 6.821814283718495e+06]);
% set (gcf, 'WindowButtonDownFcn', @mouseDown);
save('wp2.mat', 'wp');
load('smooth2')
hold on
l2 = plot(wp(:, 2), wp(:, 1), '-*b');
legend([leg, l1, l2], [leg_text, 'Path', 'Smoothed path']);