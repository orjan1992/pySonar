clear, close all
[fig, leg_text, leg] = plot_map();
load('../05_14_from_garage_to_c_and_back_rerun/paths');
% plot(pos(:, 2), pos(:, 1), 'b');
grid on
XTickLabel = get(gca,'XTick');
set(gca,'XTickLabel',num2str(XTickLabel'))
YTickLabel = get(gca,'YTick');
set(gca,'YTickLabel',num2str(YTickLabel'))

wp = [6821583.6601, 457959.0446];
wp(end+1, :) = [6821568.6356, 457962.9016];
wp(end+1, :) = [6821554.0957, 457977.5585];
wp(end+1, :) = [6821540.5252, 457987.5868];
wp(end+1, :) = [6821538.5865, 458024.2289];
wp(end+1, :) = [6821523.562, 458040.0428] ;
wp(end+1, :) = [6821495.9362, 458031.1716];
wp(end+1, :) = [6821495.4515, 457992.9867];
wp(end+1, :) = [6821501.2675, 457972.93]  ;
wp(end+1, :) = [6821520.654, 457974.4728] ;
wp(end+1, :) = [6821543.9178, 457971.0015];
wp(end+1, :) = [6821579.2982, 457965.9873];
wp(end+1, :) = [6821588.0221, 457965.2159];
l1 = plot(wp(:, 2), wp(:, 1), '-*r');
% axis([4.579953281194424e+05 4.581414368509383e+05 6.821668174986997e+06 6.821814283718495e+06]);
save('wp_wgs84', 'wp');
load('smooth_wgs84')
hold on
l2 = plot(wp(:, 2), wp(:, 1), '-*b');
legend([leg, l1, l2], [leg_text, 'Path', 'Smoothed path']);
% set (gcf, 'WindowButtonDownFcn', @mouseDown);