clear, close all
v = VideoWriter('peaks.avi');
v.FrameRate = 1;
open(v);
listing = dir();
fig = figure();
colorOrder = get(gca, 'ColorOrder');
l = size(colorOrder, 1);
load('paths_20180514-135417');
line = '-';
k = 1;
fig = figure();
for i = 1:length(listing)
    if startsWith(listing(i).name, 'obstacles')
        clf;
        fig = plot_obs_on_map(strcat(listing(i).folder, '\', listing(i).name), 'r', fig);
        fig = plot_map_ed50(fig);
        plot(pos(:, 2), pos(:, 1), 'b');
        axis([4.580112239362499e+05 4.581316138212106e+05 6.821677963662780e+06 6.821798353547739e+06])
        f = getframe();
        writeVideo(v,f);
        k = k+1;
    end
end
close(v);