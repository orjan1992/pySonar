clear, close all
v = VideoWriter('mov.avi');
v.FrameRate = 0.8;
open(v);
listing = dir();
colorOrder = get(gca, 'ColorOrder');
l = size(colorOrder, 1);
load('paths_20180514-152139');
line = '-';
k = 1;
fig = figure();
for i = 1:length(listing)
    if startsWith(listing(i).name, 'obstacles')
        clf;
        fig = plot_obs_on_map(strcat(listing(i).folder, '\', listing(i).name), 'r', fig);
        fig = plot_map_ed50(fig);
        plot(pos(:, 2), pos(:, 1), 'b');
        axis([4.579862078476861e+05 4.581146237249775e+05 6.821681313920572e+06 6.821809729797863e+06])
        f = getframe();
        writeVideo(v,f);
        k = k+1;
    end
end
close(v);