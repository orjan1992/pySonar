clear, close all
v = VideoWriter('mov.avi');
v.FrameRate = 0.8;
open(v);
listing = dir();
colorOrder = get(gca, 'ColorOrder');
l = size(colorOrder, 1);
pos_mat = [];
path_mat = {};
path_counter = 1;
for i = 1:length(listing)
    if startsWith(listing(i).name, 'paths')
        load(listing(i).name);
        pos_mat = [pos_mat; pos];
        s = size(paths);
        if length(s) > 2
            for j = 1:s(1)
                path{path_counter} = squeeze(paths(j, :, :));
                path_counter = path_counter + 1;
            end
        else
            path{path_counter} = paths;
            path_counter = path_counter + 1;
        end
    end
end

line = '-';
k = 1;
fig = figure();
for i = 1:length(listing)
    if startsWith(listing(i).name, 'obstacles')
        clf;
        fig = plot_obs_on_map(strcat(listing(i).folder, '\', listing(i).name), 'r', fig);
        fig = plot_map(fig);
        plot(pos_mat(:, 2), pos_mat(:, 1), 'b');
        for j = 1:size(path, 2)
            plot(path{j}(:, 2), path{j}(:, 1))
        end
        axis([4.579158540214643e+05 4.580717033350599e+05 6.821454878717383e+06 6.821610728030980e+06])
        f = getframe();
        writeVideo(v,f);
        k = k+1;
    end
end
close(v);