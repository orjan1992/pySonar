clear, close all
v = VideoWriter('mov.avi');
v.FrameRate = 0.8;
v.Quality = 100;
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
        if iscell(paths)
            for j = 1:s(2)
                path{path_counter} = paths{j};
                path_counter = path_counter + 1;
            end
        elseif length(s) > 2
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
pathshift = [5 6];
pathshifter = 1;
path_counter = 1;
line = '-';
fig = gcf;
fig.WindowState = 'maximized';
for i = 1:length(listing)
    if startsWith(listing(i).name, 'obstacles')
        if any(pathshift == path_counter)
            pathshifter = pathshifter + 1;
        end
        clf;
        fig = plot_obs_on_map(strcat(listing(i).folder, '\', listing(i).name), 'r', fig);
        fig = plot_map(fig);
        plot(pos_mat(:, 2), pos_mat(:, 1), 'b');
        if pathshifter < size(path, 2)
            plot(path{pathshifter}(:, 2), path{pathshifter}(:, 1), '--r')
            plot(path{pathshifter+1}(:, 2), path{pathshifter+1}(:, 1), '-r')
        else
            plot(path{pathshifter}(:, 2), path{pathshifter}(:, 1), '-r')
        end
        axis([4.579158540214643e+05 4.580717033350599e+05 6.821454878717383e+06 6.821610728030980e+06])
        drawnow;
        f = getframe();
        writeVideo(v,f);
        path_counter = path_counter + 1;
    end
end
close(v);