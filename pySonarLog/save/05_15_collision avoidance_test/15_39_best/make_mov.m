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
pathshift = [5 7 8 9 11 12 13 14];
skip = [3 4 6];
for i = 1:length(skip)
    path = [path(1:skip(i)-1), path(skip(i)+1:end)];
end
first = 3;
pathshifter = 1;
path_counter = 1;
line = '-';
fig = gcf;
extend_frame = false;
fig.WindowState = 'maximized';
%% Initial path
fig = plot_map(fig);
axis([4.579158540214643e+05 4.580717033350599e+05 6.821454878717383e+06 6.821610728030980e+06])
plot(path{pathshifter}(:, 2), path{pathshifter}(:, 1), '-r')
draw_rov([pos(1, 1:2), pos(1, 4)], 50, fig, 'r');
XTickLabel = get(gca,'XTick');
set(gca,'XTickLabel',num2str(XTickLabel'))
YTickLabel = get(gca,'YTick');
set(gca,'YTickLabel',num2str(YTickLabel'))
drawnow;
f = getframe();
writeVideo(v,f);
writeVideo(v,f);
% disp(path_counter);
% pause()

%% avoidance paths and obstacles
for i = 1:length(listing)
    if startsWith(listing(i).name, 'obstacles')
        clf;
        fig = plot_obs_on_map(strcat(listing(i).folder, '\', listing(i).name), 'r', fig);
        fig = plot_map(fig);
        plot(pos_mat(:, 2), pos_mat(:, 1), 'b');
        if pathshifter < size(path, 2) && path_counter >= first
            plot(path{pathshifter}(:, 2), path{pathshifter}(:, 1), '--r')
            plot(path{pathshifter+1}(:, 2), path{pathshifter+1}(:, 1), '-r')
        else
            plot(path{pathshifter}(:, 2), path{pathshifter}(:, 1), '-r')
        end
        axis([4.579158540214643e+05 4.580717033350599e+05 6.821454878717383e+06 6.821610728030980e+06])
        XTickLabel = get(gca,'XTick');
        set(gca,'XTickLabel',num2str(XTickLabel'))
        YTickLabel = get(gca,'YTick');
        set(gca,'YTickLabel',num2str(YTickLabel'))
        drawnow;
        f = getframe();
        writeVideo(v,f);
        if extend_frame
            writeVideo(v,f);
        end
%         disp(path_counter);
%         pause()
        path_counter = path_counter + 1;
%         if path_counter == first
%             pathshifter = pathshifter + 1;
%         end
        if any(pathshift == path_counter)
            pathshifter = pathshifter + 1;
            extend_frame = true;
        else
            extend_frame = false;
        end
    end
end
close(v);