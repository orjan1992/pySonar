clear, close all
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
fig = figure();
hold on
fig.WindowState = 'maximize';
s = size(path);
for i = 1:s(2)
    plot(path{i}(:, 2), path{i}(:, 1))
end
plot(pos(:, 2), pos(:, 1), 'b');