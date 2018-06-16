clear, close all
f = "13_33 kollisjonskurs";
v = VideoWriter(char(f + ".avi"));
v.FrameRate = 0.8;
v.Quality = 100;
open(v);
listing = dir(char(f));
%% Sort listing
% Afields = fieldnames(listing);
% Acell = struct2cell(listing);
% sz = size(Acell);   
% Acell = reshape(Acell, sz(1), []); 
% Acell = Acell';
% Acell = sortrows(Acell, 3);
% Acell = reshape(Acell', sz);
% listing = cell2struct(Acell, Afields, 1);

colorOrder = get(gca, 'ColorOrder');
l = size(colorOrder, 1);
pos_mat = [];
path_counter = 1;
for i = 1:length(listing)
    if startsWith(listing(i).name, 'paths')
        load(char(listing(i).folder + "\" + listing(i).name));
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
first = true;
line = '-';
fig = gcf;
extend_frame = false;
% fig.WindowState = 'maximized';
%% avoidance paths and obstacles
coll_info = [];
for i = 1:length(listing)
    if startsWith(listing(i).name, 'collision') || startsWith(listing(i).name, 'obstacles')
        clf;
        if startsWith(listing(i).name, 'collision')
            coll_info = strcat(listing(i).folder, '\', listing(i).name);
            [fig, leg_text, leg] = plot_obs_on_map(coll_info, 'r', fig);
        else
            if ~isempty(coll_info)
                [fig, leg_text, leg] = plot_obs_on_map(strcat(listing(i).folder, '\', listing(i).name), 'r', fig);
                load(coll_info);
                l(1) = plot(old_wps(:, 2), old_wps(:, 1), '--m');
                l(2) = plot(new_wps(:, 2), new_wps(:, 1), '-m');
                legend([leg, l], [leg_text, {'Old paths', 'New Paths'}]);
            else
                continue
            end
        end
        fig = plot_map(fig);
        plot(pos_mat(:, 2), pos_mat(:, 1), 'b');
        
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
        path_counter = path_counter + 1;
    end
end
close(v);