clear, close all
speed = 7;
save_figs = true;
columns = 1;

% folder_char = "/"; % Linux
folder_char = "\"; % windows

% f = "06_16_sim\12_47_god";
% limits_fig = [-42.0044   44.3963 -30.353639679008740 37.791405654324656];
% sim = true;

% f = "06_16_sim\12_42_god_coll";
% limits_fig = [-42.0044   44.3963 -30.353639679008740 37.791405654324656];
% sim = true;

f = "06_16_sim\12_33_ok_coll";
limits_fig = [-42.0044   44.3963 -30.353639679008740 37.791405654324656];
sim = true;


if ~exist('sim', 'var')
    sim = false;
end

%% Start
if ~save_figs
%     v = VideoWriter(char(f), 'avi');
    v = VideoWriter(char(f), 'MPEG-4');
    v.FrameRate = 30;
    v.Quality = 100;
    open(v);
end

%% sort
listing = dir(char(f));
for i = 1:length(listing)
    ok = startsWith(listing(i).name, 'collision') && endsWith(listing(i).name, '.mat');
    if ~save_figs
        ok = ok || startsWith(listing(i).name, 'obstacles') && endsWith(listing(i).name, '.mat');
    end
    if ok
        n = listing(i).name;
        s_i = regexp(n, '\d{8}-\d{6}');
%         f_time(i) = datetime([str2double(n(s_i:s_i+3)), str2double(n(s_i+4:s_i+5)), ...
%             str2double(n(s_i+6:s_i+7)), str2double(n(s_i+9:s_i+10)), ...
%             str2double(n(s_i+11:s_i+12)), str2double(n(s_i+13:s_i+14))]);
        f_time(i) = datetime(n(s_i:s_i+14), 'InputFormat','yyyyMMdd-HHmmss');
        f_ind(i) = i;
    else
        f_time(i) = datetime(2020, 12, 12);
        f_ind(i) = 0;
    end
end
tmp = sortrows(table(f_time', f_ind'), 'Var1', 'ascend');
f_time = table2array(tmp(:, 1));
f_ind = table2array(tmp(:, 2));
i = find(f_ind == 0);
f_time = f_time(1:i-1);
f_ind = f_ind(1:i-1);

%% Extract paths and positions
pos_mat = [];
path_counter = 1;
for i = 1:length(listing)
    if startsWith(listing(i).name, 'Los_log')
        data(i) = load(char(listing(i).folder + folder_char + listing(i).name));
        pos_mat = [pos_mat; data(i).pos];
        path{path_counter} = data(i).path;
        path_counter = path_counter + 1;
    end
end
first = true;
all_obj = [];
line = '-';
fig = gcf;
if ~save_figs
    fig.WindowState = 'maximized';
end

l1 = [];
new_path = path{1};
%% avoidance paths and obstacles
for i = 1:length(f_time)
    delete([all_obj l1]);
    l1 = [];
    if startsWith(listing(f_ind(i)).name, 'collision')
        load(strcat(listing(f_ind(i)).folder, folder_char, listing(f_ind(i)).name));
        last_path = old_wps;
        new_path = new_wps;
    else
        hold on
        if exist('last_path', 'var') && ~isempty(last_path)
            l1(end+1) = plot(last_path(:, 2), last_path(:, 1), '--m');
        end
        if exist('new_path', 'var') && ~isempty(new_path)
            l1(end+1) =plot(new_path(:, 2), new_path(:, 1), 'm');
        end
    end
    [fig, leg_text, leg, all_obj] = plot_obs_on_map(strcat(listing(f_ind(i)).folder, folder_char, listing(f_ind(i)).name), 'r', fig);
    uistack(all_obj, 'bottom');
    if first
        if sim
            [fig, leg2, leg_text2] = plotMapSim(fig);
        else
            if ed50
                [fig, leg_text2, leg2] = plot_map_ed50(fig);
            else
                [fig, leg_text2, leg2] = plot_map(fig);
            end
        end
        l(1) = plot(pos_mat(:, 2), pos_mat(:, 1), 'b');
        l_t = {'Actual Path'};
        if length(leg_text) < 5
            l(2) = plot(0, 0, '--m');
            l(3) = plot(0, 0, '-m');
            l_t = [l_t, {'Old Path', 'New Path'}];
        end
        legend([leg2, leg, l], [leg_text2, leg_text, l_t ],'AutoUpdate','off', 'Color', 'None');
        axis equal
        if save_figs
            axis(limits_fig);
        else
            axis(limits);
        end
        axis manual;
        XTickLabel = get(gca,'XTick');
        set(gca,'XTickLabel',num2str(XTickLabel'))
        xtickangle(45)
        YTickLabel = get(gca,'YTick');
        set(gca,'YTickLabel',num2str(YTickLabel'))
        ytickangle(45);
    end
%     if size(pos, 2) > 4
%         alt 
%         
%     alt_text_box = annotation('textbox',[.2 .5 .3 .3],'String',sprintf('%.2f', alt));    
    drawnow;
    if save_figs
        n = split(listing(f_ind(i)).name, '.');
        save_fig(fig, strcat(listing(f_ind(i)).folder, folder_char, 'figs', folder_char, n{1}), first)
    else
        f = getframe();

        if i < length(f_time)
            dt = seconds(f_time(i+1) - f_time(i));
            df = round(dt*v.FrameRate/speed);
            for j= 1:df
                writeVideo(v,f);
            end
        else
            for j= 1:round(v.FrameRate/speed)
                    writeVideo(v,f);
            end
        end
    end
    if first
        first = false;
    end
    path_counter = path_counter + 1;
end
if save_figs
    if folder_char == "\"
        gen_tex(f, columns);
    end
else
    close(v);
end