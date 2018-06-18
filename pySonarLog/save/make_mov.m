clear, close all
speed = 7;
save_figs = true;
columns = 1;

% folder_char = "/"; % Linux
folder_char = "\"; % windows

% f = "05_15_collision avoidance_test\15_30_no_smoothing";
% limits = [4.579312341240512e+05 4.580370455463799e+05 6.821491079279837e+06 6.821596890702165e+06];
% limits_fig = [4.579312341240512e+05 4.580370455463799e+05 6.821491079279837e+06 6.821596890702165e+06];
% ed50 = false;

% f = "05_15_collision avoidance_test\15_13_path_rundt_k";
% limits = [4.579057651462758e+05 4.580671674671181e+05 6.821453893920844e+06 6.821615296241687e+06];
% limits_fig = [4.579158540214643e+05 4.580717033350599e+05 6.821454878717383e+06 6.821610728030980e+06];
% ed50 = false;

% f = "05_15_collision avoidance_test\15_39_best";
% limits = [4.579181689817930e+05 4.580551459175702e+05 6.821486453806412e+06 6.821623430742187e+06];
% limits_fig = [4.579280826496579e+05 4.580715513792954e+05 6.821473206046199e+06 6.821616674775839e+06];
% ed50 = false;

% f = "05_15_collision avoidance_test\08_49_rundtur_rundt_k_veldig_bra";
% limits = [4.580005517000641e+05 4.581564010136597e+05 6.821699089349379e+06 6.821854938662973e+06];
% limits = [4.579652741900912e+05 4.581986958019651e+05 6.821668113380324e+06 6.821852215264527e+06];
% speed = 10;
% ed50 = true;

% f = "05_19 Collision Test\13_14_fra_cage";
% limits = [4.579158540214643e+05 4.580717033350599e+05 6.821454878717383e+06 6.821610728030980e+06];
% limits_fig = [4.579158540214643e+05 4.580717033350599e+05 6.821454878717383e+06 6.821610728030980e+06];
% ed50 = false;

% f = "05_19 Collision Test\13_17_tilbake_til_cage";
% limits = [4.579158540214643e+05 4.580717033350599e+05 6.821454878717383e+06 6.821610728030980e+06];
% ed50 = false;

% f = "05_19 Collision Test\13_27 dï¿½rlig los";
% limits = [4.579158540214643e+05 4.580717033350599e+05 6.821454878717383e+06 6.821610728030980e+06];
% ed50 = false;

% f = "05_19 Collision Test\13_32 dï¿½rlig lps";
% limits = [4.579158540214643e+05 4.580717033350599e+05 6.821454878717383e+06 6.821610728030980e+06];
% ed50 = false;

% f = "05_19 Collision Test\13_33 kollisjonskurs";
% limits = [4.579158540214643e+05 4.580717033350599e+05 6.821454878717383e+06 6.821610728030980e+06];
% ed50 = false;

% f = "06_13_sim\13_20_sim_trap_situation";
% f = "06_13_sim\13_48_sim_obj_detection_test";
% f = "06_13_sim\14_09_obj_det";
% f = "06_13_sim\14_21_obj_det"; % Ok Fjern slutt
% f = "06_13_sim\14_52_col_avoidance_surr";
% f = "06_13_sim\15_00_col_avoidance_new_trap";
% f = "06_13_sim\15_56_god_test"; % Ok få nye ruter
% f = "06_13_sim\16_04_high_obs_density_problems";
% f = "06_13_sim\16_11_coll_ok"; % Sync issues? manual adjustment of margins
% limits = [-40.821777474560236 64.036636840311730 -46.148215055791674 36.554631073196060];
% limits_fig = [-40.821777474560236 64.036636840311730 -46.148215055791674 36.554631073196060];
% sim = true;

% f = "06_13_sim\16_17_coll_almost_good";
% limits_fig = [-41.182786950953520 45.217884023825015 -26.119248821718493 42.025796511614920];
% sim = true;

f = "06_16_sim\12_47_god";
limits_fig = [-41.182786950953520 45.217884023825015 -26.119248821718493 42.025796511614920];
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
    if startsWith(listing(i).name, 'paths')
        load(char(listing(i).folder + folder_char + listing(i).name));
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
        legend([leg2, leg, l], [leg_text2, leg_text, l_t ],'AutoUpdate','off');
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