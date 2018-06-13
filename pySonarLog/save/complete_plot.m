clear, close all
save_figs = true;

% f = "05_15_collision avoidance_test\15_30 bra test med collision avoidance";
% limits = [4.579158540214643e+05 4.580717033350599e+05 6.821454878717383e+06 6.821610728030980e+06];
% ed50 = false;

% f = "05_15_collision avoidance_test\15_39 best";
% limits = [4.579158540214643e+05 4.580717033350599e+05 6.821454878717383e+06 6.821610728030980e+06];
% ed50 = false;

% f = "05_15_collision avoidance_test\08_49_rundtur_rundt_k_veldig_bra";
% limits = [4.579268622579122e+05 4.580712316252046e+05 6.821494547608327e+06 6.821608413125433e+06];
% % diff = [216.4842, 91.1046];
% diff = [220.1061   91.5310];
% % limits = [4.579894395242393e+05 4.581981087395944e+05 6.821670191097749e+06 6.821836430365582e+06];
% % zoom_lim = [4.580254752487370e+05 4.581047304054406e+05 6.821759297541760e+06 6.821822437278015e+06];
% ed50 = false;
% f = "05_15_collision avoidance_test\08_49_rundtur_rundt_k_veldig_bra";
% limits = [4.580253437163720e+05 4.581670163741691e+05 6.821714061669944e+06 6.821826927187094e+06];
% % limits = [4.579894395242393e+05 4.581981087395944e+05 6.821670191097749e+06 6.821836430365582e+06];
% % zoom_lim = [4.580254752487370e+05 4.581047304054406e+05 6.821759297541760e+06 6.821822437278015e+06];
% ed50 = true;

f = "06_13_sim\14_21_obj_det";
limits = [-48.227803287309860 38.172867687468724 -25.105000255043210 43.040045078290230];
end_ind = 1800;
oppacity = 0.2;
sim = true;

% folder_char = "/"; % Linux
folder_char = "\"; % windows


% f = "05_15_collision avoidance_test\08_49_rundtur_new_pos";
% % limits = [4.578659906505865e+05 4.580887096161933e+05 6.821456367706740e+06 6.821632028310260e+06];
% limits = [4.579052453425336e+05 4.581009944333989e+05 6.821454067461536e+06 6.821608456663849e+06];
% zoom_lim = [4.579345073509259e+05 4.579957681600910e+05 6.821539475134597e+06 6.821587792127634e+06];
% ed50 = false;
% diff = [0, 0];
if ~exist('sim', 'var')
    sim = false;
end
if ~exist('diff', 'var')
    diff = [0, 0];
end
if ~exist('oppacity', 'var')
    oppacity = 0.1;
end
%% sort
listing = dir(char(f));
for i = 1:length(listing)
    if startsWith(listing(i).name, 'collision') || startsWith(listing(i).name, 'obstacles')
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
all_obj = [];
line = '-';
fig = gcf;
extend_frame = false;
% fig.WindowState = 'maximized';

l1 = [];
%% avoidance paths and obstacles
for i = 1:length(f_time)
    l1 = [];
    if startsWith(listing(f_ind(i)).name, 'collision')
        load(strcat(listing(f_ind(i)).folder, folder_char, listing(f_ind(i)).name));
        last_path = old_wps;
        new_path = new_wps;
    else
        hold on
%         if exist('last_path', 'var') && ~isempty(last_path)
%             l1(end+1) = plot(last_path(:, 2), last_path(:, 1), '--m');
%         end
%         if exist('new_path', 'var') && ~isempty(new_path)
%             l1(end+1) =plot(new_path(:, 2), new_path(:, 1), 'm');
%         end
    end
    [fig, leg_text, leg, all_obj] = plot_obs_on_map(strcat(listing(f_ind(i)).folder, ...
        folder_char, listing(f_ind(i)).name), 'r', fig, false, oppacity);
    
%     drawnow;
end
if sim
    [fig, leg2, leg_text2] = plotMapSim(fig);
else
    if ed50
        [fig, leg_text2, leg2] = plot_map_ed50(fig, 1.5);
    else
        [fig, leg_text2, leg2] = plot_map(fig);
    end
end
pos_mat(:, 1:2) = pos_mat(:, 1:2) -diff;
if exist('end_ind', 'var')
    l(1) = plot(pos_mat(1:end_ind, 2), pos_mat(1:end_ind, 1), 'b');
else
    l(1) = plot(pos_mat(:, 2), pos_mat(:, 1), 'b');
end
% pos_mat(end+1, :) = [NaN, NaN, NaN, NaN];
% l(1) = patch(pos_mat(:, 2), pos_mat(:, 1), pos_mat(:, 4), 'EdgeColor', 'interp', 'LineWidth', 3);
uistack(l(1), 'bottom');
% cb = colorbar;
% ylabel(cb, 'Altitude');
legend([leg, leg2, l], [leg_text, leg_text2, {'Position'}],'AutoUpdate','off');

axis equal;


axis(limits);
XTickLabel = get(gca,'XTick');
XTickLabel = XTickLabel(1):10:XTickLabel(end);
set(gca,'XTick', XTickLabel);
set(gca,'XTickLabel',num2str(XTickLabel'))
% xtickangle(45)
YTickLabel = get(gca,'YTick');
YTickLabel = YTickLabel(1):10:YTickLabel(end);
set(gca,'YTick', YTickLabel);
set(gca,'YTickLabel',num2str(YTickLabel'))
% ytickangle(45);


if save_figs
    save_fig(fig, char(f), true);
    if exist('zoom_lim', 'var')
        axis(zoom_lim);
        XTickLabel = get(gca,'XTick');
        set(gca,'XTickLabel',num2str(XTickLabel'))
        YTickLabel = get(gca,'YTick');
        set(gca,'YTickLabel',num2str(YTickLabel'))
        save_fig(fig, char(f + '_zoom'), false);
    end
end
% if save_figs
%     set(gcf, 'PaperUnits','centimeters');
%     set(gcf, 'Units','centimeters');
%     pos=get(gcf,'Position');
%     set(gcf, 'PaperSize', [pos(3) pos(4)]);
%     set(gcf, 'PaperPositionMode', 'manual');
%     set(gcf, 'PaperPosition',[0 0 pos(3) pos(4)]);
%     print(char(f), '-depsc');
% end
% 
% if exist('zoom_lim', 'var')
%     axis(zoom_lim);
%     XTickLabel = get(gca,'XTick');
%     set(gca,'XTickLabel',num2str(XTickLabel'))
%     YTickLabel = get(gca,'YTick');
%     set(gca,'YTickLabel',num2str(YTickLabel'))
%     if save_figs
%         set(gcf, 'PaperUnits','centimeters');
%         set(gcf, 'Units','centimeters');
%         pos=get(gcf,'Position');
%         set(gcf, 'PaperSize', [pos(3) pos(4)]);
%         set(gcf, 'PaperPositionMode', 'manual');
%         set(gcf, 'PaperPosition',[0 0 pos(3) pos(4)]);
%         print(char(f + '_zoom'), '-depsc');
%     end
% end