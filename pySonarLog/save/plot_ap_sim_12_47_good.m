clear, close all
date = [2018, 6, 16];
file_path = '06_16_sim\12_47_god';
listing = dir(file_path);
limits = [datetime(2018, 6, 16, 12, 42, 36), datetime(2018, 6, 16, 12, 46, 12)];
ne_lim = [-26.3283 26.8715 -17.3747 26.7854];
% folder_char = "/"; % Linux
folder_char = "\"; % windows
w = 1.5;


n = 1;
for i = 1:length(listing)
    if startsWith(listing(i).name, 'Los_log')
        data(n) = load(strcat(listing(i).folder, folder_char, listing(i).name));
        fname = strsplit(listing(i).name, {'-', '_'});
%         t0 = datetime(fname(3), 'InputFormat','yyyyMMdd');
        t0 = fname(3);
        los_t{n} = datetime(strcat(t0,data(n).time(:, :)), 'InputFormat','yyyyMMddHH:mm:ss');
        n = n+1;
    end
end

%% combine los data
chi = [];
surge_set = [];
los_time = [];
pos = [];
cross_track = [];
slow_down_vel = [];
surge_ref = [];
wp_change = [];
roa = [];
delta = [];
for i = 1:length(data)
    chi = [chi data(i).chi];
    surge_set = [surge_set data(i).surge];
    los_time = [los_time; los_t{i}];
    pos = [pos; data(i).pos];
    cross_track = [cross_track; data(i).cross_track'];
    slow_down_vel = [slow_down_vel; data(i).slow_down_vel'];
    surge_ref = [surge_ref; data(i).surge'];
    roa = [roa; data(i).roa'];
    delta = [delta; data(i).delta'];
    if ~isempty(data(i).wp_change)
        for j = 1:size(data(i).wp_change, 1)
            wp_change = [wp_change; datetime(strcat(t0,data(i).wp_change(j, :)), 'InputFormat','yyyyMMddHH:mm:ss')];
        end
    end
end

yaw = pos(:, 4);
surge = pos(:, 5);

start_ind = find(los_time > limits(1));
start_ind = start_ind(1);
end_ind = find(los_time > limits(2));
if ~isempty(end_ind)
    end_ind = end_ind(1) - 1;
else
    end_ind = length(los_time);
end

north = pos(start_ind:end_ind, 1);
east = pos(start_ind:end_ind, 2);
psi_reference = pos(:, 8);
surge_reference = pos(:, 9);
% path = data(i).path(1:14, :);
pos_t = los_time(start_ind:end_ind);
wp_change = wp_change(wp_change >= limits(1));
wp_change = wp_change(wp_change <= limits(2));

%% North east
[ne_plot, h, h_text] = plotMapSim(figure());
hold on
plot(pos(:, 2), pos(:, 1));
for i = 1:length(data)
    plot(data(i).path(:, 2), data(i).path(:, 1), '-.');
end
% t = 0:0.0001:2*pi;
% x = cos(t);
% y = sin(t);
% path_l = plot(path(:, 2), path(:, 1), '-*', 'LineWidth', w);
% try
%     j = 1;
% %     roa_l = plot(roa(j)*x+path(j, 2), roa(j)*y+path(j, 1), 'LineWidth', w);
%     roa_l = plot(NaN, NaN);
%     for j = 2:size(path, 1)
%         plot(roa(j-1)*x+path(j, 2),roa(j-1)*y+path(j, 1), 'Color', get(roa_l, 'Color'), 'LineWidth', w);
%     end
%     plot(roa(end)*x+east(end),roa(end)*y+north(end), 'Color', get(roa_l, 'Color'), 'LineWidth', w);
% end
% plot(NaN, NaN);
% pos_l = plot(east, north, 'LineWidth', w);
% axis(ne_lim);
% legend([h, pos_l, path_l,roa_l], {h_text, 'Position', 'Path', 'ROA'});
% 
% text_pos = [23.7704360753718 -14.0180483938946 0;5.31401522058871 -9.75584161449466 0;2.51368014241656 -8.17911591999955 0;-1.30295278992389 -4.22078499323898 0;-1.56930784440332 -0.144092264309989 0;3.73156044348836 6.89783541249534 0;1.85330000000000 7.86980000000000 0;2.49838432123946 11.9998242250626 0;0.547486092422999 11.1674184161172 0;-1.43426037842109 13.4484618918994 0;-3.74718242631806 15.5344602356491 0;-7.19908397298430 16.4081657604429 0;-13.1500189237189 17.9310538802344 0;-16.2792816162109 20.3849031220016 0;-22.4253260719944 24.4722230266516 0];
% for i = 1:size(text_pos, 1)
%     t_pos(i) = text(text_pos(i, 1), text_pos(i, 2), string(i));
% end
% 
% % for i = 1:length(wp_change)
% %     ind = find(pos_t == wp_change(i));
% %     if ~isempty(ind)
% % %         ind = ind(round(length(ind)/2));
% % %         ind = ind(end);
% % %         text(east(ind(1)), north(ind(1)), string(i));
% % %         text(east(ind(end)), north(ind(end)), string(i));
% %         t_pos(i) = text(east(ind(round(length(ind)/2))), north(ind(round(length(ind)/2))), string(i));
% %     else
% %         t_pos(i) = text(east(1), north(1), string(i));
% %     end
% % end

%% Heading
heading_plot = figure();
plot(los_time, unwrap(yaw)*180/pi, 'LineWidth', w);
hold on;
stairs(los_time, unwrap(chi)*180/pi, '--', 'LineWidth', w);
plot(los_time, psi_reference*180/pi, '-.', 'LineWidth', w);
legend('Real', 'Setpoint', 'Reference');%, 'Location', 'southeast');
xlim(limits);
xlabel('Time');
ylabel('Yaw - [Deg]');
datetick('x','HH:MM:ss','keeplimits','keepticks')
for i = 1:length(wp_change)
    ind = find(los_time == wp_change(i));
    ind = ind(round(length(ind)/2));
    text(los_time(ind), unwrap(yaw(ind))*180/pi, string(i));
end
grid on

%% Heading error
heading_error_plot = figure();
hold on;
dyaw = (unwrap(yaw) - unwrap(chi'))*180/pi;
dyaw_ref = (unwrap(yaw)-unwrap(psi_reference))*180/pi;
% plot(los_time, dyaw, 'LineWidth', w);
plot(los_time, dyaw_ref, 'LineWidth', w);
% legend('Real', 'Los output', 'Location', 'southeast');
xlim(limits);
xlabel('Time');
ylabel('Heading Error - [Deg]');
datetick('x','HH:MM:ss','keeplimits','keepticks')
he_text_pos = [0.534067140296979 -2.08395669539198 0;0.534611068441714 8.01110409701433 0;0.534678219406042 10.4873722113520 0;0.534770811998635 31.8323615919169 0;0.534870338368322 6.41244085688963 0;0.535074031404677 -6.83879057399325 0;0.535138835552142 -8.07089935206289 0;0.535192118962280 -19.0212423101824 0;0.535268497183820 -18.5287243794642 0;0.535291591995221 -24.2748342573505 0;0.535370370370371 -11.4086363316824 0;0.535423547107015 6.05021684202796 0;0.535608732292200 18.8793663262500 0;0.535793917477385 6.40637966252139 0;0.536009131251067 -1.45305051983686 0];
for i = 1:length(wp_change)
    ind = find(los_time == wp_change(i));
    ind = ind(round(length(ind)/2));
    he_text(i) = text(los_time(ind), dyaw_ref(ind), string(i));
%     he_text(i) = text(he_text_pos(i, 1), he_text_pos(i, 2), string(i));
end
grid on
% 
%% delta_plot
delta_plot = figure();
plot(los_time, cross_track, 'LineWidth', w);
xlim(limits);
xlabel('Time');
ylabel('Cross track error - [m]');
datetick('x','HH:MM:ss','keeplimits','keepticks')
% % surge_text_pos = [0.534067140296979 -0.494982444542562 0;0.534615708738693 0.333401008464366 0;0.534668938812084 0.387304483280196 0;0.534803294077488 1.15285030133698 0;0.534898180150196 0.0888539499183908 0;0.535106513483530 -0.278299123855510 0;0.535148116146100 -0.133850352644187 0;0.535182838368322 0.0893929918236122 0;0.535259216589862 0.0964977827588538 0;0.535291591995221 0.297601953586182 0;0.535333247994538 0.618486898428138 0;0.535423547107015 0.827623604083627 0;0.535599451698242 0.941473350831912 0;0.535775356289469 0.263317121111110 0;0.536004490954088 -0.0738740673741517 0];
% for i = 1:length(wp_change)
%     ind = find(los_time == wp_change(i));
%     ind = ind(round(length(ind)/2));
% %     d_text(i) = text(los_time(ind), cross_track(ind), string(i));
%     d_text(i) = text(surge_text_pos(i, 1), surge_text_pos(i, 2), string(i));
% end
grid on;

%% u

u_plot = figure();
hold on;
plot(los_time, surge, 'LineWidth', w);
stairs(los_time, surge_ref, '-.', 'LineWidth', w);
stairs(los_time, surge_reference, '-.', 'LineWidth', w);
xlim(limits);
% xlim([datetime(2018, 6,16, 12,50,25), datetime(2018, 6, 16, 12, 51, 13)]);
xlabel('Time');
ylabel('Surge velocity - [m/s]');
legend('Velocity', 'Setpoint', 'Reference', 'Location', 'southeast');
datetick('x','HH:MM:ss','keeplimits','keepticks')
grid on;
% text_pos_y = [0.000426231214078143 0.400001049041748 0.400022685527802 0.400063812732697 0.400511175394058 0.390816183479465 0.410417954897394 0.318973223043948 0.416091194201489 0.335310161113739 0.379873217733539 0.412091243145417 0.400002986192703 0.400006055831909 0.399998813867569];
% for i = 1:length(wp_change)
%     ind = find(los_time == wp_change(i));
%     ind = ind(round(length(ind)/2));
%     surge_text(i) = text(los_time(ind), text_pos_y(i), string(i));
% end
% %% along_track
% along_fig = figure();
% plot(los_time, delta)
% for i = 1:length(wp_change)
%     ind = find(los_time == wp_change(i));
%     ind = ind(round(length(ind)/2));
%     text(los_time(ind), delta(ind), string(i));
% end

save_fig(ne_plot, [file_path '\ne'], true);
save_fig(heading_plot, [file_path '\heading'], true);
save_fig(heading_error_plot, [file_path '\heading_error'], true);
save_fig(delta_plot, [file_path '\cross_track'], true);
save_fig(u_plot, [file_path '\surge'], true);