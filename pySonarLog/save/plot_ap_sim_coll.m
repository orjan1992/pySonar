clear, close all
date = [2018, 6, 13];
file_path = '06_13_sim\16_17_coll_almost_good';
listing = dir(file_path);
limits = [datetime(2018, 6, 13, 14, 17, 57), datetime(2018, 6, 13, 16, 20, 49)];
ne_lim = [-29.816457700555016 21.740505222351775 -12.3846 15.0926];
folder_char = "/"; % Linux
% folder_char = "\"; % windows
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
% path = data(i).path(1:14, :);
pos_t = los_time(start_ind:end_ind);
wp_change = wp_change(wp_change >= limits(1));
wp_change = wp_change(wp_change <= limits(2));

%% North east
[ne_plot, h, h_text] = plotMapSim(figure());
hold on
t = 0:0.0001:2*pi;
x = cos(t);
y = sin(t);
% path_l = plot(path(:, 2), path(:, 1), '-*', 'LineWidth', w);
for i = 1:length(data)
    plot(data(i).path(:, 2), data(i).path(:, 1), '-*', 'LineWidth', w);
end
% try
%     j = 1;
% %     roa_l = plot(roa(j)*x+path(j, 2), roa(j)*y+path(j, 1), 'LineWidth', w);
%     roa_l = plot(NaN, NaN);
%     for j = 2:size(path, 1)
%         plot(roa(j-1)*x+path(j, 2),roa(j-1)*y+path(j, 1), 'Color', get(roa_l, 'Color'), 'LineWidth', w);
%     end
% end
plot(NaN, NaN);
pos_l = plot(east, north, 'LineWidth', w);
axis(ne_lim);
% legend([h, pos_l, path_l,roa_l], {h_text, 'Position', 'Path', 'ROA'}, 'Location', 'south');

text_pos = [-26.9734117561012 -10.4996100924362 0;-27.1089263157812 0.455159757471064 0;-26.7372267243273 3.83578714590409 0;-26.2496447916180 6.25641828003491 0;-24.4482723642963 7.34818323876406 0;-20.6062500000000 5.99204639099963 0;-11.7759746920288 11.5868529685017 0;-9.30268153920632 12.8421256969063 0;-3.91574498689872 11.4503782931113 0;0.861191236444112 9.62623608188860 0;11.6711113923321 5.26855442505525 0;14.2592432998893 3.42695530911103 0;17.3695723349617 -0.699390551394785 0;17.4533657416682 -7.23048339492897 0];
for i = 1:size(path, 1)
    t_pos(i) = text(text_pos(i, 1), text_pos(i, 2), string(i));
end

% for i = 1:length(wp_change)
%     ind = find(pos_t == wp_change(i));
%     if ~isempty(ind)
% %         ind = ind(round(length(ind)/2));
% %         ind = ind(end);
% %         text(east(ind(1)), north(ind(1)), string(i));
% %         text(east(ind(end)), north(ind(end)), string(i));
% %         t(i) = text(east(ind(round(length(ind)/2))), north(ind(round(length(ind)/2))), string(i));
%     end
% end
% 
% %% Heading
% heading_plot = figure();
% plot(los_time, unwrap(yaw)*180/pi, 'LineWidth', w);
% hold on;
% stairs(los_time, unwrap(chi)*180/pi, 'LineWidth', w);
% legend('Real', 'Los output', 'Location', 'southeast');
% xlim(limits);
% xlabel('Time');
% ylabel('Yaw - [Deg]');
% datetick('x','HH:MM:ss','keeplimits','keepticks')
% for i = 1:length(wp_change)
%     ind = find(los_time == wp_change(i));
%     ind = ind(round(length(ind)/2));
%     text(los_time(ind), unwrap(yaw(ind))*180/pi, string(i));
% end
% grid on
% 
% %% Heading error
% heading_error_plot = figure();
% dyaw = (unwrap(chi')-unwrap(yaw))*180/pi;
% plot(los_time, dyaw, 'LineWidth', w);
% % legend('Real', 'Los output', 'Location', 'southeast');
% xlim(limits);
% xlabel('Time');
% ylabel('Heading Error - [Deg]');
% datetick('x','HH:MM:ss','keeplimits','keepticks')
% for i = 1:length(wp_change)
%     ind = find(los_time == wp_change(i));
%     ind = ind(round(length(ind)/2));
%     text(los_time(ind), dyaw(ind), string(i));
% end
% grid on
% 
%% delta_plot
delta_plot = figure();
plot(los_time, cross_track, 'LineWidth', w);
xlim(limits);
xlabel('Time');
ylabel('Cross track error - [m]');
datetick('x','HH:MM:ss','keeplimits','keepticks')
% for i = 1:length(wp_change)
%     ind = find(los_time == wp_change(i));
%     ind = ind(round(length(ind)/2));
%     text(los_time(ind), cross_track(ind), string(i));
% end
grid on;

% u
% fake_data = load(strcat(listing(1).folder, folder_char, 'data_sim.mat'));
% f_time = limits(1) + seconds(fake_data.t);

u_plot = figure();
hold on;
plot(los_time, surge, 'LineWidth', w);
stairs(los_time, surge_ref, 'LineWidth', w);
% plot(f_time, fake_data.v);
% stairs(f_time, fake_data.vel_ref);
xlim(limits);
xlabel('Time');
ylabel('Surge velocity - [m/s]');
datetick('x','HH:MM','keeplimits','keepticks')
grid on;

% %% along_track
% along_fig = figure();
% plot(los_time, delta)
% for i = 1:length(wp_change)
%     ind = find(los_time == wp_change(i));
%     ind = ind(round(length(ind)/2));
%     text(los_time(ind), delta(ind), string(i));
% end
% 
% save_fig(ne_plot, [file_path '\ne'], true);
% save_fig(heading_plot, [file_path '\heading'], true);
% save_fig(heading_error_plot, [file_path '\heading_error'], true);
% save_fig(delta_plot, [file_path '\cross_track'], true);
% save_fig(u_plot, [file_path '\surge'], true);