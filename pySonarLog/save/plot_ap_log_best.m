clear, close all

% date = [2018, 04, 27];
% path = 'los_path_test_simulator';
% ap_dt = hours(0);
% limits = [datetime(date(1), date(2), date(3), 14, 43, 00),
% datetime(date(1), date(2), date(3), 14, 45, 30)];

file_path = '05_15_collision avoidance_test\15_39_best';
date = [2018, 05, 15];
% ap_dt = seconds(0);%(hours(15) + minutes(40) + seconds(18)) - (hours(15) + minutes(41) + seconds(16));% + seconds(3);
limits = [datetime([2018, 5, 15, 15, 39, 19]), datetime([2018, 5, 15, 15, 42, 48])];
y_limits = [86.1309 169.7340];
un_wrap = false;
% path_start_end = [1 5];

% file_path = '05_15_collision avoidance_test\15_01_path_test';
% date = [2018, 05, 15];
% ap_dt = (hours(15) + minutes(5) + seconds(41)) - (hours(15) + minutes(06) + seconds(45)) + seconds(6);
% % limits = [datetime([2018, 5, 15, 15, 39, 0]), datetime([2018, 5, 15, 15, 43, 0])];
% limits = [datetime([2018, 5, 15, 15, 3, 30]), datetime([2018, 5, 15, 15, 5, 23])];
% y_limits = [117.0091  225.9583];
% un_wrap = false;

w = 1.5;

listing = dir(file_path);
% data = struct();
n = 1;
k = 1;
for i = 1:length(listing)
    if startsWith(listing(i).name, 'Los_log')
        data(n) = load(strcat(listing(i).folder, '\', listing(i).name));
        fname = strsplit(listing(i).name, {'-', '_'});
%         t0 = datetime(fname(3), 'InputFormat','yyyyMMdd');
        t0 = fname(3);
        los_t{n} = datetime(strcat(t0,data(n).time(:, :)), 'InputFormat','yyyyMMddHH:mm:ss');
        
        n = n+1;
    end
    if endsWith(listing(i).name, '.csv')
        if ~exist('ap_log', 'var')
            ap_log = importdata(strcat(listing(i).folder, '\', listing(i).name));
            ap_log = ap_log.data;
            ap_time = strsplit(listing(i).name, '-');
        else
            tmp_log = importdata(strcat(listing(i).folder, '\', listing(i).name));
            tmp_log = tmp_log.data;
%             s_log = size(tmp_log);
%             tmp_time = strsplit(listing(i).name, '-');
%             s_time = size(tmp_time);
            
%             ap_log(end:end+s_log(1), end:end+s_log(2)) = tmp_log;
            ap_log = [ap_log; tmp_log];
%             tmp_time(end:end+s_time(1)) = tmp_time;
        end
    end
end
%% extract data
% ap_log = ap_log.data;
north = ap_log(:, 1);
east = ap_log(:, 2);
depth = ap_log(:, 3);
yaw = ap_log(:, 6);
yaw_ref = ap_log(:, 12);
surge_vel = ap_log(:, 13);
surge_vel_ref = ap_log(:, 19);
hms = ap_log(:, 27:29);
ap_time = datetime([ones(length(hms), 1)*date(1), ones(length(hms), 1)*date(2), ones(length(hms), 1)*date(3), hms]);

%% combine los data
chi = [];
surge_set = [];
los_time = [];
for i = 1:length(data)
    chi = [chi data(i).chi];
    surge_set = [surge_set data(i).surge];
    los_time = [los_time; los_t{i}];
end

if un_wrap
    chi = unwrap(chi);
    yaw = unwrap(yaw);
    yaw_ref = unwrap(yaw_ref);
end
chi = chi*180/pi;
yaw = yaw*180/pi;
yaw_ref = yaw_ref*180/pi;

%% combine ap data
ap_dt = los_time(end) - datetime([2018, 5, 15, 15, 43, 45]);
ap_time = ap_time + ap_dt;


%% North east
% ne_plot = figure();
% ne_plot = plot_map(ne_plot, 1.5);
% start_ind = find(ap_time>limits(1),1);
% end_ind = find(ap_time>limits(2),1)-1;
% 
% plot(east(start_ind:end_ind), north(start_ind:end_ind))
% hold on
% 
% los_t_start = NaT(1, length(los_t));
% for i=1:length(los_t)
%     los_t_start(i) = los_t{i}(1);
% end
% start_ind = find(los_t_start>limits(1),1)-1;
% end_ind = find(los_t_start>limits(2),1)-1;
% if isempty(end_ind)
%     end_ind = length(los_t);
% end
% 
% % for i = path_start_end(1):path_start_end(2)
% for i = start_ind:end_ind
% % for i = 1:length(data)
%     tmp = plot(data(i).path(:, 2), data(i).path(:, 1), '--o');
%     tmp = plot(data(i).path(1, 2), data(i).path(1, 1), '*', 'Color', get(tmp, 'Color'));
% end

% %% Fake path
% fake_path{1} = data(2).path(1:4, 1:2);
% fake_path{end+1} = data(3).path(1, 1:2);
% fake_path{end+1} = data(4).path(1, 1:2);
% fake_path{end+1} = data(5).path(1, 1:2);
% fake_path{end+1} = data(6).path(1, 1:2);
% % fake_path{end+1} = data(4).path(1:2, 1:2);
% % fake_path{end+1} = data(4).path(1:2, 1:2);
% for i = 1:length(fake_path)
%     plot(fake_path{i}(:, 2), fake_path{i}(:, 1), 'r-*');
% end
% axis([4.579679058454501e+05 4.579802799715712e+05 6.821548290499089e+06 6.821578070230909e+06]);
% ne_plot.WindowState = 'maximize';
% set (gcf, 'WindowButtonDownFcn', @mouseDownMat);
% legend

% %% Heading
heading_plot = figure();
hold on;
% plot(ap_time, unwrap(yaw)*180/pi, ap_time, unwrap(yaw_ref)*180/pi, los_time, unwrap(chi)*180/pi);
stairs(los_time, chi, 'Linewidth', w);
plot(ap_time, yaw, '--', 'Linewidth', w);
plot(ap_time, yaw_ref, '-.', 'Linewidth', w);
legend('Los output', 'Estimated', 'Reference signal');
xlim(limits)
ylim(y_limits);
xlabel('Time');
ylabel('Yaw - [Deg]');
grid on

%% Surge
surge_fig = figure();
hold on;
stairs(los_time, surge_set, 'Linewidth', w);
plot(ap_time, surge_vel, '--', 'Linewidth', w);
plot(ap_time, surge_vel_ref, '-.', 'Linewidth', w);
grid on;
xlim(limits);
xlabel('Time');
ylabel('Surge velocity - [m/s]');
legend('Los output', 'Estimated', 'Reference signal');

save_fig(heading_plot, [file_path '\heading'], true);
save_fig(surge_fig, [file_path '\surge'], true);
