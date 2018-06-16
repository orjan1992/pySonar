clear, close all

file_path = 'los_path_test_simulator';
date = [2018, 04, 27];
ap_dt = (minutes(44) + seconds(44)) - (minutes(44) + seconds(54)) + seconds(7);
% limits = [datetime([2018, 5, 15, 15, 14, 00]), datetime([2018, 5, 15, 15, 18, 10])];
limits = [datetime([2018, 04, 27, 14, 43, 19]), datetime([2018, 04, 27, 14, 45, 07])];
y_limits = [-20.6557, 142.3203];
ne_lim = [-57.6381 -36.4039 15.7052 25.1874];
un_wrap = false;

% wp_shift = [datetime([2018, 04, 27, 14, 43, 23]),datetime([2018, 04, 27, 14, 43, 24]), ...
%     datetime([2018, 04, 27, 14, 44, 2]),datetime([2018, 04, 27, 14, 44, 8]), ...
%     datetime([2018, 04, 27, 14, 44, 9]),datetime([2018, 04, 27, 14, 44, 32]), ...
%     datetime([2018, 04, 27, 14, 44, 37]),datetime([2018, 04, 27, 14, 44, 43]), ...
%     datetime([2018, 04, 27, 14, 44, 44]),datetime([2018, 04, 27, 14, 44, 54]), ...
%     datetime([2018, 04, 27, 14, 45, 3]),];



w = 1.5;
f_size = 15;
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
ap_time = ap_time + ap_dt;

%% combine los data
chi = [];
surge_set = [];
los_time = [];
% delta = [];
% cross_track = [];
for i = 1:length(data)
    chi = [chi data(i).chi];
%     delta = [delta data(i).delta];
%     cross_track = [cross_track data(i).cross_track];
    surge_set = [surge_set data(i).surge];
    los_time = [los_time; los_t{i}];
end

if un_wrap
    chi = unwrap(chi);
    yaw = unwrap(yaw);
    yaw_ref = unwrap(yaw_ref);
end
chi = wrapTo180(chi*180/pi);
yaw = wrapTo180(yaw*180/pi);
yaw_ref = wrapTo180(yaw_ref*180/pi);

%% find shift inds
shift_ind = [26 46 60 64 79 89 95 101];
wp_shift = ap_time(shift_ind);
% for i = 1:length(wp_shift)
%     shift_ind(i) = find(ap_time == wp_shift(i));
% end
% for i = 1:length(wp_shift)
%     tmp = find(los_time == wp_shift(i));
%     shift_ind_los(i) = tmp(1);
% end

%% North east
ne_plot = figure();
% [ne_plot, leg_text2, leg2] = plot_map(ne_plot);
start_ind = find(ap_time>limits(1),1);
end_ind = find(ap_time>limits(2),1)-1;

pos = plot(east(start_ind:end_ind), north(start_ind:end_ind), 'LineWidth', w);
% plot(east, north);
hold on
px = [data(2).path(2:10, 2); -38.1728];
py = [data(2).path(2:10, 1); 36.258];
p = plot(px, py, '--o', 'LineWidth', w);

t = 0:0.001:2*pi;
x = cos(t);
y = sin(t);
for i = 2:length(px)
     if sqrt((px(i)-px(i-1))^2 + (py(i)-py(i-1))^2) < 2
         r = sqrt((px(i)-px(i-1))^2 + (py(i)-py(i-1))^2)/2;
     else
         r = 2;
     end
    plot(px(i)+r*x, py(i)+r*y, 'b');
end


axis equal
% axis([-58.841002028406180 -36.142440717455560 16.437431724538015 34.340006693981340]);
axis(ne_lim);
legend([pos, p], 'Position', 'Path', 'Location', 'northwest');
grid on
% text(457965.5408, 6821567.115, "1", 'FontSize', f_size*.8, 'HorizontalAlignment', 'Right');
% text(east(shift_ind(1)), north(shift_ind(1)), string(2), 'FontSize', f_size*.8, 'HorizontalAlignment', 'Left');
for i =1:length(wp_shift)
    text(east(shift_ind(i)), north(shift_ind(i)), string(i+1), 'FontSize', f_size*.8, 'HorizontalAlignment', 'Right');
end
% t = [6821567.9755, 457965.5408; 6821564.3615, 457966.5734; 6821561.4358, 457963.3035; 6821562.2963, 457969.1549; 6821538.2026, 458025.9472; 6821531.4908, 458034.5521; 6821522.7138, 458033.5195];
% t = [6821566.3865, 457965.5398; 6821564.3532, 457965.9164; 6821562.4706, 457963.9584; 6821560.2114, 457967.046; 6821537.2429, 458025.1072; 6821531.1431, 458033.7674; 6821522.7841, 458035.7254]; 
% for i =1:length(t)
%     text(t(i, 2), t(i, 1), string(i), 'FontSize', f_size*.8);
% end
xlabel('East');
ylabel('North');
XTickLabel = get(gca,'XTick');
% XTickLabel = XTickLabel(1):10:XTickLabel(end);
set(gca,'XTick', XTickLabel);
set(gca,'XTickLabel',num2str(XTickLabel'))
xtickangle(45)
YTickLabel = get(gca,'YTick');
% YTickLabel = YTickLabel(1):10:YTickLabel(end);
set(gca,'YTick', YTickLabel);
set(gca,'YTickLabel',num2str(YTickLabel'))
ytickangle(45);
axis manual;

%% Heading
heading_plot = figure();
hold on;
% plot(ap_time, unwrap(yaw)*180/pi, ap_time, unwrap(yaw_ref)*180/pi, los_time, unwrap(chi)*180/pi);
stairs(los_time, chi, 'Linewidth', w);
plot(ap_time, yaw, '--', 'Linewidth', w);
plot(ap_time, yaw_ref, '-.', 'Linewidth', w);
legend('Los output', 'Estimate', 'Reference signal', 'Location', 'south');
xlim(limits)
ylim(y_limits);
xlabel('Time');
ylabel('Yaw - [Deg]');
grid on
datetick('x','HH:MM:ss','keeplimits','keepticks')



for i =1:length(wp_shift)
    text(wp_shift(i), yaw_ref(shift_ind(i)), string(i+1), 'FontSize', f_size);
end


%% Surge
surge_fig = figure();
hold on;
stairs(los_time, surge_set, 'Linewidth', w);
plot(ap_time, surge_vel, '--', 'Linewidth', w);
plot(ap_time, surge_vel_ref, '-.', 'Linewidth', w);
% ylim([0.05 0.55]);
grid on;
xlim(limits);
xlabel('Time');
ylabel('Surge velocity - [m/s]');
legend('Los output', 'Estimate', 'Reference signal', 'Location', 'southeast');
for i =1:length(wp_shift)
    text(wp_shift(i), surge_vel_ref(shift_ind(i)), string(i+1), 'FontSize', f_size);
end
datetick('x','HH:MM:ss','keeplimits','keepticks')
save_fig(heading_plot, [file_path '\heading'], true);
save_fig(surge_fig, [file_path '\surge'], true);
save_fig(ne_plot, [file_path '\ne'], true);
% save_fig(se_fig, [file_path '\cross_track'], true);
