clear, close all
save = true;
file_path = '\';
date = [2018, 05, 15];
ap_dt = (hours(15) + minutes(5) + seconds(41)) - (hours(15) + minutes(06) + seconds(45)) + seconds(6);
% limits = [datetime([2018, 5, 15, 15, 14, 00]), datetime([2018, 5, 15, 15, 18, 10])];
limits = [datetime([2018, 5, 15, 15, 14, 34]), datetime([2018, 5, 15, 15, 18, 10])];
y_limits = [105 200];%[97.1501 160];%211.0971];
un_wrap = false;

% wp_shift = [datetime([2018, 5, 15, 15, 14, 44]), datetime([2018, 5, 15, 15, 14, 49]), ...
%     datetime([2018, 5, 15, 15, 14, 57]), ... %datetime([2018, 5, 15, 15, 14, 58]), ...
%     datetime([2018, 5, 15, 15, 17, 15]), datetime([2018, 5, 15, 15, 17, 40]), ...
%     datetime([2018, 5, 15, 15, 18, 3])];
wp_shift = [datetime([2018, 5, 15, 15, 14, 40]), datetime([2018, 5, 15, 15, 14, 52]), ...
    datetime([2018, 5, 15, 15, 14, 59]), datetime([2018, 5, 15, 15, 17, 16]), ...
    datetime([2018, 5, 15, 15, 17, 42]), datetime([2018, 5, 15, 15, 18, 05])];


w = 1.5;
f_size = 15;
listing = dir();
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
delta = [];
cross_track = [];
for i = 1:length(data)
    chi = [chi data(i).chi];
    delta = [delta data(i).delta];
    cross_track = [cross_track data(i).cross_track];
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

%% find shift inds
for i = 1:length(wp_shift)
    shift_ind(i) = find(ap_time == wp_shift(i));
end
shift_ind = shift_ind-1;
for i = 1:length(wp_shift)
    tmp = find(los_time == wp_shift(i));
    shift_ind_los(i) = tmp(1);
end
%% s and e plot
se_fig = figure();
hold on;
plot(los_time, cross_track, 'LineWidth', w);
xlim(limits);
grid on
xlabel('Time');
ylabel('Cross-track error - [m]');
for i =1:length(wp_shift)
    text(wp_shift(i), cross_track(shift_ind_los(i)), string(i+1), 'FontSize', f_size);
end
%% North east
ne_plot = figure();
[ne_plot, leg_text2, leg2] = plot_map(ne_plot);
start_ind = find(ap_time>limits(1),1);
end_ind = find(ap_time>limits(2),1)-1;

pos = plot(east(start_ind:end_ind), north(start_ind:end_ind), 'LineWidth', w);
hold on

p = plot(data(3).path(1:7, 2), data(3).path(1:7, 1), '--+', 'LineWidth', w);

axis equal
axis([4.579629628540017e+05 4.580386788809545e+05 6.821520562572032e+06 6.821580280535221e+06]);

grid on
% text(457965.5408, 6821567.115, "1", 'FontSize', f_size*.8, 'HorizontalAlignment', 'Right');
% text(east(shift_ind(1)), north(shift_ind(1)), string(2), 'FontSize', f_size*.8, 'HorizontalAlignment', 'Left');
% for i =2:length(wp_shift)
%     text(east(shift_ind(i)), north(shift_ind(i)), string(i+1), 'FontSize', f_size*.8, 'HorizontalAlignment', 'Right');
% end
% t = [6821567.9755, 457965.5408; 6821564.3615, 457966.5734; 6821561.4358, 457963.3035; 6821562.2963, 457969.1549; 6821538.2026, 458025.9472; 6821531.4908, 458034.5521; 6821522.7138, 458033.5195];

t = 0:0.001:2*pi;
x = cos(t);
y = sin(t);
px = data(3).path(1:7, 2);
py = data(3).path(1:7, 1);
f = true;
for i = 2:length(px)
	if sqrt((px(i)-px(i-1))^2 + (py(i)-py(i-1))^2)/2 < 2
        r = sqrt((px(i)-px(i-1))^2 + (py(i)-py(i-1))^2)/2;
    else
        r = 2;
	end
    if f
        roa = plot(px(i)+r*x, py(i)+r*y, 'LineWidth', w);
        f = false;
    end
    plot(px(i)+r*x, py(i)+r*y, 'Color', get(roa, 'Color'), 'LineWidth', w);
end
legend([leg2, pos, p, roa], [leg_text2, 'Position', 'Path', 'ROA']);
% t = [6821566.3865, 457965.5398; 6821564.3532, 457965.9164; 6821562.4706, 457963.9584; 6821560.2114, 457967.046; 6821537.2429, 458025.1072; 6821531.1431, 458033.7674; 6821522.7841, 458035.7254]; 
% for i =1:length(t)
%     text(t(i, 2), t(i, 1), string(i), 'FontSize', f_size*.8);
% end
% set(p, 'MarkerSize', 0.000000001);
% text(east(wp_shift(i)), north(shift_ind(i)), string(i+1), 'FontSize', f_size);
for i =1:length(wp_shift)
    text(east(shift_ind(i)), north(shift_ind(i)), string(i+1), 'FontSize', f_size);
%     plot(east(shift_ind(i)), north(shift_ind(i)), '*r');
end

XTickLabel = get(gca,'XTick');
XTickLabel = XTickLabel(1):10:XTickLabel(end);
set(gca,'XTick', XTickLabel);
set(gca,'XTickLabel',num2str(XTickLabel'))
xtickangle(45)
YTickLabel = get(gca,'YTick');
YTickLabel = YTickLabel(1):10:YTickLabel(end);
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
legend('Los output', 'Estimated', 'Reference signal', 'Location', 'north');
xlim(limits)
ylim(y_limits);
xlabel('Time');
ylabel('Yaw - [Deg]');
grid on



for i =1:length(wp_shift)
    if i == 1
        text(wp_shift(i), yaw_ref(shift_ind(i))-20, string(i+1), 'FontSize', f_size);
    else
        text(wp_shift(i), yaw_ref(shift_ind(i)), string(i+1), 'FontSize', f_size);
    end
end


%% Surge
surge_fig = figure();
hold on;
stairs(los_time, surge_set, 'Linewidth', w);
plot(ap_time, surge_vel, '--', 'Linewidth', w);
plot(ap_time, surge_vel_ref, '-.', 'Linewidth', w);
ylim([0.05 0.55]);
grid on;
xlim([limits(1), datetime([2018, 5, 15, 15, 16, 0])]);
xlabel('Time');
ylabel('Surge velocity - [m/s]');
legend('Los output', 'Estimated', 'Reference signal', 'Location', 'southeast');
for i =1:length(wp_shift)
    text(wp_shift(i), surge_vel_ref(shift_ind(i)), string(i+1), 'FontSize', f_size);
end
if save
    save_fig(heading_plot, [file_path 'heading'], true);
    save_fig(surge_fig, [file_path 'surge'], true);
    save_fig(ne_plot, [file_path 'ne'], true);
    save_fig(se_fig, [file_path 'cross_track'], true);

    figure(ne_plot);
    axis([4.579622406267729e+05 4.579738914022426e+05 6.821558804645933e+06 6.821567993725296e+06]);
    set(gca, 'XTickMode', 'auto', 'XTickLabelMode', 'auto')
    set(gca, 'YTickMode', 'auto', 'YTickLabelMode', 'auto')
    XTickLabel = get(gca,'XTick');
    XTickLabel = XTickLabel(1):2:XTickLabel(end);
    set(gca,'XTick', XTickLabel);
    set(gca,'XTickLabel',num2str(XTickLabel'))
    xtickangle(45)
    YTickLabel = get(gca,'YTick');
    YTickLabel = YTickLabel(1):2:YTickLabel(end);
    set(gca,'YTick', YTickLabel);
    set(gca,'YTickLabel',num2str(YTickLabel'))
    ytickangle(45);
    axis manual;
    save_fig(ne_plot, [file_path 'ne_close_up'], false);
end