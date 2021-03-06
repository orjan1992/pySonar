clear, close all
date = [2018, 05, 15];
listing = dir();
% data = struct();
n = 1;
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
        ap_log = importdata(strcat(listing(i).folder, '\', listing(i).name));
        ap_time_txt = strsplit(listing(i).name, {'-', ' ', '.'});
        ap_time_txt = ap_time_txt(1:4);
        
    end
end
%% extract data
ap_log = ap_log.data;
north = ap_log(:, 1);
east = ap_log(:, 2);
depth = ap_log(:, 3);
yaw = ap_log(:, 6);
yaw_ref = ap_log(:, 12);
surge_vel = ap_log(:, 13);
surge_vel_ref = ap_log(:, 19);
hms = ap_log(:, 27:29);
ap_time = datetime([ones(length(hms), 1)*date(1), ones(length(hms), 1)*date(2), ones(length(hms), 1)*date(3), hms]);
% dt = diff([ap_time(end), datetime([date(1), date(2), date(3), 16, 21, 00])]);
dt = minutes(-1) + seconds(2);
ap_time = ap_time + dt;


%% combine los data
chi = [];
surge_set = [];
los_time = [];
for i = 1:length(data)
    chi = [chi data(i).chi];
    surge_set = [surge_set data(i).surge];
    los_time = [los_time; los_t{i}];
end
%% North east
ne_plot = figure();
plot(north, east)
hold on
for i = 1:length(data)
    plot(data(i).path(:, 1), data(i).path(:, 2), '--')
end

%% Heading
heading_plot = figure();
plot(ap_time, unwrap(yaw)*180/pi, ap_time, unwrap(yaw_ref)*180/pi);
hold on;
stairs(los_time, unwrap(chi)*180/pi);
legend('Real', 'Reference signal', 'Los output');
xlim([los_time(1)-seconds(10), los_time(end)+seconds(10)])
xlabel('Time');
ylabel('Yaw - [Deg]');

%% Surge
surge_plot = figure();
plot(ap_time, surge_vel, ap_time, surge_vel_ref);
hold on;
stairs(los_time, surge_set);
ylabel('Surge speed [m/s]');
xlabel('Time');
xlim([los_time(1)-seconds(10), los_time(end)+seconds(10)])
