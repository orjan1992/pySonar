clear, close all
date = [2018, 04, 27];
path = 'los_path_test_simulator';
listing = dir(path);
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
        ap_time = strsplit(listing(i).name, '-');
        
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
plot(ap_time, unwrap(yaw), ap_time, unwrap(yaw_ref));
hold on;
% for i = 1:length(data)
%     plot(los_time{i}, unwrap(data(i).chi), 'r');
% end
plot(los_time, unwrap(chi));