clear, close all
date = [2018, 04, 27];
path = '06_13_sim/14_21_obj_det';
listing = dir(path);
limits = [datetime(2018, 6, 13, 14, 17, 57), datetime(2018, 6, 13, 14, 20, 50)];

folder_char = "/"; % Linux
% folder_char = "\"; % windows

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
for i = 1:length(data)
    chi = [chi data(i).chi];
    surge_set = [surge_set data(i).surge];
    los_time = [los_time; los_t{i}];
    pos = [pos; data(i).pos];
end

yaw = pos(:, 4);

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


%% North east
ne_plot = plotMapSim(figure());
plot(east, north)
hold on
for i = 1:length(data)
    plot(data(i).path(:, 2), data(i).path(:, 1), '--')
end

%% Heading
heading_plot = figure();
plot(los_time, unwrap(yaw)*180/pi, los_time, unwrap(chi)*180/pi);
legend('Real', 'Los output');
xlim(limits);
xlabel('Time');
ylabel('Yaw - [Deg]');
