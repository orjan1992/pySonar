clear, close all
date = [2018, 6, 13];
path = '06_13_sim\14_21_obj_det';
listing = dir(path);
limits = [datetime(2018, 6, 13, 14, 17, 57), datetime(2018, 6, 13, 14, 20, 50)];
ne_lim = [-29.816457700555016 21.740505222351775 -19.868693475388444 20.794782120258994];
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
surge = [];
wp_change = [];
for i = 1:length(data)
    chi = [chi data(i).chi];
    surge_set = [surge_set data(i).surge];
    los_time = [los_time; los_t{i}];
    pos = [pos; data(i).pos];
    cross_track = [cross_track; data(i).cross_track];
    slow_down_vel = [slow_down_vel; data(i).slow_down_vel];
    surge = [surge; data(i).surge];
    if ~isempty(data(i).wp_change)
        for j = 1:size(data(i).wp_change, 2)
            wp_change = [wp_change; datetime(strcat(t0,data(i).wp_change(j, :)), 'InputFormat','yyyyMMddHH:mm:ss')];
        end
    end
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
[ne_plot, h, h_text] = plotMapSim(figure());
hold on
t = 0:0.0001:2*pi;
x = cos(t);
y = sin(t);
for i = 1:length(data)
    path_l = plot(data(i).path(1:15, 2), data(i).path(1:15, 1), '-*', 'LineWidth', w);
    try
        j = 1;
        roa_l = plot(data(i).roa(j)*x+data(i).path(j, 2), data(i).roa(j)*y+data(i).path(j, 1), 'LineWidth', w);
        for j = 2:size(data(i).path, 1)
            plot(data(i).roa(j)*x+data(i).path(j, 2), data(i).roa(j)*y+data(i).path(j, 1), 'Color', get(roa_l, 'Color'), 'LineWidth', w);
        end
    catch
        continue
    end
end
plot(NaN, NaN);
pos_l = plot(east, north, 'LineWidth', w);
axis(ne_lim);
legend([h, pos_l, path_l,roa_l], {h_text, 'Position', 'Path', 'ROA'});


%% Heading
heading_plot = figure();
plot(los_time, unwrap(yaw)*180/pi, los_time, unwrap(chi)*180/pi);
legend('Real', 'Los output');
xlim(limits);
xlabel('Time');
ylabel('Yaw - [Deg]');
datetick('x','HH:MM','keeplimits','keepticks')

%% delta_plot
delta_plot = figure();
plot(los_time, cross_track, 'LineWidth', w);
xlim(limits);
xlabel('Time');
ylabel('Cross track error - [m]');
datetick('x','HH:MM','keeplimits','keepticks')

%% u
u_plot = figure();
plot(los_time, pos(:, 5), 'LineWidth', w);
xlim(limits);
xlabel('Time');
ylabel('Surge velocity - [m/s]');
datetick('x','HH:MM','keeplimits','keepticks')