clear
d = 'New Folder';
listing = dir(d);
position = load('pos');
f_ind = [];
% t_time = [];
for i = 1:length(listing)
    if endsWith(listing(i).name, '.mat')
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

for i = 1:length(f_time)
    load([listing(f_ind(i)).folder, '\', listing(f_ind(i)).name]);
    [~, ind] = min(abs(position.time - f_time(i)));
    pos = position.pos(ind, :);
    
    save([listing(f_ind(i)).folder, '\new\', listing(f_ind(i)).name], 'grid', 'obs', 'pos', 'range_scale');
end    