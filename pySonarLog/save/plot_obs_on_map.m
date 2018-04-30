function [fig, leg] = plot_obs_on_map(file, color, fig)
    if ~exist('file', 'var')
        file = 'obstacles20180430-085413.mat';
    end
    if ~exist('color', 'var')
        color = 'r';
    end
    if ~exist('fig', 'var')
        fig = figure();
    else
        figure(fig);
    end
    load(file)
    hold on;
    s = size(obs);
    for i = 1:s(2)
        s_obs = size(obs{i});
        cur_obs = double(reshape(obs{i}, [s_obs(1), s_obs(3)]));
        [n, e] = grid2ned(cur_obs(:, 1), cur_obs(:, 2), range_scale, pos(1), pos(2), pos(3));
        leg = fill(e, n, color);
    end
    boxx = [-1, -1, 1, 1, -1]*range_scale;
    boxy = [-1, 1, 1, -1, -1]*range_scale;
    [n, e] = vehicle2ned(boxx, boxy, pos(1), pos(2), pos(3))
    plot(e, n, color);
    plot(pos(2), pos(1), 'r*')