function [fig, leg_text, leg] = plot_obs_on_map(file, color, fig, new_pos)
    if ~exist('file', 'var')
        file = 'obstacles20180430-085413.mat';
    end
    if ~exist('color', 'var')
        color = [0, 1, .7];
    end
    if ~exist('fig', 'var')
        fig = figure();
    else
        figure(fig);
    end
    load(file)
    if exist('new_pos', 'var')
        pos(1:2) = new_pos;
    end
    hold on;
    s = size(obs);
    if iscell(obs)
        for i = 1:s(2)
            s_obs = size(obs{i});
            cur_obs = double(reshape(obs{i}, [s_obs(1), s_obs(3)]));
            [n, e] = grid2ned(cur_obs(:, 1), cur_obs(:, 2), range_scale, pos(1), pos(2), pos(3));
            leg(1) = fill(e, n, color);
        end
    else
        for i = 1:s(1)
            s_obs = size(obs);
            cur_obs = double(reshape(obs(i, :, :, :), [s_obs(2), s_obs(4)]));
            [n, e] = grid2ned(cur_obs(:, 1), cur_obs(:, 2), range_scale, pos(1), pos(2), pos(3));
            leg(1) = fill(e, n, color);
        end
    end
%     boxx = [-1, -1, 1, 1, -1]*range_scale;
%     boxy = [-1, 1, 1, -1, -1]*range_scale;
    t = [0:pi/50:pi 0];
    boxx = sin(t)*range_scale;
    boxy = cos(t)*range_scale;
    [n, e] = vehicle2ned(boxx, boxy, pos(1), pos(2), pos(3));
%     plot(e, n, 'Color', color);
    leg(3) = plot(e, n, 'Color', color);
%     plot(e(3:4), n(3:4), '*');
%     plot(pos(2), pos(1), 'r*')
    
    rovx = [1 1.5 1 -1 -1];
    rovy = [-1 0 1 1 -1];
    [n, e] = vehicle2ned(rovx, rovy, pos(1), pos(2), pos(3));
    leg(2) = fill(e, n, 'b');
    leg_text = {'Obstacles', 'ROV', 'Sonar Field of View'};
    
    XTickLabel = get(gca,'XTick');
    set(gca,'XTickLabel',num2str(XTickLabel'))
    YTickLabel = get(gca,'YTick');
    set(gca,'YTickLabel',num2str(YTickLabel'))
    
    grid on