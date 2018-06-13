function [fig, leg_text, leg, all_obj] = plot_obs_on_map(file, color, fig, draw_outline, oppacity, rov_oppacity, draw_path, error)
    if ~exist('color', 'var')
        color = [0, 1, .7];
    end
    if ~exist('fig', 'var')
        fig = figure();
    else
        figure(fig);
    end
    load(file)
    if exist('error', 'var')
        pos(1:2) = pos(1:2) - error;
    else
        error = [0 0];
    end
    if ~exist('draw_outline', 'var')
        draw_outline = true;
    end
    if ~exist('oppacity', 'var')
        oppacity = 1;
    end
    if ~exist('rov_oppacity', 'var')
        rov_oppacity = 1;
    end
    if exist('draw_path', 'var')
        if draw_path
            plot_paths = true;
        else
            plot_paths = false;
        end
    else
        plot_paths = true;
    end
    if ~exist('obs', 'var')
        obs = obstacles;
    else
        plot_paths = false;
    end
    
    range_scale = double(range_scale);
    all_obj = [];
    hold on;
    s = size(obs);
    if iscell(obs)
        for i = 1:s(2)
            s_obs = size(obs{i});
            cur_obs = double(reshape(obs{i}, [s_obs(1), s_obs(3)]));
            [n, e] = grid2ned(cur_obs(:, 1), cur_obs(:, 2), range_scale, pos(1), pos(2), pos(3));
%             n = n - error(1);
%             e = e - error(2);
            leg(1) = patch(e, n, color,'FaceAlpha', oppacity, 'EdgeColor', 'none');
            all_obj(end+1) = leg(1);
        end
    else
        for i = 1:s(1)
            s_obs = size(obs);
            cur_obs = double(reshape(obs(i, :, :, :), [s_obs(2), s_obs(4)]));
            [n, e] = grid2ned(cur_obs(:, 1), cur_obs(:, 2), range_scale, pos(1), pos(2), pos(3));
%             n = n - error(1);
%             e = e - error(2);
            leg(1) = patch(e, n, color,'FaceAlpha', oppacity, 'EdgeColor', 'none');
            all_obj(end+1) = leg(1);
        end
    end
    
    leg_text = {'Obstacles'};
    if draw_outline
        t = [0:pi/50:pi 0];
        boxx = sin(t)*range_scale;
        boxy = cos(t)*range_scale;
        [n, e] = vehicle2ned(boxx, boxy, pos(1), pos(2), pos(3));
        leg(end+1) = plot(e, n, 'Color', color);
        leg_text{end+1} = 'Sonar Field of View';
        all_obj(end+1) = leg(end);
    end
    
    if rov_oppacity ~= 0
        rovx = [1 1.5 1 -1 -1];
        rovy = [-.75 0 .75 .75 -.75];
        [n, e] = vehicle2ned(rovx, rovy, pos(1), pos(2), pos(3));
        leg(end+1) = patch(e, n, 'b','FaceAlpha',rov_oppacity);
        leg_text{end+1} = 'ROV';
        all_obj(end+1) = leg(end);
    end
    if plot_paths
        if ~isempty(old_wps)
            if iscell(old_wps)
                old_wps = cell2mat(old_wps);
            end
            leg(end+1) = plot(old_wps(:, 2), old_wps(:, 1), '--m');
            all_obj(end+1) = leg(end);
            leg_text = [leg_text {'Old Path'}];
        end
        if ~isempty(new_wps)
            if iscell(new_wps)
                new_wps = cell2mat(new_wps);
            end
            leg(end+1) = plot(new_wps(:, 2), new_wps(:, 1), '-m');
            all_obj(end+1) = leg(end);
            leg_text = [leg_text {'New Path'}];
            
            tmp = voronoi_vertices(voronoi_end_wp+1, :);
            [end_north, end_east] = grid2ned(tmp(1), tmp(2), range_scale, pos(1), pos(2), pos(3));
            leg(end+1) = plot(end_east, end_north, '*m');
            all_obj(end+1) = leg(end);
            leg_text = [leg_text {'End of optimization'}];
            
        end
    end
    grid on