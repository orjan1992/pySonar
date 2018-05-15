function f = plot_collision_info(fname)
% clear, close all
% fname = 'collision_info20180427-125811.mat';
    load(fname)
    if ~isempty(new_wps)
        new_wps = new_wps(:, 1:2);
    end
    old_wps = old_wps(:, 1:2);
    voronoi_ridge_vertices = voronoi_ridge_vertices + 1;
    voronoi_ridge_points = voronoi_ridge_points + 1;
    % voronoi_regions = voronoi_regions + 1;
    voronoi_point_region = voronoi_point_region + 1;
    pos = double(pos);
    range_scale = double(range_scale);


    f = figure();
    % obstacles
    plot(voronoi_points(:, 1), voronoi_points(:, 2), 'or')
    % ylim([min(voronoi_points(:, 2)), max(voronoi_points(:, 2))]);
    % xlim([min(voronoi_points(:, 1)), max(voronoi_points(:, 1))]);
%     axis manual;
    set(gca, 'YDir','reverse')
    hold on
    for i = 1:length(obstacles)
        if iscell(obstacles)
            obs = obstacles{i};
            s = size(obs);
            obs = reshape(obs, s(1), s(3));
        else
            obs = obstacles(i, :, :, :);
            s = size(obs);
            obs = reshape(obs, s(2), s(4));  
        end
        obs = [obs; obs(1, :)];
        plot(obs(:, 1), obs(:, 2), 'r')
    end

    % Vertices and ridges
    plot(voronoi_vertices(:, 1), voronoi_vertices(:, 2), 'bo');
    for i = 1:length(voronoi_ridge_vertices)
        if any(voronoi_ridge_vertices(i, :) == 0)
            continue
        end
        valid = false;
        if connection(voronoi_ridge_vertices(i, 1), voronoi_ridge_vertices(i, 2)) ~= 0
            valid = true;
        end
        v1 = voronoi_vertices(voronoi_ridge_vertices(i, 1), :);
        v2 = voronoi_vertices(voronoi_ridge_vertices(i, 2), :);
        if valid
            plot([v1(1), v2(1)], [v1(2), v2(2)], 'b', 'LineWidth', 1.5)
        else
            plot([v1(1), v2(1)], [v1(2), v2(2)], 'r')
        end
    end

    %% paths

    old_wps_grid = ned2grid(old_wps, pos, range_scale);
    l(1) = plot(old_wps_grid(:, 1), old_wps_grid(:, 2), 'k', 'LineWidth', 2);
    plot(old_wps_grid(:, 1), old_wps_grid(:, 2), 'ko');

    if ~isempty(new_wps)
        new_wps_grid = ned2grid(new_wps, pos, range_scale);
        l(2) = plot(new_wps_grid(:, 1), new_wps_grid(:, 2), 'g', 'LineWidth', 2);
        plot(new_wps_grid(:, 1), new_wps_grid(:, 2), 'og');
    end
    legend(l, {'old', 'new'});
    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset; 
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    ax.Position = [left bottom ax_width ax_height];
    %     save(f, strcat('png\',fname, '.png'));



