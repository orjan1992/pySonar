function [fig, leg_text, leg] = draw_rov(pos, range_scale, fig, color)
    if ~exist('fig', 'var')
        fig = figure();
    else
        figure(fig);
    end
    if ~exist('color', 'var')
        color = [0, 1, .7];
    end
    t = [0:pi/50:pi 0];
    boxx = sin(t)*range_scale;
    boxy = cos(t)*range_scale;
    [n, e] = vehicle2ned(boxx, boxy, pos(1), pos(2), pos(3));
    %     plot(e, n, 'Color', color);
    leg(2) = plot(e, n, 'Color', color);
    %     plot(e(3:4), n(3:4), '*');
    %     plot(pos(2), pos(1), 'r*')

    rovx = [1 1.5 1 -1 -1];
    rovy = [-1 0 1 1 -1];
    [n, e] = vehicle2ned(rovx, rovy, pos(1), pos(2), pos(3));
    leg(1) = fill(e, n, 'b');
    leg_text = {'ROV', 'Sonar Field of View'};
end

