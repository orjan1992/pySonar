clear, close all
fig = figure();
[fig, leg_text2, leg2] = plotMapSim(fig);
grid on
% grid minor
% hold on
t_pos = [-1.54611048787614 18.6107280747882;7.54611048787613 13.7051444931347;-1.27885836985100 6.93318697049372;-18.3158725679229 -3.01160625182588;13.3594730937774 -1.75815582822085;27.6775411042945 -9.83657510955301;2.42848156587788 -11.1960482033304;2.56210762489045 -36.2584703476482];

for i = 1:8
    t(i) = text(t_pos(i, 1), t_pos(i, 2), string(i));
end

% t_pos = zeros(8, 3);
% for i = 1:8
%     t_pos(i, :) = t(i).Position;
% end
save_fig(fig, 'sim_map', true);