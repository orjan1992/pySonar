clear, %close all
fig = figure();

for i = 1:43
	[grid_plot, rov_plot] = plotGridBinary(fig, i, 2);
    obstacle= plotMap(fig);
    xlabel('East [m]');
    ylabel('North [m]');
%     legend([grid_plot, rov_plot, obstacle], {'Occupancy grid', 'ROV', 'Obstacles'});
    pause(0.8)
    hold off;
end