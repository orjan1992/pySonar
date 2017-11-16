clear, close all
fig = figure();

for i = 1:27
	plotGridBinary(fig, i, 2)
    plotMap(fig)
    xlabel('East [m]');
    ylabel('North [m]');
    pause(0.8)
    hold off;
end