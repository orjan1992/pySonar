clear, close all
cellsize = 0.2;
fig = figure();
hold on;
% grid on;
lWidth = 1.5;
fsize = 11;
cell_y_pos = cellsize*2.5;
cell_centre = [-cellsize, cell_y_pos];
cell_centre2 = [0, cell_y_pos];
cell_centre3 = [cellsize*2.5, cell_y_pos];
cone_pos = -pi/6;
coneL = 1;
bins = 5;
beamWidth = pi/8;
th = linspace(-beamWidth/2, beamWidth/2, 100);

strange_color = [0.635000000000000 0.0780000000000000 0.184000000000000];
blue = [0 0.447000000000000 0.741000000000000];
orange = [0.850000000000000 0.325000000000000 0.0980000000000000];
yellow = [0.929000000000000 0.694000000000000 0.125000000000000];
purple = [0.494000000000000 0.184000000000000 0.556000000000000];
green = [0.466000000000000 0.674000000000000 0.188000000000000];
lightblue = [0.301000000000000 0.745000000000000 0.933000000000000];

%% cell numbers
% c = 0;
% for i = .9:-.2:.1
%     for j = -.7:.2:.7
%         text(j, i, string(c));
%         c = c + 1;
%     end
% end

%% cells
gray = [217, 217, 217]/255;
for i = 1:-.2:0
    cells = plot([-.8, .8], [i, i], 'Color', gray);
end
for j = -.8:.2:.8
    plot([j j], [1 0], 'Color', gray);
end
        

%% selected cone
x = [0 .2 .2 .4 .4 .6 .6 .8 .8 .6 .6 .2 .2 0 0 NaN .6 .6 NaN .4 .4 NaN .2 .2 NaN .6 .2 NaN .6 .2 NaN .4 0 NaN .2 0]*-1;
y = [0 0 .2 .2 .4 .4 .6 .6 .8 .8 1 1 .6 .6 0 NaN .6 .8 NaN .2 1 NaN .2 .6 NaN .8 .8 NaN .6 .6 NaN .4 .4 NaN .2 .2];
select = plot(x, y, 'Color', lightblue, 'LineWidth', lWidth);

%% Cone
cone = [coneL*sin(-beamWidth/2+cone_pos), coneL*cos(-beamWidth/2+cone_pos); 0, 0; ...
    coneL*sin(beamWidth/2+cone_pos), coneL*cos(beamWidth/2+cone_pos)];
l_cone = plot(cone(:, 1), cone(:, 2), 'Color', orange, 'Linewidth', 1.5);
%% Bins
for i = linspace(0, coneL, bins)
    binx = (i)*sin(th+cone_pos);
    biny = (i)*cos(th+cone_pos);
    plot(binx, biny, 'Color', orange, 'Linewidth', lWidth)
end

%% Cell
cell = [cellsize/2, cellsize/2; -cellsize/2, cellsize/2; ...
    -cellsize/2, -cellsize/2; cellsize/2, -cellsize/2; ...
    cellsize/2, cellsize/2];

l_cell = plot(cell(:, 1)+cell_centre3(1), cell(:, 2)+cell_centre3(2), 'Color', blue, 'Linewidth', lWidth);
l_cell_centre = plot(cell_centre3(1), cell_centre3(2), '*', 'Color', blue);

% l_cell = plot(cell(:, 1)+cell_centre(1), cell(:, 2)+cell_centre(2), 'Color', blue, 'Linewidth', 1.5);
% l_cell_centre = plot(cell_centre(1), cell_centre(2), '*', 'Color', blue);
legend([l_cone, l_cell, l_cell_centre, select], {'Sonar Cone', 'Grid cell', 'Cell center', 'Overlapping cells'},'AutoUpdate','off');

% plot(cell(:, 1)+cell_centre2(1), cell(:, 2)+cell_centre2(2), 'Color', blue, 'Linewidth', lWidth);
% plot(cell_centre2(1), cell_centre2(2), '*', 'Color', blue);




%% Angles
yMax = ylim;
yMax = yMax(2);
h = plot([0, 0], [0, yMax], '--', 'Linewidth', lWidth, 'Color', purple);
c = h.Color();
alpha1 = atan2(-cellsize/2+cell_centre3(1), cellsize/2+cell_centre3(2));
alpha2 = atan2(cellsize/2+cell_centre3(1), -cellsize/2+cell_centre3(2));
alpha3 = atan2(cell_centre3(1), cell_centre3(2));
% SMALL ANGLE
r1 = 2.2*cellsize/3; %sqrt((-cellsize/2+cell_centre3(1))^2+(cellsize/2+cell_centre3(2))^2)/2;
%large angle
r2 = 1.6*1.4*cellsize;%sqrt((cellsize/2+cell_centre3(1))^2+(-cellsize/2+cell_centre3(2))^2)/2;
%angle
r3 = 2*(2/3)*cellsize;%sqrt((cell_centre3(1))^2+(cell_centre3(2))^2)/2;
th1 = linspace(0, alpha1, 100);
th2 = linspace(0, alpha2, 100);
th3 = linspace(0, alpha3, 100);
h = plot(r1*sin(th1), r1*cos(th1), '--', 'Linewidth', lWidth);
text(r1*sin(th1(50))+ cellsize/8, r1*cos(th1(50)), 'Small Angle', 'Color', h.Color(),'FontSize',fsize);
plot([0, -cellsize/2+cell_centre3(1)], [0, cellsize/2+cell_centre3(2)], '--', 'Linewidth', lWidth, 'Color', h.Color())
h = plot(r2*sin(th2), r2*cos(th2), '--', 'Linewidth', lWidth);
text(r2*sin(th2(60))+ cellsize/8, r2*cos(th2(60)), 'Large Angle', 'Color', h.Color(),'FontSize',fsize);
plot([0, cellsize/2+cell_centre3(1)], [0, -cellsize/2+cell_centre3(2)], '--', 'Linewidth', lWidth, 'Color', h.Color())
h = plot(r3*sin(th3), r3*cos(th3), '--', 'Linewidth', lWidth);
text(r3*sin(th3(60))+ cellsize/8, r3*cos(th3(60)), 'Angle', 'Color', h.Color(),'FontSize',fsize);
plot([0, cell_centre3(1)], [0, cell_centre3(2)], '--', 'Linewidth', lWidth, 'Color', h.Color())


grid on


axis equal
a = gca;
a.XTick = -.7:.2:.7;
a.YTick = .1:.2:.9;
a.GridLineStyle = 'None';
a.XTickLabel = num2str((0:7)');
a.YTickLabel = num2str((4:-1:0)');
xlabel('Cell column');
ylabel('Cell row');

% set(gca, 'xtick', [], 'ytick', [], 'xticklabel', [], 'yticklabel', [])
ylim([0 1.05])
xlim([-.8 .8]);
% saveas(gcf, '../grid_parameters', 'epsc');
save_fig(fig, 'grid_params', true);