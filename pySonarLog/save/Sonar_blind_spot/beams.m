clear, close all
width = 40*pi/180;
l = 15;
ROV_size = [2.5, 1.5, 1.5];
ROV_size(1) = ROV_size(2)*1278/786;
veh_y_0 = 2;
y_0 = 1.24;
x_0 = -.03;
y = atan(width/2)*l;
beam_x = [0 -l -l];
beam_y = [y_0+veh_y_0, y+y_0+veh_y_0, -y+y_0+veh_y_0];
bs_x = [0 -l 0];
bs_y = [y_0+veh_y_0, -y+y_0+veh_y_0, -y+y_0+veh_y_0];
fig = figure();
plot(NaN, NaN);
hold on
im = imread('ROV.PNG');
image([x_0, ROV_size(1)+x_0], [ROV_size(2)+veh_y_0, veh_y_0], im);
axis equal
grid on
beam = patch(beam_x, beam_y, [0, 102, 255]/255, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
bs = patch(bs_x, bs_y, [252, 27, 27]/255, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
legend([beam, bs], 'Sonar cone', 'Blind-spot');
xlim([-l ROV_size(1)]);
ylim([0 9]);
xlabel('X-axis [m]');
ylabel('Z-axis [m]');
XTickLabel = get(gca,'XTick');
set(gca,'XTick', XTickLabel);
set(gca,'XTickLabel',num2str(-XTickLabel'))
YTickLabel = get(gca,'YTick');
YTickLabel = YTickLabel(1):2:YTickLabel(end);
set(gca,'YTick', YTickLabel);
set(gca,'YTickLabel',num2str(YTickLabel'))
save_fig(fig, 'ROV_with_sonar_cone', true);
% pos = axxy2figxy([0, ROV_size(2), ROV_size(1), ROV_size(2)]);

% pos(4) = pos(3)*1.6667;

% pos(3:4) = 1;
% axes('pos',pos)
% image('ROV.png')
% hold on;
% plot(0, 0, '*r');