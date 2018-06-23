clear, close all
fig = figure();
% axis([-25, 41, -22, 30]);
% axis([-4.575204218508427 14.788934589630431 -3.495092177629891 11.777591495240907]);
% axis([-20 45 -20 34.068893928689960]);
% daspect([1 1 1])
% axis([-24.141388313857020 44.412219530289725 -18.104830989848004 35.964062938841950])
axis([-20.223751447758850 36.262680796868890 -14.361959745586134 30.189436008644492]);
xlabel('East');
ylabel('North');
save_fig(fig, 'coordinate_conversion', true)
% axis equal
% axis manual
grid on
hold on;

pos = [5, 6, 10*pi/180];
% pos = [0, 0, 0];

%% ROV
rovx = [1 1.5 1 -1 -1];
rovy = [-.75 0 .75 .75 -.75];
[n, e] = vehicle2ned(rovx, rovy, pos(1), pos(2), pos(3));
rov = patch(e, n, 'b');

%% Vehicle x
rov_arrow_xx = [-2 10];
rov_arrow_xy = [0 0];
[n, e] = vehicle2ned(rov_arrow_xx, rov_arrow_xy, pos(1), pos(2), pos(3));
[x, y] = axxy2figxy(e, n);
a(1) = annotation('textarrow',x, y,'String','X_{VEH}');

%% vehicly y
rov_arrow_yx = [0 0];
rov_arrow_yy = [-2 10];
[n, e] = vehicle2ned(rov_arrow_yx, rov_arrow_yy, pos(1), pos(2), pos(3));
[x, y] = axxy2figxy(e, n);
a(2) = annotation('textarrow',x, y,'String','Y_{VEH}');

%% Grid
range = 16;
step = 4;
horizontal_x = -range:step:range;
c = [179, 179, 179]/255;
for i = 1:length(horizontal_x)
    [n1, e1] = vehicle2ned(horizontal_x(i), range, pos(1), pos(2), pos(3));
    [n2, e2] = vehicle2ned(horizontal_x(i), -range, pos(1), pos(2), pos(3));
    grid_lines = plot([e1 e2], [n1 n2], 'Color', c);
    
    [n1, e1] = vehicle2ned(range, horizontal_x(i), pos(1), pos(2), pos(3));
    [n2, e2] = vehicle2ned(-range, horizontal_x(i), pos(1), pos(2), pos(3));
    plot([e1 e2], [n1 n2], 'Color', c);
end

grid_arrow_xx = [range range];
grid_arrow_xy = [-range*1.2 range];
[n, e] = vehicle2ned(grid_arrow_xx, grid_arrow_xy, pos(1), pos(2), pos(3));
[x, y] = axxy2figxy(e, n);
a(3) = annotation('textarrow',x, y,'String','X_{g}');

grid_arrow_yx = [range*1.2 -range];
grid_arrow_yy = [-range -range];
[n, e] = vehicle2ned(grid_arrow_yx, grid_arrow_yy, pos(1), pos(2), pos(3));
[x, y] = axxy2figxy(e, n);
a(4) = annotation('textarrow',x, y,'String','Y_{g}');

%% NED
[x, y] = axxy2figxy([0 0], [-2.5 20]);
legend([rov, grid_lines], 'Vehicle', 'Grid lines');
a(5) = annotation('textarrow',x, y,'String','X_{N}');
[x, y] = axxy2figxy([-2.5 20], [0 0]);
legend([rov, grid_lines], 'Vehicle', 'Grid lines');
a(6) = annotation('textarrow',x, y,'String','Y_{N}');
save_fig(fig, 'coordinate_conversion', false)