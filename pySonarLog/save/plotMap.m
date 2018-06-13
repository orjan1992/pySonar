clear
% close all
figure();
hold on;
red = [0.635000000000000 0.0780000000000000 0.184000000000000];
blue = [0 0.447000000000000 0.741000000000000];
orange = [0.850000000000000 0.325000000000000 0.0980000000000000];
yellow = [0.929000000000000 0.694000000000000 0.125000000000000];
purple = [0.494000000000000 0.184000000000000 0.556000000000000];
green = [0.466000000000000 0.674000000000000 0.188000000000000];
lightblue = [0.301000000000000 0.745000000000000 0.933000000000000];
lightorange = [255, 178, 102]./255;

west_ellipse = [35 34 112]/255;
north_square = [100, 97, 100]/255;
north_ellipse = [8 93 68]/255;
north_circle = [105 55 174]/255;
south_square = [156 97 15]/255;
south_rect = [125 32 22]/255;
west_rect = [125 32 56]/255;
xt_color = [124 119 5]/255;



square = [-.5, -.5; .5, -.5; .5, .5; -.5, .5; -.5, -.5];

cube = ones(5, 1).*[-0.20295, -12.06624] + [2 2].*square;
cube2 = ones(5, 1).*[31.14198, -12.06624] + [2 20].*square;
cube3 = ones(5, 1).*[5.99079, -40.69405] + [40, 2].*square;
cube4 = ones(5, 1).*[-0.20295, 21.19193] + [20, 2].*square;

% cube5 = ones(5, 1).*[10.16381, -3.10354] + [3, 3].*square;


t=-pi:0.001:pi;

ellipse1 = [10.80412, 13.37201] + [2*cos(t'), 2*sin(t')];
ellipse2 = [-5.35571, 6.09444] + [2*2.488*cos(t'), 2*sin(t')];
ellipse3 = [-21.47458, -3.72886] + [2*cos(t'), 16*sin(t')];

xt0 = [10.16381-.7, -3.10354+.7] + [0.55*cos(t'), 0.55*sin(t')];
xt1 = [10.16381+.7, -3.10354+.7] + [0.55*cos(t'), 0.55*sin(t')];
xt2 = [10.16381, -3.10354-.6] + [0.55*cos(t'), 0.55*sin(t')];
xt3 = ones(5, 1).*[10.16381, -4.4] + [3, 0.1].*square;
xt_main = [10.16381, -3.10354] + [3, 3].*square;

h(1) = fill(cube(:, 1), cube(:, 2), south_square);
h(2) = fill(cube2(:, 1), cube2(:, 2), west_rect);
h(3) = fill(cube3(:, 1), cube3(:, 2), south_rect);
h(4) = fill(cube4(:, 1), cube4(:, 2), north_square);
h(5) = fill(ellipse1(:, 1), ellipse1(:, 2), north_circle);
h(6) = fill(ellipse2(:, 1), ellipse2(:, 2), north_ellipse);
h(7) = fill(ellipse3(:, 1), ellipse3(:, 2), west_ellipse);

h(8) = fill(xt_main(:, 1), xt_main(:, 2), lightorange);
h(9) = fill(xt0(:, 1), xt0(:, 2), xt_color);
h(10) = fill(xt1(:, 1), xt1(:, 2), xt_color);
h(11) = fill(xt2(:, 1), xt2(:, 2), xt_color);
h(12) = fill(xt3(:, 1), xt3(:, 2), xt_color);

% plot(cube(:, 1), cube(:, 2), 'b', cube2(:, 1), cube2(:, 2), 'b', ...
%     cube3(:, 1), cube3(:, 2), 'b', cube4(:, 1), cube4(:, 2), 'b', ...
%     ellipse1(:, 1), ellipse1(:, 2), 'b', ellipse2(:, 1), ellipse2(:, 2), 'b', ...
%     ellipse3(:, 1), ellipse3(:, 2), 'b', 'Linewidth', 1.5)
hold on
% plot(xt0(:, 1), xt0(:, 2), 'b', xt1(:, 1), xt1(:, 2), 'b', ...
%     xt2(:, 1), xt2(:, 2), 'b', xt3(:, 1), xt3(:, 2), 'b', 'Linewidth', 1.5)
axis equal, grid on
xlabel('East [m]');
ylabel('North [m]');
% saveas(gcf, '../map', 'epsc');
