close all
head = pi/4;
co = cos(head);
si = sin(head);
lat = 0;
long = 0;

square = [-.5, -.5; .5, -.5; .5, .5; -.5, .5; -.5, -.5];

cube = ones(5, 1).*[-0.20295, -12.06624] + [1 1].*square;
cube2 = ones(5, 1).*[31.14198, -12.06624] + [1 10].*square;
cube3 = ones(5, 1).*[5.99079, -40.69405] + [20, 1].*square;
cube4 = ones(5, 1).*[-0.20295, 21.19193] + [10, 1].*square;

% cube5 = ones(5, 1).*[10.16381, -3.10354] + [3, 3].*square;


t=-pi:0.01:pi;

ellipse1 = [10.80412, 13.37201] + [1*cos(t'), 1*sin(t')];
ellipse2 = [-5.35571, 6.09444] + [2.488*cos(t'), 1*sin(t')];
ellipse3 = [-21.47458, -3.72886] + [1*cos(t'), 8*sin(t')];

xt0 = [10.16381-.7, -3.10354+.7] + [0.55*cos(t'), 0.55*sin(t')];
xt1 = [10.16381+.7, -3.10354+.7] + [0.55*cos(t'), 0.55*sin(t')];
xt2 = [10.16381, -3.10354-.6] + [0.55*cos(t'), 0.55*sin(t')];
xt3 = ones(5, 1).*[10.16381, -4.4] + [3, 0.1].*square;

h = plot(cube(:, 1), cube(:, 2), 'b', cube2(:, 1), cube2(:, 2), 'b', ...
    cube3(:, 1), cube3(:, 2), 'b', cube4(:, 1), cube4(:, 2), 'b', ...
    ellipse1(:, 1), ellipse1(:, 2), 'b', ellipse2(:, 1), ellipse2(:, 2), 'b', ...
    ellipse3(:, 1), ellipse3(:, 2), 'b', ...
    xt0(:, 1), xt0(:, 2), 'b', xt1(:, 1), xt1(:, 2), 'b', ...
    xt2(:, 1), xt2(:, 2), 'b', xt3(:, 1), xt3(:, 2), 'b', 'Linewidth', 1.5)
rotate(h, [0 0 1], 25)