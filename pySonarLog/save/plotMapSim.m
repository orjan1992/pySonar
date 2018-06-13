function [fig, h, h_text] = plotMapSim(fig)
if ~exist('fig', 'var')
    fig = figure();
else
    figure(fig);
end
hold on;

% red = [0.635000000000000 0.0780000000000000 0.184000000000000];
% blue = [0 0.447000000000000 0.741000000000000];
% orange = [0.850000000000000 0.325000000000000 0.0980000000000000];
% yellow = [0.929000000000000 0.694000000000000 0.125000000000000];
% purple = [0.494000000000000 0.184000000000000 0.556000000000000];
green = [0.466000000000000 0.674000000000000 0.188000000000000];
% lightblue = [0.301000000000000 0.745000000000000 0.933000000000000];
% lightorange = [255, 178, 102]./255;
% 
% west_ellipse = [35 34 112]/255;
% north_square = [100, 97, 100]/255;
% north_ellipse = [8 93 68]/255;
% north_circle = [105 55 174]/255;
% south_square = [156 97 15]/255;
% south_rect = [125 32 22]/255;
% west_rect = [125 32 56]/255;
% xt_color = [124 119 5]/255;



square = [-.5, -.5; .5, -.5; .5, .5; -.5, .5; -.5, -.5];


shapes{1} = ones(5, 1).*[-0.20295, -12.06624] + [2 2].*square;
shapes{end+1} = ones(5, 1).*[31.14198, -12.06624] + [2 20].*square;
shapes{end+1} = ones(5, 1).*[5.99079, -40.69405] + [40, 2].*square;
shapes{end+1} = ones(5, 1).*[-0.20295, 21.19193] + [20, 2].*square;

% cube5 = ones(5, 1).*[10.16381, -3.10354] + [3, 3].*square;


t=-pi:0.0001:pi;

% shapes{end+1} = [10.80412, 13.37201] + [2*cos(t'), 2*sin(t')];
% shapes{end+1} = [-5.35571, 6.09444] + [2*2.488*cos(t'), 2*sin(t')];
% shapes{end+1} = [-21.47458, -3.72886] + [2*cos(t'), 16*sin(t')];
shapes{end+1} = [10.80412, 13.37201] + [1*cos(t'), 1*sin(t')];
shapes{end+1} = [-5.35571, 6.09444] + [1*2.488*cos(t'), 1*sin(t')];
shapes{end+1} = [-21.47458, -3.72886] + [1*cos(t'), 8*sin(t')];

shapes{end+1} = [10.16381-.7, -3.10354+.7] + [0.55*cos(t'), 0.55*sin(t')];
shapes{end+1} = [10.16381+.7, -3.10354+.7] + [0.55*cos(t'), 0.55*sin(t')];
shapes{end+1} = [10.16381, -3.10354-.6] + [0.55*cos(t'), 0.55*sin(t')];
shapes{end+1} = ones(5, 1).*[10.16381, -4.4] + [3, 0.1].*square;
shapes{end+1} = [10.16381, -3.10354] + [3, 3].*square;

for i=1:length(shapes)
    h = plot([shapes{i}(:, 1); shapes{i}(1, 1)], [shapes{i}(:, 2); shapes{i}(1, 2)], 'LineWidth', 2, 'Color', green);
end
h_text = 'Installation Line';
hold on
axis equal, grid on
xlabel('East [m]');
ylabel('North [m]');
end

