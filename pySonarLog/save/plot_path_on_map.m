function [ fig ] = plot_path_on_map( fig, file, color )
%PLOT_PATH_ON_MAP Summary of this function goes here
%   Detailed explanation goes here
figure(fig);
hold on;
load(file);
if exist('color', 'var')
    plot(path(:, 2), path(:, 1), '--', 'Color', color);
else
    plot(path(:, 2), path(:, 1), '--');
end

end

