clear, close all
save_figs = true;
load('data.mat')
orig = orig(1:801, :);
thresh = thresh(1:801, :);

%% normal
grid_fig = figure();
imagesc([-30, 30], [30, 0], orig);
colormap(jet)
colorbar()
axis equal
grid on
set(gca,'Ydir','Normal')
ylim([0, 30]);

if save_figs
    set(gcf, 'PaperUnits', 'normalized')
    set(gcf, 'PaperPosition', [0 0 1 1])
    print -depsc occ_grid
end

%% Binary
bin_fig = figure();
imagesc([-30, 30], [30, 0], thresh);
colormap(jet)
colorbar()
axis equal
grid on
set(gca,'Ydir','Normal')
ylim([0, 30]);

if save_figs
    set(gcf, 'PaperUnits', 'normalized')
    set(gcf, 'PaperPosition', [0 0 1 1])
    print -depsc binary
end

%% Contour plot
contour_orig_fig = figure();
imagesc([-30, 30], [30, 0], thresh);
colormap(jet)
colorbar()
axis equal
grid on
set(gca,'Ydir','Normal')
ylim([0, 30]);
hold on
s = size(contour_orig);
for i = 1:s(2)
    s_obs = size(contour_orig{i});
    cur_obs = double(reshape(contour_orig{i}, [s_obs(1), s_obs(3)]));
    [x, y] = grid2vehicle(cur_obs(:, 1), cur_obs(:, 2), 30);
    lin = plot([y; y(1)], [x; x(1)], 'g', 'LineWidth', 2);
end
legend(lin, 'Detected Obstacles');
if save_figs
    set(gcf, 'PaperUnits', 'normalized')
    set(gcf, 'PaperPosition', [0 0 1 1])
    print -depsc first_contour
end

%% Contour filtered plot
contour_filter_fig = figure();
imagesc([-30, 30], [30,0], thresh);
colormap(jet)
colorbar()
axis equal
grid on
set(gca,'Ydir','Normal')
hold on
ylim([0, 30]);
s = size(contour_filter);
for i = 1:s(2)
    s_obs = size(contour_filter{i});
    cur_obs = double(reshape(contour_filter{i}, [s_obs(1), s_obs(3)]));
    [x, y] = grid2vehicle(cur_obs(:, 1), cur_obs(:, 2), 30);
    lin = plot([y; y(1)], [x; x(1)], 'g', 'LineWidth', 2);
end
legend(lin, 'Filtered Obstacles');

if save_figs
    set(gcf, 'PaperUnits', 'normalized')
    set(gcf, 'PaperPosition', [0 0 1 1])
    print -depsc filtered_contours
end

%% Contour dilated plot
contour_dilate_fig = figure();
imagesc([-30, 30], [30,0], thresh);
colormap(jet)
colorbar()
axis equal
grid on
set(gca,'Ydir','Normal')
hold on
ylim([0, 30]);
s = size(contours_dilated);
for i = 1:s(2)
    s_obs = size(contours_dilated{i});
    cur_obs = double(reshape(contours_dilated{i}, [s_obs(1), s_obs(3)]));
    [x, y] = grid2vehicle(cur_obs(:, 1), cur_obs(:, 2), 30);
    lin = plot([y; y(1)], [x; x(1)], 'g', 'LineWidth', 2);
end
legend(lin, 'Dilated Contours');

if save_figs
    set(gcf, 'PaperUnits', 'normalized')
    set(gcf, 'PaperPosition', [0 0 1 1])
    print -depsc dilated_contours
end

%% Occ
contour_dilate_fig = figure(grid_fig);
hold on
s = size(contours_convex);
for i = 1:s(2)
    s_obs = size(contours_convex{i});
    cur_obs = double(reshape(contours_convex{i}, [s_obs(1), s_obs(3)]));
    [x, y] = grid2vehicle(cur_obs(:, 1), cur_obs(:, 2), 30);
    lin = plot([y; y(1)], [x; x(1)], 'r', 'LineWidth', 2);
end
legend(lin, 'Convex contours');

if save_figs
    set(gcf, 'PaperUnits', 'normalized')
    set(gcf, 'PaperPosition', [0 0 1 1])
    print -depsc convex_contours
end