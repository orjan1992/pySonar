clear, close all
save_figs = true;
l_width = 1.5;
load('scanlines_new.mat')
n = 29;
data = double(scanlines{n});
data_interpx=1:0.01:length(data);
data_interpy = interp1(1:length(data), data, data_interpx, 'linear');
smooth = smooth(n, :);
peaks = smooth_peaks{n}+1;
valleys = smooth_valleys{n}+1;
final = final_peaks{n}+1;
threshold = 200;

% Normal way
fig(1) = figure();
hold on;
plot(data, 'LineWidth', l_width)

threshold = min(max(threshold * ad_span(n) / 255.0 + ad_low(n), 0), 255);
mean_val = mean(data);
max_val = double(max(data));
threshold = max(max_val - (max_val - mean_val) / 8, threshold);
data_cp = data_interpy;
data_cp(data_cp < threshold) = NaN;
th = plot(data_interpx, data_cp, 'LineWidth', l_width); 
plot([0, length(data)], [threshold, threshold], '--', 'LineWidth', l_width, 'Color', get(th, 'Color'));


xlabel('Bin');
ylabel('Value');
lgd = legend('Data', 'Data over threshold', 'Threshold');
lgd.Location = 'north';

fig(2) = figure();
plot(smooth, 'LineWidth', l_width)
hold on
p = plot(peaks, smooth(peaks), '*');
plot(valleys, smooth(valleys), '*')
plot(final, smooth(final), 'o', 'MarkerSize', 15, 'Color', get(p, 'Color'), 'LineWidth', l_width)
xlabel('Bin');
ylabel('Value');
lgd = legend('Smooth signal', 'Peaks', 'Valleys', 'Selected Peaks');
lgd.Location = 'north';

fig(3) = figure();
plot(data, 'LineWidth', l_width)
hold on
grad = gradient(data);
[m, i] = max(grad);
thresh = data(i);
data_cp = data_interpy;
data_cp(data_cp < thresh) = NaN;
th = plot(data_interpx, data_cp, 'LineWidth', l_width); 
plot([0, length(data)], [thresh, thresh], '--', 'LineWidth', l_width, 'Color', get(th, 'Color'));
plot(i, thresh, 'o', 'MarkerSize', 15, 'LineWidth', l_width);
xlabel('Bin');
ylabel('Value');
lgd = legend('Data', 'Data over threshold', 'Threshold', 'Max gradient');
lgd.Location = 'north';

if save_figs
    for i=1:length(fig)
%         p = get(fig(i), 'OuterPosition');
%         set(fig(i), 'OuterPosition', [p(1), p(2), p(3), p(4)*4/5]);
        set(fig(i), 'PaperUnits', 'normalized')
        set(fig(i), 'PaperPosition', [0 0 1 0.3])
%         set(fig(i), 'PaperPositionMode', 'manual');
%         set(fig(i), 'PaperPosition', [0 0 1]);
        
        print(fig(i), sprintf('hit_inds_%i', i), '-depsc');
    end
end