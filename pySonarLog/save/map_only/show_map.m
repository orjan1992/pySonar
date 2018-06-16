clear, close all
fig = figure();
[fig, leg_text2, leg2] = plot_map(fig, 1.5);
grid on
grid minor
hold on
axis([4.578954180233119e+05 4.580597903462450e+05 6.821450653917387e+06 6.821615026240325e+06]);
XTickLabel = get(gca,'XTick');
XTickLabel = XTickLabel(1):50:XTickLabel(end);
set(gca,'XTick', XTickLabel);
set(gca,'XTickLabel',num2str(XTickLabel'))
% xtickangle(45)
YTickLabel = get(gca,'YTick');
YTickLabel = YTickLabel(1):50:YTickLabel(end);
set(gca,'YTick', YTickLabel);
set(gca,'YTickLabel',num2str(YTickLabel'))
% ytickangle(45);
axis manual;

% p = [6821595.4413, 457960.7201; 6821608.5964, 457969.4139; 6821586.334, 457990.5814; 6821570.1432, 457974.7058; 6821554.4583, 457952.7823; 6821526.6303, 457966.39; 6821514.9932, 457990.2034; 6821516.0051, 458009.859; 6821478.0578, 457951.6483; 6821490.2009, 457979.2417; 6821460.8551, 457973.9498];

% p = [6821599.3104, 457962.9762; 6821586.334, 457990.5814; 6821572.0009, 457964.131; 6821570.1432, 457974.7058; 6821554.4583, 457952.7823; 6821523.0499, 457964.131; 6821514.9932, 457990.2034; 6821516.0051, 458009.859; 6821490.2009, 457979.2417; 6821478.0578, 457951.6483; 6821460.8551, 457973.9498];
% p = [6821596.9453, 457953.8044; 6821587.5526, 457997.6371; 6821574.4028, 457961.6806; 6821570.6457, 457984.2818; 6821553.7388, 457935.655; 6821522.2733, 457954.1469; 6821512.8806, 457981.8847; 6821515.6984, 458022.6354; 6821492.6863, 457984.2818; 6821478.1276, 457933.6003; 6821460.2814, 457981.1998];
p = [6821601.2922, 457955.8591; 6821587.0829, 457993.1853; 6821572.9939, 457964.4202; 6821570.6457, 457978.4603; 6821555.6174, 457935.655; 6821527.4393, 457957.9138; 6821513.8199, 457981.8847; 6821516.168, 458020.2383; 6821493.1559, 457983.5969; 6821478.1276, 457934.9701; 6821462.6296, 457977.7754];
% alig = {'left', 'right', 'left', 'right', 'left', 'left', 'left', 'right', 'right', 'left', 'right'};
r = 'right';
l = 'left';
alig = {r, l, r, l, r, r, r, l, l, r, l};
str = {{'Garage and', 'spool'}, '??????', 'Toolstand', 'Module A', 'Module B', {'Scrap', 'metal'}, 'Module C', 'Module D', 'Module E', 'Module F', 'Module G'};

for i = 1:size(p, 1)
%     t(i) = text(p(i, 2), p(i, 1), num2str(i));
    t(i) = text(p(i, 2), p(i, 1), str{i}, 'HorizontalAlignment', alig{i}, 'FontSize', 12);
%     l(i) = line(nan, nan, 'Color', 'none');
    l_t{i} = num2str(i);
end
t(2) = text(p(2, 2), p(2, 1), str{2}, 'HorizontalAlignment', alig{i}, 'FontSize', 12, 'Color', 'red');
% plot(p(:, 2), p(:, 1), '*r');
% 
% [lgd,icons,plots,txt] = legend([leg2 l_t], [leg_text2 l_t]);
% k = 1;
% for i = 3:13
%     plots(i) = t(k);
%     k = k+1;
% end
save_fig(fig, 'map')