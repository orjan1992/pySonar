clear, close all

save_figs = true;
fig(1) = figure();
new_pos = [6821549.44242929, 457963.280194848];
[fig(1), leg_text, leg] = plot_obs_on_map('obstacles20180502-122919', 'r', fig(1), new_pos);
[fig(1), leg_text2, leg2] = plot_map(fig(1));
xlim([457937.8554393220 457981.5119819951]);
ylim([6821540.853014296 6821573.595421301]);
lgd = legend([leg leg2], [leg_text, leg_text2]);
lgd.Location = 'southeast';

fig(2) = figure();
new_pos = [6821549.84242929, 457963.080194848];
[fig(2), leg_text, leg] = plot_obs_on_map('obstacles20180502-123108.mat', 'r', fig(2), new_pos);
[fig(2), leg_text2, leg2] = plot_map(fig(2));
lgd = legend([leg leg2], [leg_text, leg_text2]);
lgd.Location = 'southeast';
xlim([457937.8554393220 457981.5119819951]);
ylim([6821540.853014296 6821573.595421301]);

fig(3) = figure();
new_pos = [6821547.1099567, 457958.662528077];
[fig(3), leg_text, leg] = plot_obs_on_map('obstacles20180502-124217', 'r', fig(3), new_pos);
[fig(3), leg_text2, leg2] = plot_map(fig(3));
lgd = legend([leg leg2], [leg_text, leg_text2]);
lgd.Location = 'southeast';
xlim([457937.8554393220 457981.5119819951]);
ylim([6821540.853014296 6821573.595421301]);

fig(4) = figure();
new_pos = [6821542.44242929, 457960.280194848];
[fig(4), leg_text, leg] = plot_obs_on_map('obstacles20180502-124621', 'r', fig(4), new_pos);
[fig(4), leg_text2, leg2] = plot_map(fig(4));
lgd = legend([leg leg2], [leg_text, leg_text2]);
lgd.Location = 'southeast';
xlim([457937.8554393220 457981.5119819951]);
ylim([6821540.853014296 6821573.595421301]);

if save_figs
    for i=1:length(fig)
        set(fig(i), 'PaperUnits', 'normalized')
%         set(fig(i), 'PaperPosition', [0 0 1 1])
        set(fig(i), 'PaperPositionMode', 'auto')
        
        print(fig(i), sprintf('fig_%i', i), '-depsc');
    end
end