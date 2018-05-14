clear, close all
plot_map_ed50()
% xlim([4.580077097463052e+05 4.581446866820825e+05]);
% ylim([6.821656150314445e+06 6.821793127250223e+06]);
pos_a = [457969.51250816154, 6821763.814683317];
pos_b = [458014.74679709796, 6821712.464212717];
pos_c = [457964.3089998879, 6821790.950091968];
k_module = [458101, 6821732];
offset = pos_b - k_module;

pos_a = pos_a - offset;
pos_b = pos_b - offset;
pos_c = pos_c - offset;
% pos_c = [458051 6821772];
plot(pos_a(1), pos_a(2), 'ro');
plot(pos_b(1), pos_b(2), 'ro');
plot(pos_c(1), pos_c(2), 'bo');