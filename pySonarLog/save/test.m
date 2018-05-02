clear, close all
path = 'D:\SkyStorage\OneDrive - NTNU\Code\pySonar\pySonarLog\save\Snorre B WGS84\Installation_line';
inst_line = m_shaperead(path);
hold on
k = 4
plot(inst_line.ncst{k}(:, 1),inst_line.ncst{k}(:, 2), 'r', 'LineWidth', 5);
tmp = 1:length(inst_line.ncst);
tmp =[tmp(1) tmp(3:end)];
for k=tmp
    plot(inst_line.ncst{k}(:, 1),inst_line.ncst{k}(:, 2));  
    t{k} = num2str(k);
end
xlim([4.578895027624309e+05 4.581068139963168e+05]); ylim([6.821422897196262e+06 6.821600467289720e+06]);
% legend(t)