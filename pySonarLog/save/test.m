fig = fig();
fig = plot_obs_on_map('D:\SkyStorage\OneDrive - NTNU\Code\pySonar\pySonarLog\save\05_15_collision avoidance_test\08_49_rundtur_rundt_k_veldig_bra\obstacles20180515-090152.mat');
plot_map_ed50(fig)
axis([4.580245081291553e+05 4.581062449341996e+05 6.821696224594763e+06 6.821777961399807e+06]);
set (gcf, 'WindowButtonDownFcn', @mouseDown);