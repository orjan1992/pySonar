clear, close all
f(1) = "05_15_collision avoidance_test\15_01_path_test";
f(2) = "05_15_collision avoidance_test\15_13_path_rundt_k";
f(3) = "05_15_collision avoidance_test\15_23_litt_rot";
f(4) = "05_15_collision avoidance_test\15_30_no_smoothing";
f(5) = "05_19 Collision Test\15_39_best";
f(6) = "05_19 Collision Test\13_14_fra_cage";
f(7) = "05_19 Collision Test\13_17_tilbake_til_cage";
f(8) = "05_19 Collision Test\13_27_dårlig_los";
f(9) = "05_19 Collision Test\13_32_dårlig_los";
f(10) = "05_19 Collision Test\13_33_kollisjonskurs";
f(11) = "05_15_collision avoidance_test\08_49_rundtur_rundt_k_veldig_bra";
f(12) = "05_14_from_garage_to_c_and_back_rerun";


% f(1) = "05_15_collision avoidance_test\15_13_path_rundt_k";
% f(2) = "05_15_collision avoidance_test\08_49_rundtur_rundt_k_veldig_bra";
ed50 = false;

%% sort
fig = gcf;
legend('off');
for ssdd = 1:size(f, 2)
    listing = dir(char(f(ssdd)));
    for i = 1:length(listing)
        if (startsWith(listing(i).name, 'collision') || startsWith(listing(i).name, 'obstacles')) && endsWith(listing(i).name, '.mat')
            n = listing(i).name;
            s_i = regexp(n, '\d{8}-\d{6}');
    %         f_time(i) = datetime([str2double(n(s_i:s_i+3)), str2double(n(s_i+4:s_i+5)), ...
    %             str2double(n(s_i+6:s_i+7)), str2double(n(s_i+9:s_i+10)), ...
    %             str2double(n(s_i+11:s_i+12)), str2double(n(s_i+13:s_i+14))]);
            f_time(i) = datetime(n(s_i:s_i+14), 'InputFormat','yyyyMMdd-HHmmss');
            f_ind(i) = i;
        else
            f_time(i) = datetime(2020, 12, 12);
            f_ind(i) = 0;
        end
    end
    tmp = sortrows(table(f_time', f_ind'), 'Var1', 'ascend');
    f_time = table2array(tmp(:, 1));
    f_ind = table2array(tmp(:, 2));
    i = find(f_ind == 0);
    f_time = f_time(1:i-1);
    f_ind = f_ind(1:i-1);


    for i = 1:length(f_time)
        [fig, leg_text, leg, all_obj] = plot_obs_on_map(strcat(listing(f_ind(i)).folder, ...
            '\', listing(f_ind(i)).name), 'r', fig, false, 0.1, false);
    end
end
plot_map(fig)
axis([4.578671149495515e+05 4.581206925223056e+05 6.821419244847148e+06 6.821672822419900e+06]);