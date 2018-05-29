function [fig, leg_text, leg] = plot_map(fig, w, disp_anchor_line, disp_cables, disp_installation_line, ...
                        disp_installation_point, disp_obstruction_point, disp_restricted_areas, ...
                        disp_pipelines, disp_protection, disp_safety_zone, disp_wells)
    path = 'D:\SkyStorage\OneDrive - NTNU\Code\pySonar\pySonarLog\save\Snorre B ED50';
    if ~exist('disp_anchor_line', 'var')
        disp_anchor_line = false;
    end
    if ~exist('disp_cables', 'var')
        disp_cables = false;
    end
    if ~exist('disp_installation_line', 'var')
        disp_installation_line = true;
    end
    if ~exist('disp_installation_point', 'var')
        disp_installation_point = true;
    end
    if ~exist('disp_obstruction_point', 'var')
        disp_obstruction_point = false;
    end
    if ~exist('disp_restricted_areas', 'var')
        disp_restricted_areas = false;
    end
    if ~exist('disp_pipelines', 'var')
        disp_pipelines = false;
    end
    if ~exist('disp_protection', 'var')
        disp_protection = false;
    end
    if ~exist('disp_safety_zone', 'var')
        disp_safety_zone = false;
    end
    if ~exist('disp_wells', 'var')
        disp_wells = false;
    end
    listing = dir(path);
    n = 1;
    s = struct;

    if ~exist('fig', 'var')
        fig = figure();
    else
        figure(fig);
    end
    hold on
    colorOrder = [get(gca, 'ColorOrder'); get(gca, 'ColorOrder')];
    % leg = {};
    leg_text = {};
    leg_counter = 1;
    for i = 1:length(listing)
        if endsWith(listing(i).name, '.shx')
            name = split(listing(i).name, '.');
            s = setfield(s, char(name(1)), m_shaperead(char(strcat(listing(i).folder, '\', name(1)))));
        end
    end

    if ~exist('w', 'var')
        w = 1;
    end
    %% Restricted areas
    if disp_restricted_areas
        first = true;
        for k=1:length(s.Other_restricted_areas.ncst)
            if first
                Other_restricted_areas = fill(s.Other_restricted_areas.ncst{k}(:, 1), s.Other_restricted_areas.ncst{k}(:, 2), colorOrder(leg_counter, :));
                leg(leg_counter) = Other_restricted_areas;
                leg_text{leg_counter} = 'Other Restricted Areas';
                leg_counter = leg_counter + 1;
                first = false;
            else
                fill(s.Other_restricted_areas.ncst{k}(:, 1), s.Other_restricted_areas.ncst{k}(:, 2), [229, 125, 41]/255);
            end
        end
    end

    %% Protection
    if disp_protection
        first = true;
        for k=1:length(s.Protection.ncst)
            if first
                Protection = fill(s.Protection.ncst{k}(:, 1), s.Protection.ncst{k}(:, 2), colorOrder(leg_counter, :));
                leg(leg_counter) = Protection;
                leg_text{leg_counter} = 'Protection';
                leg_counter = leg_counter + 1;
                first = false;
            else
                fill(s.Protection.ncst{k}(:, 1), s.Protection.ncst{k}(:, 2), [165, 0, 149]/255);
            end
        end
    end

    %% Safety_zone
    if disp_safety_zone
        first = true;
        for k=1:length(s.Safety_zone.ncst)
            if first
                Safety_zone = fill(s.Safety_zone.ncst{k}(:, 1), s.Safety_zone.ncst{k}(:, 2), colorOrder(leg_counter, :));
                leg(leg_counter) = Safety_zone;
                leg_text{leg_counter} = 'Safety Zone';
                leg_counter = leg_counter + 1;
                first = false;
            else
                fill(s.Safety_zone.ncst{k}(:, 1), s.Safety_zone.ncst{k}(:, 2), [209, 205, 12]/255);
            end
        end
    end

    %% Wells
    if disp_wells
        first = true;
        for k=1:length(s.Wells.ncst) 
            if first
                wells = plot(s.Wells.ncst{k}(1), s.Wells.ncst{k}(2), 'o', 'Color', colorOrder(leg_counter, :));
                leg(leg_counter) = wells;
                leg_text{leg_counter} = 'Wells';
                leg_counter = leg_counter + 1;
                first = false;
            else
                plot(s.Wells.ncst{k}(1), s.Wells.ncst{k}(2), '*', 'Color', get(wells, 'Color'));
            end
        end
    end


    %% Anchor Lines
    if disp_anchor_line
        first = true;
        for k=1:length(s.Anchor_line.ncst)
            if first
                anchor = plot(s.Anchor_line.ncst{k}(:, 1),s.Anchor_line.ncst{k}(:, 2), 'Color', colorOrder(leg_counter, :)); 
                leg(leg_counter) = anchor;
                leg_text{leg_counter} = 'Anchor Lines';
                leg_counter = leg_counter + 1;
                first = false;
            else
                plot(s.Anchor_line.ncst{k}(:, 1),s.Anchor_line.ncst{k}(:, 2), 'Color', get(anchor, 'Color')); 
            end
        end
    end

    %% Obstruction point
    if disp_obstruction_point
        first = true;
        for k=1:length(s.Obstruction_point.ncst) 
            if first
                obstruction = plot(s.Obstruction_point.ncst{k}(1), s.Obstruction_point.ncst{k}(2), 'x', 'Color', colorOrder(leg_counter, :));
                leg(leg_counter) = obstruction;
                leg_text{leg_counter} = 'Obstruction Point';
                leg_counter = leg_counter + 1;
                first = false;
            else
                plot(s.Obstruction_point.ncst{k}(1), s.Obstruction_point.ncst{k}(2), '*', 'Color', get(obstruction, 'Color')); 
            end
        end
    end

    %% Cables
    if disp_cables
        first = true;
        for k=1:length(s.Cables.ncst)
            if first
                Cables = plot(s.Cables.ncst{k}(:, 1),s.Cables.ncst{k}(:, 2), 'Color', colorOrder(leg_counter, :));
                leg(leg_counter) = Cables;
                leg_text{leg_counter} = 'Cables';
                leg_counter = leg_counter + 1; 
                first = false;
            else
                plot(s.Cables.ncst{k}(:, 1),s.Cables.ncst{k}(:, 2), 'Color', get(Cables, 'Color')); 
            end
        end
    end
    %% Installation_line
    if disp_installation_line
        first = true;
        for k=[1 3:length(s.Installation_line.ncst)]
            if first
                Installation_line = plot(s.Installation_line.ncst{k}(:, 1),s.Installation_line.ncst{k}(:, 2), 'Color', colorOrder(5, :), 'LineWidth', w); 
                leg(leg_counter) = Installation_line;
                leg_text{leg_counter} = 'Installation Line';
                leg_counter = leg_counter + 1; 
                first = false;
            else
                plot(s.Installation_line.ncst{k}(:, 1),s.Installation_line.ncst{k}(:, 2), 'Color', get(Installation_line, 'Color'), 'LineWidth', w); 
            end
        end
        %% Hand drawn
        % Garage
        mod{1} = [6821811.8469, 458049.0655; 6821815.7441, 458047.856; 6821825.4202, 458055.5834; 6821828.8471, 458062.0341; 6821825.4874, 458062.9748; 6821818.4319, 458059.9511; 6821810.2342, 458053.9707];
        % Toolstand
        mod{2} = [6821793.8339, 458059.7163; 6821793.4347, 458058.3986; 6821791.9829, 458059.135; 6821792.5636, 458060.3751];        
        % Other
        mod{3} = [6821744.4469, 458049.5985; 6821742.8293, 458053.5526; 6821739.055, 458053.3728; 6821739.6841, 458049.329; 6821743.5482, 458048.5202];
        mod{4} = [6821730.4281, 458033.2433; 6821731.5065, 458035.4899; 6821728.9903, 458036.8379; 6821728.4511, 458033.7825; 6821730.2484, 458032.9737];
        mod{5} = [6821769.8266, 458106.3445; 6821767.7131, 458109.1473; 6821765.2779, 458107.1256; 6821766.8401, 458104.6905];
        mod{6} = [6821745.569, 458054.6486; 6821749.3031, 458057.8954; 6821744.5863, 458062.6086; 6821740.6557, 458061.4565];
        mod{7} = [6821805.3145, 458065.2271; 6821805.3145, 458068.1597; 6821803.7423, 458068.3692; 6821803.1527, 458064.9129];
        mod{8} = [6821848.9445, 458055.9054; 6821848.3549, 458060.3044; 6821845.2104, 458060.3044; 6821845.2104, 458055.4865];
        mod{9} = [6821841.0832, 458017.6763; 6821841.4763, 458015.372; 6821844.0312, 458015.1626; 6821843.8346, 458018.5142;];
        mod{10} = [6821726.3089, 458068.6834; 6821726.3089, 458071.5113; 6821724.3436, 458075.9103; 6821721.7887, 458075.7008; 6821722.9679, 458072.7682; 6821721.9852, 458068.9976];
        for i = 1:length(mod)
            mod{i} = [mod{i}; mod{i}(1, :)];
            g = plot(mod{i}(:, 2), mod{i}(:, 1), 'Color', [199, 234, 70]/255, 'LineWidth', w);
        end
        leg(leg_counter) = g;
        leg_text{leg_counter} = 'Hand drawn obstacles';
        leg_counter = leg_counter + 1; 
    end

    %% Installation_point
    if disp_installation_point
        first = true;
        for k=1:length(s.Installation_point.ncst) 
            if first
                if disp_installation_line
                    Installation_point = plot(s.Installation_point.ncst{k}(1), s.Installation_point.ncst{k}(2), '*', 'Color', get(Installation_line, 'Color'));
                else
                    Installation_point = plot(s.Installation_point.ncst{k}(1), s.Installation_point.ncst{k}(2), '*', 'Color', colorOrder(4, :));
                end
                leg(leg_counter) = Installation_point;
                leg_text{leg_counter} = 'Installation Point';
                leg_counter = leg_counter + 1; 
                first = false;
            else
                plot(s.Installation_point.ncst{k}(1), s.Installation_point.ncst{k}(2), '*', 'Color', get(Installation_point, 'Color')); 
            end
        end
    end

    %% Pipelines
    if disp_pipelines
        first = true;
        for k=1:length(s.Pipelines.ncst)
            if first
                Pipelines = plot(s.Pipelines.ncst{k}(:, 1),s.Pipelines.ncst{k}(:, 2), 'Color', colorOrder(leg_counter, :)); 
                leg(leg_counter) = Pipelines;
                leg_text{leg_counter} = 'Pipelines';
                leg_counter = leg_counter + 1; 
                first = false;
            else
                plot(s.Pipelines.ncst{k}(:, 1),s.Pipelines.ncst{k}(:, 2), 'Color', get(Pipelines, 'Color')); 
            end
        end
    end

    xlabel('East [m]')
    ylabel('North [m]')
    legend(leg, leg_text);
end
%% ROV
% plot(458020.9429046898, 6821614.360178768, 'or');