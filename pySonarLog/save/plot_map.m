function fig = plot_map(disp_anchor_line, disp_cables, disp_installation_line, ...
                        disp_installation_point, disp_obstruction_point, disp_restricted_areas, ...
                        disp_pipelines, disp_protection, disp_safety_zone, disp_wells)
    path = 'Snorre B WGS84';
    if ~exist('disp_anchor_line', 'var')
        disp_anchor_line = true;
    end
    if ~exist('disp_cables', 'var')
        disp_cables = true;
    end
    if ~exist('disp_installation_line', 'var')
        disp_installation_line = true;
    end
    if ~exist('disp_installation_point', 'var')
        disp_installation_point = true;
    end
    if ~exist('disp_obstruction_point', 'var')
        disp_obstruction_point = true;
    end
    if ~exist('disp_restricted_areas', 'var')
        disp_restricted_areas = false;
    end
    if ~exist('disp_pipelines', 'var')
        disp_pipelines = true;
    end
    if ~exist('disp_protection', 'var')
        disp_protection = false;
    end
    if ~exist('disp_safety_zone', 'var')
        disp_safety_zone = false;
    end
    if ~exist('disp_wells', 'var')
        disp_wells = true;
    end
    listing = dir(path);
    n = 1;
    s = struct;

    colorOrder = [get(gca, 'ColorOrder'); get(gca, 'ColorOrder')];


    fig = figure();
    % leg = {};
    leg_text = {};
    leg_counter = 1;
    for i = 1:length(listing)
        if endsWith(listing(i).name, '.shx')
            name = split(listing(i).name, '.');
            s = setfield(s, char(name(1)), m_shaperead(char(strcat(listing(i).folder, '\', name(1)))));
        end
    end
    hold on

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
                wells = plot(s.Wells.ncst{k}(1), s.Wells.ncst{k}(2), '*', 'Color', colorOrder(leg_counter, :));
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
                obstruction = plot(s.Obstruction_point.ncst{k}(1), s.Obstruction_point.ncst{k}(2), '*', 'Color', colorOrder(leg_counter, :));
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
        for k=1:length(s.Installation_line.ncst)
            if first
                Installation_line = plot(s.Installation_line.ncst{k}(:, 1),s.Installation_line.ncst{k}(:, 2), 'Color', colorOrder(leg_counter, :)); 
                leg(leg_counter) = Installation_line;
                leg_text{leg_counter} = 'Installation Line';
                leg_counter = leg_counter + 1; 
                first = false;
            else
                plot(s.Installation_line.ncst{k}(:, 1),s.Installation_line.ncst{k}(:, 2), 'Color', get(Installation_line, 'Color')); 
            end
        end
    end

    %% Installation_point
    if disp_installation_point
        first = true;
        for k=1:length(s.Installation_point.ncst) 
            if first
                Installation_point = plot(s.Installation_point.ncst{k}(1), s.Installation_point.ncst{k}(2), '*', 'Color', colorOrder(leg_counter, :));
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

    xlabel('East')
    ylabel('North')
    legend(leg, leg_text);

%% ROV
% plot(458020.9429046898, 6821614.360178768, 'or');