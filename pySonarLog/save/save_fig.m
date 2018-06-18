function save_fig(fig, name, first, save_png)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
figure(fig);
if ~exist('first', 'var')
    first = false;
end
if ~exist('save_png', 'var')
    save_png = false;
end
[filepath,fname,~] = fileparts(name);
if contains(filepath, '\')
    % Windows
    folder_char = '\';
else
    folder_char = '/';
end

pngpath = char(strcat(filepath, folder_char, 'png', folder_char));
pngname = char(strcat(pngpath, fname));
if first
    ax = gca;
    outerpos = ax.OuterPosition;
    ti = ax.TightInset; 
    left = outerpos(1) + ti(1);
    bottom = outerpos(2) + ti(2);
    ax_width = outerpos(3) - ti(1) - ti(3);
    ax_height = outerpos(4) - ti(2) - ti(4);
    ax.Position = [left bottom ax_width ax_height];

    fig.PaperPositionMode = 'auto';
    fig_pos = fig.PaperPosition;
    fig.PaperSize = [fig_pos(3) fig_pos(4)];
    
    
    if ~exist(filepath, 'dir')
        mkdir(filepath);
    end
    if save_png
        if ~exist(pngpath, 'dir')
            mkdir(pngpath);
        end
    end
end

print(fig,name,'-depsc')
if save_png
    print(fig,pngname,'-dpng')
end
end

