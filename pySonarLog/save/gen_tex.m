function gen_tex(f, columns, include)
    tmp = split(f, '\');

    f = f + "\figs";
    tex_path = "fig/Experiment/" + tmp{end} + "/";
    %% sort
    listing = dir(char(f));
    for i = 1:length(listing)
        if endsWith(listing(i).name, '.eps')
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
    if exist('include', 'var')
        f_time = f_time(include);
        f_ind = f_ind(include);
    end

    fid = fopen(f+"\fig.tex", 'w');
    
    width = "0.9";
    max_rows = 2;
    if columns == 2
        width = "0.45";
        max_rows = 3;
    end
    if columns == 3
        width = "0.3";
        max_rows = 3;
    end
    label = replace(tex_path, '/', ':');
    i = 1;
    for r = 1:ceil(length(f_time)/max_rows)
        fprintf(fid, '\\begin{figure}[H]\n\t\\centering\n');
        while i < r*max_rows +1 && i < length(f_time)+1
            name = listing(f_ind(i)).name;
            tmp = split(name, '.');
            fprintf(fid, ['\t\\begin{subfigure}[b]{%s\\textwidth}', ...
                '\n\t\t\\includegraphics[width=\\textwidth]{%s}', ...
                '\n\t\t\\caption{}', ...
                '\n\t\t\\label{%s}', ...
            '\n\t\\end{subfigure}\n'], width, tex_path + name, label + tmp{1});
            if i < r*max_rows && i < length(f_time)
                if columns ~= 1 && mod(mod(i, max_rows)+1, columns) == 0
                    fprintf(fid, '\t~\n');
                else
                    fprintf(fid, '\n');
                end
            end
            i = i + 1;
        end
        fprintf(fid, '\t\\caption{%s%i}\n\t\\label{%s%i}\n\\end{figure}\n\n', replace(label, {':', '_'}, ' '), r, label, r);
    end
    fclose(fid);
end
        