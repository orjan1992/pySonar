function ta = textArrow(Start, End, text)
%TEXTARROW Summary of this function goes here
%   Detailed explanation goes here
%% Get limits
axun = get(gca,'Units');
set(gca,'Units','normalized');
axpos = plotboxpos(gca);
axlim = axis(gca);
axwidth = diff(axlim(1:2));
axheight = diff(axlim(3:4));


%% Transform data
start_a = [(Start(1)-axlim(1))*axpos(3)/axwidth + axpos(1), (Start(2)-axlim(3))*axpos(4)/axheight + axpos(2)];
end_a = [(End(1)-axlim(1))*axpos(3)/axwidth + axpos(1), (End(2)-axlim(3))*axpos(4)/axheight + axpos(2)];


%% Restore axes units
set(gca,'Units',axun)
%% annotate
ta = annotation('textarrow', start_a, end_a);
ta.String = text;
end

