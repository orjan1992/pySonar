function mouseDownMat( object, eventdata )
%MOUSEDOWN Summary of this function goes here
%   Detailed explanation goes here
c = get (gca, 'CurrentPoint');
a = [c(1, 2) c(1, 1)];
fprintf('%s, %s; ', num2str(a(1)), num2str(a(2)));
%set (gcf, 'WindowButtonDownFcn', @mouseDownMat);
end

