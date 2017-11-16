a = 37;
[x,y] = meshgrid(linspace(-4,4,30));
z = exp(-x.^2/15-y.^2);
contour(x,y,z)
xlim([-5 5])
ylim([-5 5])
axis equal
figure
[C,h] = contour(x,y,z);
rotate(get(h,'children'),[0 1],a) 
xlim([-5 5])
ylim([-5 5])
axis equal