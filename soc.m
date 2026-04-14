function r = soc(x, theta)
%SOC Square-of-circle for one neuron

r = x(1) + x(2) * cos(theta) + x(3) * sin(theta);
r = r.^2 / 4;

end