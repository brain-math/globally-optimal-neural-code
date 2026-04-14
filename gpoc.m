function r = gpoc(x, theta)
%GPOC Gamma-Poisson for one neuron

y = x(1) + x(2) * cos(theta) + x(3) * sin(theta);
r = (cosh(x(4)*y)-1) ./ (2*x(4)^2);

end