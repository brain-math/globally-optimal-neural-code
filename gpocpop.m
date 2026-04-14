function [R, Y] = gpocpop(y0, u, v, sigma, theta)
%GPOCPOP Gamma-Poisson model for a population of neurons
% y0, u, v, sigma are column vectors; theta is a row vector

Y = y0 + u * cos(theta) + v * sin(theta);
R = (cosh(sigma.*Y)-1)./(2*sigma.^2);

end