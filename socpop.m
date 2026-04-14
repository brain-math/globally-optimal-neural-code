function [R, Y] = socpop(y0, u, v, theta)
%SOCPOP Square-of-circle for a population of neurons
% y0, u, v are column vectors; theta is a row vector

Y = y0 + u * cos(theta) + v * sin(theta);
R = Y .^ 2 / 4;

end