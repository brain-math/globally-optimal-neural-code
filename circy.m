function [y, t] = circy(T, y0, a, b)
%CIRCY Generate circles (which can be transformed to get tuning curves)
% Inputs:
%   T: Number of thetas (granularity of the tuning curves)
%   y0: Center of the circle
%   a, b: Two radii vector of the circle
% Outputs:
%   y: Circles (N x T)

t = linspace(0, 2*pi, T+1); t = t(2:end); % Tuning curve mesh
y = y0 + a * cos(t) + b * sin(t);

end