function [r, t] = genvmr(N, T, kappa, g0)
%GENVMR Generate tuning curves using the von Mises model
% Inputs:
%   N: Number of neurons
%   T: Number of thetas (granularity of the tuning curves)
%   kappa: Width parameter
%   g0: Height parameter
% Outputs:
%   r: Tuning curves (N x T)

t = linspace(0, 2*pi, T); % Tuning curve mesh
theta = linspace(0, 2*pi, N+1)'; theta = theta(2:end); % Preferred theta's
r = g0^2 * exp(kappa * cos(theta - t));

end