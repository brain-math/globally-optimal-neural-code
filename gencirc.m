function [y0, a, b] = gencirc(N, beta, rho)
%GENCIRC Sample (the parameters) of a circle
% Inputs:
%   N: Number of neurons
%   beta: Parameter for the distribution of relative vertical shift
%   rho: Radius of the circle
% Outputs:
%   y0: Center of the circle.
%   a, b: Two radii vector of the circle

x = rand(N, 1) * beta; % Relative vertical shift
a = rand(N, 1) - 1/2; b = rand(N, 1) - 1/2; % Random 2D subspace for the circle
b = b - (b'*a/norm(a)) * a/norm(a); % Orthogonalize
b = b/norm(b)*rho; a = a/norm(a)*rho; % Radius
A = sqrt(a.^2+b.^2);
y0 = x .* A;

end