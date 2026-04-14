function [y0, a, b] = gencircnneg(N, rho)
%GENCIRCNNEG Sample (the parameters) of a nonnegative circle
% Inputs:
%   N: Number of neurons
%   rho: Radius of the circle
% Outputs:
%   y0: Center of the circle
%   a, b: Two radii vector of the circle

a = rand(N, 1) - 1/2; b = rand(N, 1) - 1/2; % Random 2D subspace for the circle
b = b - (b'*a/norm(a)) * a/norm(a); % Orthogonalize
b = b/norm(b)*rho; a = a/norm(a)*rho; % Radius
y0 = sqrt(a.^2 + b.^2);

end