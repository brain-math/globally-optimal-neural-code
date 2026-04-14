function z = neusel(t, r)
%NEUSEL Neuronal selectivity according to Failor et al. (2025)
% Inputs:
%   t: Theta's (tuning curve mesh).
%   r: Tuning curve matrix (N x T).

z = zeros(size(r, 1), 2); 
r0 = sum(r, 2);
z(:, 1) = sum(cos(t).*r, 2) ./ r0;
z(:, 2) = sum(sin(t).*r, 2) ./ r0;
z = vecnorm(z, 2, 2);

end