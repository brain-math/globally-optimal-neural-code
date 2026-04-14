function [ratio, dPhi, rsq] = fitbivmneu(r, theta)
%FITBIVMNEU Fit sum of two von Mises model to one tuning curve
% Inputs:
%       r: A row vector of firing rates
%       theta: Orientations corresponding to the firing rates
% Outputs:
%       ratio: Ratio between the magnitude of the small
%            peaks versus the large peak. A value close to 1 means bimodal.
%       dPhi: Phase difference between the two peaks
%       rsq: R squared of the fit.

options = optimoptions('lsqnonlin', 'Display', 'off');

% Fit sequentially: fit one peak, subtract, then fit another
% Fitting the two peaks simultaneously yields virtually the same results,
% but may go to a local minimum that does not capture the largest two peaks
x0 = guessfitvm(r, theta);
fitted1 = lsqnonlin(@(x) vm(x, theta)-r, x0, [0, 0, -pi],...
    [Inf, 1000, pi], options);
r1 = vm(fitted1, theta);
r2 = r - r1;
x0 = guessfitvm(r2, theta);
fitted2 = lsqnonlin(@(x) vm(x, theta)-r2, x0, [0, 0, fitted1(3)+pi/8],...
    [Inf, Inf, fitted1(3)+2*pi-pi/8], options);

rm1 = max(r); rm2 = max(r2);
ratio = min(rm1, rm2) / max(rm1, rm2);
dPhi = fitted2(3) - fitted1(3);
if dPhi > pi
    dPhi = 2*pi - dPhi;
end

fittedR = r1 + vm(fitted2, theta);
rsq = r2score(r', fittedR');

end

function x0 = guessfitvm(r, theta)
% Guess a single von Mises model's parameters
[m, i] = max(r);
x0 = [m, 1, theta(i)];
end

