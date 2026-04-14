function [dPhi, peakRatio, rsq] = fitbivmpop(R, theta)
%FITBIVMPOP Fit sum of two von Mises model to tuning curve matrix R (N x T)
% Inputs:
%       R (N x T): Matrix of firing rates. N is the number of neurons
%            and T is the number of theta values.
%       theta (1 x T): Orientations corresponding to the firing rates
% Outputs:
%       dPhi (N x 1): Phase difference between the two peaks
%       peakRatio (N x 1): Ratio between the magnitude of the small
%            peaks versus the large peak. A value close to 1 means bimodal.
%       rsq: R squared of the fit.

N = size(R, 1); dPhi = zeros(N, 1); peakRatio = dPhi; rsq = dPhi;
for i = 1 : N
    r = R(i, :);
    [temp1, temp2, temp3] = fitbivmneu(r, theta);
    peakRatio(i) = temp1;
    dPhi(i) = temp2;
    rsq(i) = temp3;
end

end