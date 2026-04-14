function cError = getnmcw(rFun, Wx, nPair, dTheta, nSample)
%GETNMCW Calculate neurometric curve with linear classifier in linearized RNN
% Inputs:
%   rFun: Tuning curve function r(theta)
%   xFun: Spike count sampler function
%   nPair: Number of random theta pairs for each delta theta
%   dThetaAll: List of delta thetas
%   nSample: Dataset size (including training and testing) for each pair
% Outputs:
%   cError: Classification error matrix (nPair x ndTheta)

N = size(Wx, 1); cError = zeros(1, nPair);
for iPair = 1 : nPair
    theta1 = rand * 2 * pi; theta2 = theta1 + dTheta; % Choose a pair of thetas
    r1 = rFun(theta1); r2 = rFun(theta2); % Get firing rates
    X1 = (diag(r1.^(-1/2))-Wx) \ randn(N, nSample) + r1; % Spike counts
    X2 = (diag(r2.^(-1/2))-Wx) \ randn(N, nSample) + r2;
    X = [X1, X2]';
    Y = [ones(nSample, 1); zeros(nSample, 1)]; % Class labels

    % Linear classifier
    mdl = fitclinear(X, Y, 'Crossval', 'on', 'KFold', 5);
    cError(iPair) = kfoldLoss(mdl);
end

end