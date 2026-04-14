function cError = getnmc(rFun, xFun, nPair, dThetaAll, nSample)
%GETNMC Calculate neurometric curve with likelihood ratio classifier
% Inputs:
%   rFun: Tuning curve function r(theta), maps to column vector
%   xFun: Spike count sampler function
%   nPair: Number of random theta pairs for each delta theta
%   dThetaAll: List of delta thetas
%   nSample: Dataset size (including training and testing) for each pair
% Outputs:
%   cError: Classification error matrix (nPair x ndTheta)

ndTheta = length(dThetaAll);
cError = zeros(nPair, ndTheta);
h = waitbar(0);
for idTheta = 1 : ndTheta
    dTheta = dThetaAll(idTheta);
    for iPair = 1 : nPair
        theta1 = rand * 2 * pi; theta2 = theta1 + dTheta; % Choose a pair of thetas
        r1 = rFun(theta1); r2 = rFun(theta2); % Get firing rates
        X = xFun([ones(nSample, 1)*r1'; ones(nSample, 1)*r2']); % Spike counts
        Y = [ones(nSample, 1); zeros(nSample, 1)]; % Class labels

        % Likelihood ratio classifier
        YPred = X * log(r1./r2) > sum(r1-r2);
        cError(iPair, idTheta) = mean(YPred ~= Y);

        waitbar(((idTheta-1)*nPair+iPair)/(ndTheta*nPair), h)
    end
end
delete(h)

end