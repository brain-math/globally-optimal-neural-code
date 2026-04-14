function sigma = fitnbin(X)
%FITNBIN Fit negative binomial to spike counts
% X is a cell array of spike count vectors

options = optimset('Display', 'off');
nBin = length(X);
fitted = fminsearch(@(p) nllnbin(X, p), ones(nBin+1, 1), options);
sigma = sqrt(max(1/fitted(end), 0));

end

function nl = nllnbin(X, p)
r = p(end);
ll = cellfun(@(x, i) sum(log(gamma(x+r))-log(gamma(x+1))-log(gamma(r))-r*log(1+p(i))+x*(log(p(i))-log(1+p(i)))), X, num2cell(1:length(X)));
nl = -sum(ll);
end