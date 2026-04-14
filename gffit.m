function fitted = gffit(r, theta)
%GFFIT Fit the Gamma-Poisson model to one tuning curve row vector r with sigma free

options = optimoptions('lsqnonlin', 'Display', 'off');
sigma0 = 1; x0 = guessfit(r, theta); x0 = [x0, sigma0];
fitted = lsqnonlin(@(x) gpoc(x, theta) - r, x0, [-Inf, -Inf, -Inf, 0],...
    [Inf, Inf, Inf, Inf], options);

end

