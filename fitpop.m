function [y0, u, v] = fitpop(R, theta)
%FITPOP Fit the square-of-circle model to tuning curve matrix R (#neuron x #stimulus)

n = size(R, 1); y0 = zeros(n, 1); u = y0; v = y0;
for i = 1 : n
    r = R(i, :);
    fitted = fitneu(r, theta);
    y0(i) = fitted(1); u(i) = fitted(2); v(i) = fitted(3);
end

end