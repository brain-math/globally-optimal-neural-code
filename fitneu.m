function fitted = fitneu(r, theta)
%FITNEU Fit the square-of-circle model to one tuning curve row vector r

options = optimoptions('lsqnonlin', 'Display', 'off');
x0 = guessfit(r, theta);
fitted = lsqnonlin(@(x) soc(x, theta) - r, x0, [], [], options);

end