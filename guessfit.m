function x0 = guessfit(r, theta)
%GUESSFIT Guess the cosine parameters of tuning curve as initial condition

[m, i] = max(r);
phi = theta(i); tanphi = tan(phi);
u = sqrt(m / (1+tanphi^2));
v = u * tanphi;
x0 = [0, u, v];

end