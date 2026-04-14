function r = vm(x, theta)
%VM One von Mises model for one neuron
r = x(1)*exp(x(2)*(cos(theta-x(3))-1));

end