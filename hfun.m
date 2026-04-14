function h = hfun(d)
%HFUN Approximate classification error h(d)

h = erfc(d/2/sqrt(2))/2;

end
