function [ x ] = ctfrescale2( X, sidelength, oldpx, newpx, oldz, newz, cs, voltage )

lambda = 12.2643247 / sqrt(voltage * (1.0 + voltage * 0.978466e-6));

c = cs;
l = lambda;
Z = oldz;
z = newz;
T = oldpx;
t = newpx;
n = 1 / (sidelength);

summand1 = c^2 * l^4 * n^4 * X^4;
summand2 = 2 * c * l^2 * n^2 * T^2 * X^2 * Z;
summand3 = T^4 * z^2;

firstroot = -sqrt(t^4 * T^4 * (summand1 + summand2 + summand3));
numerator = firstroot + t^2 * T^4 * abs(z);
denominator = c * l^2 * n^2 * T^4;

x = sqrt(abs(numerator / denominator));

end

