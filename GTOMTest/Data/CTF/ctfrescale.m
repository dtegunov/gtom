function [ x ] = ctfrescale( x, sidelength, pixelsize, oldz, newz, cs, voltage )

ny = 1 / pixelsize;
lambda = sqrt(150.4 / (voltage * (1 + voltage / 1022000.0))) * 1e-10;

K = x.*(ny / sidelength);
c = cs;
l = lambda;
D = oldz;
d = newz;

k = sqrt(abs(abs(d)-sqrt(c.^2.*K.^4.*l.^4+2*c.*D.*K.^2.*l.^2+d.^2))./(c*l^2));
% k = sqrt((sqrt(c^2.*K.^4.*l.^4-2.*c.*D.*K.^2.*l.^2+d.^2)+d)./(c.*l.^2));

x = k./(ny / sidelength);

end

