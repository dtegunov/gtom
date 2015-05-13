function [ x ] = radialoffset( x, ny, sidelength, defocus, defocusdelta, cs, lambda, refangle, angle )

r = x * ny * 2 / sidelength;
term1 = defocus;
term2 = 2 * cs * lambda^2 * r^2 * term1;
term3 = sqrt(defocus^2 + cs^2 * lambda^4 * r^4 - (term2));
term4 = sqrt(cs * (abs(abs(defocus) - term3))) / (cs * lambda);
x = term4 / (ny * 2 / sidelength);

end

