nx = 256;
n = floor(nx / 2) + 1;
pixelsize = 0.8e-10;
fper = 1 / (pixelsize * n);

Cs = 2e-3;
V = 300e3;
lambda = sqrt(150.4 / (V * (1.0 + V / 1022000.0))) * 1e-10;
z = 1.5e-6;
ny = 0.5 / pixelsize;
Tgrid = 2 / (pixelsize / 512);

periods = (1:n)';
for x=1:n
    s = x / (n - 1) / (2 * pixelsize);    
    
    A = 0.5 * z * lambda;
    B = 0.25 * Cs * lambda^3;
    
    c4 = B;
    c3 = 4 * B * s;
    c2 = 6 * B * s^2 - A;
    c1 = 4 * B * s^3 - 2 * A * s;
    c0 = -1;

    r = roots([c4; c3; c2; c1; c0]);
    periods(x) = min(abs(r));
end;