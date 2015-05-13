dims = [32 32 32];
nimages = 1;
proj = zeros(dims(1), dims(2), nimages);

proj(17, :, :) = 1;
proj(:, 17, :) = 1;
proj = padarray(proj, [16 16], 0);

angles = [0; 20/180*pi; 20/180*pi];
shifts = [0; 0];

[rec, recpsf] = gtom_rec_fourier(proj, ones(size(proj)), angles, shifts);