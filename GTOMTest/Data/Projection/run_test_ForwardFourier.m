vol = zeros(32, 32, 32);
vol(17,:,:) = 1;
vol(:,17,:) = 1;

% vol = padarray(vol, [16 16 16], 0);

angles = [0; 0; 20/180*pi];
shifts = [0; 0];

[proj, projpsf] = gtom_proj_forward_fourier(vol, ones(size(vol)), angles, shifts);