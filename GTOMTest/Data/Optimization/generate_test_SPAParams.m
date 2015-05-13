vol = zeros(32, 32, 32);
vol(1:2:end-1,1:2:end-1,:) = 1;
vol(1:2:end-1,:,1:2:end-1) = 1;
vol(:,1:2:end-1,1:2:end-1) = 1;

vol = tom_emread('4CR2.em');
vol = vol.Value;
vol = tom_cut_out(vol,'center',[96 96 96]);
vol = tom_rescale(vol, [32 32 32]);
vol = tom_rescale(vol, [64 64 64]);

% vol = padarray(vol, [16 16 16], 0);

nimages = 10;

% angles = [(0:nimages-1).*20; ones(1,nimages).*10; -(0:nimages-1).*20]./180.*pi;
angles = [zeros(1,nimages); 0:9; zeros(1,nimages)]./180.*pi;
shifts = zeros(2, nimages);

proj = gtom_proj_forward_raytrace(vol, angles, shifts, ones(size(shifts)));

fid = fopen('Input_SPAParams_images.bin','W');
fwrite(fid, proj, 'single');
fclose(fid);

% fid = fopen('Input_SPAParams_psf.bin','W');
% fwrite(fid, projpsf, 'single');
% fclose(fid);