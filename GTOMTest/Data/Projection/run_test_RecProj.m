image = zeros(32, 32);
image(17,:) = 1;
image(:,17) = 1;

% image = padarray(image, [16 16], 0);

angles = [0; 0/180*pi; 0/180*pi];
shifts = [5; 0];

[vol, volpsf] = gtom_rec_fourier(image, ones(size(image)), angles, shifts);

[proj, projpsf] = gtom_proj_forward_fourier(vol, volpsf, angles, shifts);

[vol2, volpsf2] = gtom_rec_fourier(proj, projpsf, angles, shifts);

[proj2, projpsf2] = gtom_proj_forward_fourier(vol2, volpsf2, angles, shifts);

proj = tom_cut_out(proj, 'center', [32 32]);
proj2 = tom_cut_out(proj2, 'center', [32 32]);

% projpsf = [projpsf; flipud(projpsf(2:16,:))];
% projpsf2 = [projpsf2; flipud(projpsf2(2:16,:))];
projpsf = hermitiansymmetrypad(projpsf, [32 32]);
projpsf2 = hermitiansymmetrypad(projpsf2, [32 32]);

imageft = fftshift(fftn(image));

imageft1 = imageft.*projpsf;
image1 = real(ifftn(ifftshift(imageft1)));
image1 = tom_cut_out(image1, 'center', [32 32]);

imageft2 = imageft.*projpsf2;
image2 = real(ifftn(ifftshift(imageft2)));
image2 = tom_cut_out(image2, 'center', [32 32]);

disp(tom_realcc(image1, proj, tom_spheremask(ones(32), 8)));
disp(tom_realcc(image2, proj2, tom_spheremask(ones(32), 8)));