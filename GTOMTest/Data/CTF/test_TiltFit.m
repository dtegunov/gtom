stack = tom_mrcread('L3Tomo3_plus.st');
stack = stack.Value;

mdoc = tom_read_mdoc('L3Tomo3_plus_orig.st.mdoc');

stagetilts = [mdoc.rot; mdoc.theta]./180.*pi;
defocus = -mdoc.defocus(1) * 1e-6;
defocusbracket = [-3e-6, 3e-6, 0.1e-6]';
singleparams = gtom_ctfparams('pixelsize', 3.42e-10, 'defocus', defocus, 'decayspread', 0);
startparams = repmat(singleparams, [1, size(stack, 3)]);

[defoci, specimentilt] = gtom_ctf_tiltfit(stack, defocusbracket, [256; 24; 72], 15/180*pi, stagetilts, startparams);