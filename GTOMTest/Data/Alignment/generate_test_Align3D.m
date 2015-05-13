rng(123);
dim = 32;
nvols = 200;

indata = zeros(32, 32, 32);
indata(17,17,:) = 1;
indata(:,17,17) = 1;
indata(17,:,17) = 1;
indata = tom_emread('mediator.em');
indata = indata.Value;

psf = ones(16, 16, 8);
psf = padarray(psf, [8 8 12], 0);
psf = psf(1:17,:,:);
psf = ones(32,32,32);
psf = tom_rescale(tom_av3_wedge(ones(128, 128, 128), -60, 60), [32 32 32]);

target = indata;
indata = tom_rescale(indata,[128, 128, 128]);
instack = zeros(32, 32, 32*nvols);

fid = fopen('Input_Align3DData_1.bin', 'W');
fid2 = fopen('Input_Align3DDataPSF_1.bin', 'W');
for i=1:nvols
    rotated = tom_rescale(tom_rotate(tom_snr(indata, 0.1), [rand(1)*360 rand(1)*360 rand(1)*180]), [32 32 32]);
    instack(:, :, (i-1)*32+1:i*32) = rotated;
    fwrite(fid, rotated, 'single');
    fwrite(fid2, psf(1:17,:,:), 'single');
end;
fclose(fid);
fclose(fid2);

fid = fopen('Input_Align3DTarget_1.bin', 'W');
fid2 = fopen('Input_Align3DTargetPSF_1.bin', 'W');
fwrite(fid, target, 'single');
fwrite(fid2, psf, 'single');
fclose(fid);
fclose(fid2);