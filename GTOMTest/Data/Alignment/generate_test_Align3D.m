rng(123);
dim = 32;

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

target = indata;
indata = tom_rescale(indata,[128, 128, 128]);

fid = fopen('Input_Align3DData_1.bin', 'W');
fid2 = fopen('Input_Align3DDataPSF_1.bin', 'W');
for i=0:50
    fwrite(fid, tom_rescale(tom_rotate(indata, [-i i*2 i]), [32 32 32]), 'single');
    fwrite(fid2, psf, 'single');
end;
fclose(fid);
fclose(fid2);

fid = fopen('Input_Align3DTarget_1.bin', 'W');
fid2 = fopen('Input_Align3DTargetPSF_1.bin', 'W');
fwrite(fid, target, 'single');
fwrite(fid2, psf, 'single');
fclose(fid);
fclose(fid2);