%Case 1:
dimx=32;
dimy=32;
dimz=32;
window=16;
indata1 = single(rand(dimx,dimy,dimz));
indata2 = tom_bandpass(indata1, 0, 10, 1);
resolution = zeros(dimx,dimy,dimz);

padin1 = padarray(indata1,window/2,'symmetric');
padin2 = padarray(indata2,window/2,'symmetric');

outdata = tom_fsc(indata1, indata2, 16, 1, 0);

fid = fopen('Input1_LocalFSC_1.bin','W');
fwrite(fid,indata1,'single');
fclose(fid);
fid = fopen('Input2_LocalFSC_1.bin','W');
fwrite(fid,indata2,'single');
fclose(fid);
fid = fopen('Output_LocalFSC_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 2:
dimx=280;
dimy=280;
dimz=280;
window=40;
indata1 = vol1;
indata2 = vol2;
outdata = tom_fsc(indata1, indata2, 40, 1, 0);

fid = fopen('Input1_LocalFSC_2.bin','W');
fwrite(fid,indata1,'single');
fclose(fid);
fid = fopen('Input2_LocalFSC_2.bin','W');
fwrite(fid,indata2,'single');
fclose(fid);
fid = fopen('Output_LocalFSC_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);