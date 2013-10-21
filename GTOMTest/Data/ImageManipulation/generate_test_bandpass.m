%Case 1:
dimx=256;
indata = single(rand(dimx,1));
outdata = tom_bandpass(indata,5,126,10);
fid = fopen('Input_Bandpass_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Bandpass_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 2:
dimx=256;
dimy=255;
indata = single(rand(dimx,dimy));
outdata = tom_bandpass(indata,5,126,10);
fid = fopen('Input_Bandpass_2.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Bandpass_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 3:
dimx=256;
dimy=255;
dimz=66;
indata = single(rand(dimx,dimy,dimz));
outdata = tom_bandpass(indata,5,28,6);
fid = fopen('Input_Bandpass_3.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Bandpass_3.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 4:
dimx=1855;
dimy=1855;
indata = single(rand(dimx,dimy));
outdata = tom_bandpass(indata,5,1855/5,20);
fid = fopen('Input_Bandpass_4.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Bandpass_4.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);