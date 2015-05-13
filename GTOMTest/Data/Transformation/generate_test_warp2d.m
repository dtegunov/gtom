rng(123);

%Case 1:
dimx=1024;
dimy=1024;
indata = rand(dimx, dimy);
indata = tom_bandpass(indata, 0, 128);
%indata(1:end/2, 1:end/2) = 1;
%indata(end/2+1:end, end/2+1:end) = 1;
% indata(1:4:end,:) = 1;
% indata(:,1:4:end) = 1;

fid = fopen('Input_Warp2D.bin','W');
fwrite(fid,indata,'single');
fclose(fid);