%Case 1:
dimx=4;
dimy=3;
dimz=5;
indata = single(rand(dimx,dimy,dimz)).*rand(1);
outdata = tom_norm(indata, 'mean0+1std');
fid = fopen('Input_Norm_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Norm_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

clear indata;
clear outdata;

%Case 2:
dimx=128;
dimy=128;
dimz=4000;
for i=1:dimz
    indata(:,:,i) = single(rand(dimx,dimy)).*rand(1);
    outdata(:,:,i) = tom_norm(indata(:,:,i), 'mean0+1std');
end;
fid = fopen('Input_Norm_2.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Norm_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);