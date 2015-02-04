%Case 1:
dimx=8;
dimy=8;
delta=[1 2];
image1 = tom_norm(rand(dimx,dimy), 'mean0+1std');
image2 = tom_shift(image1, delta);
indata = tom_corr(image1, image2);
outdata = delta + [floor(dimx/2) floor(dimy/2)];
if(size(outdata,2)==2)
    outdata = [outdata 0];
end;
fid = fopen('Input_Peak_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Peak_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 2:
dimx=20;
dimy=20;
delta=[1.5 2.5];
image1 = tom_norm(rand(dimx,dimy), 'mean0+1std');
image2 = tom_shift(image1, delta);
indata = tom_corr(image1, image2);
outdata = delta + [floor(dimx/2) floor(dimy/2)];
if(size(outdata,2)==2)
    outdata = [outdata 0];
end;
fid = fopen('Input_Peak_2.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Peak_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);