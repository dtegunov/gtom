%Case 1:
dimx=16;
dimy=15;
dimz=1;
indata = single(rand(dimx,dimy,dimz));
indata(4,6)=1000;
indata(10,10)=1000;
outdata = tom_xraycorrect2(indata);
fid = fopen('Input_Xray_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Xray_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);