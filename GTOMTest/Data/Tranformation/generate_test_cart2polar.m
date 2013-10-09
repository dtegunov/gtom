%Case 1:
indata = single(rand(16));
outdata = tom_cart2polar(indata);
fid = fopen('Input_Cart2Polar_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Cart2Polar_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 2:
indata = single(rand(15));
outdata = tom_cart2polar(indata);
fid = fopen('Input_Cart2Polar_2.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Cart2Polar_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);