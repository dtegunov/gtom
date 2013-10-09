%Case 1:
dimx=8;
indata = single(rand(dimx,1));
x=0:dimx;
x_1=0:0.001:dimx-0.001;
outdata = interp1(x,indata,x_1,'spline');
fid = fopen('Input_Spline1d_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Spline1d_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);