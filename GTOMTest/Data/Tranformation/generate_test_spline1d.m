%Case 1:
dimx=4;
%indata = single(rand(dimx,1));
indata = single(0:dimx-1);
%indata = padarray(indata,[0 1],'replicate');
x=0:1:dimx-1;
x_1=0:0.001:dimx-1-0.001;
outdata = interp1(x,indata,x_1,'pchip');
fid = fopen('Input_Spline1d_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Spline1d_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);