%Case 1:
dimx=8;
dimy=8;
dimz=8;
indata = tom_spheremask(single(ones(dimx, dimy)), 3, 0);
outdata = single(zeros(dimx, dimy, dimz));
tom_backproj3d(outdata, indata, 0, 0, [0 0 0]);
fid = fopen('Input_Backward_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Backward_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 2:
dimx=8;
dimy=8;
dimz=8;
indata = tom_spheremask(single(ones(dimx, dimy)), 3, 0);
indata(1,1) = 3;
indata(3,3) = 2;
outdata = single(zeros(dimx, dimy, dimz));
tom_backproj3d(outdata, indata, 90, 0, [0 0 0]);
fid = fopen('Input_Backward_2.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Backward_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 3:
dimx=8;
dimy=8;
dimz=8;
indata = tom_spheremask(single(ones(dimx, dimy)), 3, 0);
indata(1,1) = 3;
indata(3,3) = 2;
outdata = single(zeros(dimx, dimy, dimz));
tom_backproj3d(outdata, indata, 0, 90, [0 0 0]);
fid = fopen('Input_Backward_3.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Backward_3.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 4:
dimx=8;
dimy=8;
dimz=8;
indata = tom_spheremask(single(ones(dimx, dimy)), 3, 0);
indata(1,1) = 3;
indata(3,3) = 2;
outdata = single(zeros(dimx, dimy, dimz));
tom_backproj3d(outdata, indata, 30, 45, [0 0 0]);
fid = fopen('Input_Backward_4.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Backward_4.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);