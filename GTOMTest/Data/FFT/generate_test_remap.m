%Case 1:
dimx=4;
dimy=3;
dimz=5;
indata = reshape(ndgrid(1:(dimx*dimy*dimz)),dimx,dimy,dimz);
outdata = ifftshift(indata);
outdata = outdata(1:(floor(dimx/2)+1),:,:);
fid = fopen('Input_Remap_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Remap_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 2:
dimx=8;
dimy=8;
dimz=8;
indata = reshape(ndgrid(1:(dimx*dimy*dimz)),dimx,dimy,dimz);
outdata = ifftshift(indata);
outdata = outdata(1:(floor(dimx/2)+1),:,:);
fid = fopen('Input_Remap_2.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Remap_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 3:
dimx=8;
dimy=8;
<<<<<<< HEAD
dimz=8;
indata = reshape(ndgrid(1:(dimx*dimy*dimz)),dimx,dimy,dimz);
outdata = ifftshift(indata);
indata = vertcat(indata((floor(dimx/2)+1):dimx,:,:),indata(1,:,:));
outdata = outdata(1:(floor(dimx/2)+1),:,:);
=======
dimz=1;
indata = reshape(ndgrid(1:(dimx*dimy*dimz)),dimx,dimy,dimz);
outdata = fftshift(indata);
>>>>>>> 77ee24d2625debc91b0cc36e1f8bdad326e7221b
fid = fopen('Input_Remap_3.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Remap_3.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

<<<<<<< HEAD
%Case 4:
dimx=8;
dimy=8;
dimz=8;
indata = reshape(ndgrid(1:(dimx*dimy*dimz)),dimx,dimy,dimz);
outdata = fftshift(indata);
indata = vertcat(indata((floor(dimx/2)+1):dimx,:,:),indata(1,:,:));
outdata = outdata(1:(floor(dimx/2)+1),:,:);
=======
%Case 3:
dimx=7;
dimy=7;
dimz=7;
indata = reshape(ndgrid(1:(dimx*dimy*dimz)),dimx,dimy,dimz);
outdata = fftshift(indata);
>>>>>>> 77ee24d2625debc91b0cc36e1f8bdad326e7221b
fid = fopen('Input_Remap_4.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Remap_4.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);