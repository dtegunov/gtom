%Case 1:
dimx=32;
dimy=32;
indata = single(zeros(dimx, dimy));
indata(:,17) = 1;
indata(17,:) = 1;

indata = tom_norm(indata, 'mean0+1std');
stack = [];
stack(:,:,1) = indata;
stack(:,:,2) = tom_norm(imrotate(indata, 0, 'bicubic', 'crop'), 'mean0+1std');
stack(:,:,3) = tom_norm(imrotate(indata, 33, 'bicubic', 'crop'), 'mean0+1std');

fid = fopen('Input_SimMatrix.bin','W');
fwrite(fid, stack, 'single');
fclose(fid);