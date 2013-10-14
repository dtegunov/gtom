%Case 1:

dimx = 16;
dimy = 16;
indata = single(rand(dimx,dimy));
debug_outdata=tom_rescale(indata,[8 8]);
outdata = complexinterleaved(hermitiansymmetrytrim(debug_outdata));
fid = fopen('Input_Resize_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Resize_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 2:

dimx = 4;
dimy = 1;
indata = single(rand(dimx,dimy));
debug_outdata=tom_rescale(indata,[8 1]);
outdata = complexinterleaved(hermitiansymmetrytrim(debug_outdata));
fid = fopen('Input_Resize_2.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Resize_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);