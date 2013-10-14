%Case 1:
dimx = 15;
dimy = 15;
indata = single(rand(dimx,dimy));
debug_outdata=tom_rescale(indata,[7 7]);
outdata = complexinterleaved(hermitiansymmetrytrim(debug_outdata));
fid = fopen('Input_Resize_1.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Resize_1.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 2:
dimx = 8;
dimy = 8;
indata = single(rand(dimx,dimy));
debug_outdata=tom_rescale(indata,[16 16]);
outdata = complexinterleaved(hermitiansymmetrytrim(debug_outdata));
fid = fopen('Input_Resize_2.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Resize_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);