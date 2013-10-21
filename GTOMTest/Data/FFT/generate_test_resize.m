%Case 1:
dimx = 16;
dimy = 17;
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
dimx = 16;
dimy = 17;
indata = single(rand(dimx,dimy));
debug_outdata=tom_rescale(indata,[7 8]);
outdata = complexinterleaved(hermitiansymmetrytrim(debug_outdata));
fid = fopen('Input_Resize_2.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Resize_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 3:
dimx = 16;
dimy = 16;
indata = single(rand(dimx,dimy));
debug_outdata=tom_rescale(indata,[7 7]);
outdata = complexinterleaved(hermitiansymmetrytrim(debug_outdata));
fid = fopen('Input_Resize_3.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Resize_3.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 4:
dimx = 16;
dimy = 16;
indata = single(rand(dimx,dimy));
debug_outdata=tom_rescale(indata,[7 8]);
outdata = complexinterleaved(hermitiansymmetrytrim(debug_outdata));
fid = fopen('Input_Resize_4.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Resize_4.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 5:
dimx = 8;
dimy = 8;
indata = single(rand(dimx,dimy));
debug_outdata=tom_rescale(indata,[16 16]);
outdata = complexinterleaved(hermitiansymmetrytrim(debug_outdata));
fid = fopen('Input_Resize_5.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Resize_5.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 6:
dimx = 7;
dimy = 8;
indata = single(rand(dimx,dimy));
debug_outdata=tom_rescale(indata,[16 16]);
outdata = complexinterleaved(hermitiansymmetrytrim(debug_outdata));
fid = fopen('Input_Resize_6.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Resize_6.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 7:
dimx = 8;
dimy = 8;
indata = single(rand(dimx,dimy));
debug_outdata=tom_rescale(indata,[16 17]);
outdata = complexinterleaved(hermitiansymmetrytrim(debug_outdata));
fid = fopen('Input_Resize_7.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Resize_7.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);

%Case 8:
dimx = 16;
dimy = 16;
indata = single(rand(dimx,dimy));
debug_outdata=tom_rescale(indata,[7 7]);
outdata = complexinterleaved((debug_outdata));
fid = fopen('Input_Resize_8.bin','W');
fwrite(fid,indata,'single');
fclose(fid);
fid = fopen('Output_Resize_8.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);