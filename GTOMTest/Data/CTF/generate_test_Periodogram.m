%Case 1:
image = tom_mrcread('stack_02-Nov-2013_21-50-06.dat.2-1.mrc');
image = image.Value;
fid = fopen('Input_Periodogram.bin','W');
fwrite(fid,image,'single');
fclose(fid);

gram = tom_calc_periodogram_parallel(image, 512, 0, 32);
fid = fopen('Output_Periodogram.bin','W');
fwrite(fid,gram,'single');
fclose(fid);
