rng(123);
dim = 128;

indata1 = single(rand(dim/8,dim/8,dim/8));
indata1 = tom_rescale(indata1, [dim dim dim]);
indata2 = single(rand(dim/4,dim/4,dim/4));
indata2 = tom_rescale(indata2, [dim dim dim]);
indata3 = single(rand(dim/16,dim/16,dim/16));
indata3 = tom_rescale(indata3, [dim dim dim]);
indata = tom_spheremask(indata1+indata2+indata3, dim/2-3, 1);

target = tom_rotate(indata, [90, 45, 30]);

fid = fopen('Input_Align3DData_1.bin', 'W');
fwrite(fid, indata, 'single');
fclose(fid);

fid = fopen('Input_Align3DTarget_1.bin', 'W');
fwrite(fid, target, 'single');
fclose(fid);