% 1D
for i=9:14
   dim = 2^i;
   indata = single(rand(dim,1));
   outdata = complexinterleaved(hermitiansymmetrytrim(fft(indata)));
   fid = fopen(strcat('Input_1D_',int2str(dim),'.bin'),'W');
   fwrite(fid,indata,'single');
   fclose(fid);
   fid = fopen(strcat('Output_1D_',int2str(dim),'.bin'),'W');
   fwrite(fid,outdata,'single');
   fclose(fid);
end
% 2D
for i=9:13
   dim = 2^i;
   indata = single(rand(dim,dim));
   outdata = complexinterleaved(hermitiansymmetrytrim(fft2(indata)));
   fid = fopen(strcat('Input_2D_',int2str(dim),'.bin'),'W');
   fwrite(fid,indata,'single');
   fclose(fid);
   fid = fopen(strcat('Output_2D_',int2str(dim),'.bin'),'W');
   fwrite(fid,outdata,'single');
   fclose(fid);
end
% 3D
for i=6:9
   dim = 2^i;
   indata = single(rand(dim,dim,dim));
   outdata = complexinterleaved(hermitiansymmetrytrim(fftn(indata)));
   fid = fopen(strcat('Input_3D_',int2str(dim),'.bin'),'W');
   fwrite(fid,indata,'single');
   fclose(fid);
   fid = fopen(strcat('Output_3D_',int2str(dim),'.bin'),'W');
   fwrite(fid,outdata,'single');
   fclose(fid);
end