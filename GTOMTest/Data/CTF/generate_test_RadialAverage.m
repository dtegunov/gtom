%Case 1:
[phase amplitude decay E_k E_i_k E_e_k]=tom_ctf2(-1.0e-6, 0.5e-6, 45, 1.35e-10,300e3,[1024 1024],2.0e-3,0,0.00,2.2e-3,0.0);
image = 0.07.*amplitude - sqrt(1-0.07^2).*phase;
% image = tom_cut_out(image, 'center', [128 128]);
image = image(1:end/2+1,:);
fid = fopen('Input_RadialAverage1.bin','W');
fwrite(fid,image,'single');
fclose(fid);

%Case 2:
fid = fopen('Input_RadialAverage2.bin','W');
for z=0.0:0.01:0.09
    ps=tom_ctf2d(512, 1e-10, 0.1e-10, 0, 300e3, 2e-3, -(1+z)*1e-6, 0.2e-6, 45/180*pi, 0.07, 0, 0);
    fwrite(fid, ps, 'single');
end;
fclose(fid);