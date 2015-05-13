%Case 1:
[phase amplitude decay E_k E_i_k E_e_k]=tom_ctf2(3.0e-6, 0, 0, 1.35e-10,300e3,[1024 1024],2.0e-3,0,0.00,2.2e-3,0.0);
image = E_e_k.*E_i_k.*(sqrt(1 - amplitude * 0.00).*phase + amplitude * 0.00);
% image = tom_cut_out(image, 'center', [128 128]);
image = image(1:end/2+1,:);
fid = fopen('Input_RadialAverage.bin','W');
fwrite(fid,image,'single');
fclose(fid);