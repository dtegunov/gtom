[phase amplitude decay E_k E_i_k E_e_k]=tom_ctf2(-1.0e-6, 0, 0, 1.35e-10,300e3,[1024 1024],2.0e-3,0,0.00,2.2e-3,0.0);
image = E_e_k.*E_i_k.*(sqrt(1 - amplitude * 0.00).*phase + amplitude * 0.00);
spec1 = image(513,513:end)';

[phase amplitude decay E_k E_i_k E_e_k]=tom_ctf2(-3.0e-6, 0, 0, 1.35e-10,300e3,[1024 1024],2.0e-3,0,0.00,2.2e-3,0.0);
image = E_e_k.*E_i_k.*(sqrt(1 - amplitude * 0.00).*phase + amplitude * 0.00);
spec2 = image(513,513:end)';

fid = fopen('Input_Accumulate.bin','W');
fwrite(fid,[spec1, spec2],'single');
fclose(fid);