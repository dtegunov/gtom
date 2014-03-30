% %Case 1:
% dimx=4;
% dimy=4;
% dimz=4;
% %indata = tom_spheremask(single(ones(dimx, dimy)), 3, 0);
% indata = single(zeros(dimx, dimy, dimz));
% %indata = tom_spheremask(indata, dimx/2-3, 1);
% indata(3,3,3) = 1;
% angles = tom_av2_equal_angular_spacing([0 90],[0 359.9],10,'xmipp_doc');
% %angles = [0, 0; 0, 90; 90, 90];
% outdata = [];
% for a=1:size(angles,1)
%     outdata(:,:,a) = tom_proj3d2c(indata,angles(a,:));
% end;
% fid = fopen('Input_ARTProj_1.bin','W');
% fwrite(fid,outdata,'single');
% fclose(fid);
% fid = fopen('Input_ARTAngles_1.bin','W');
% fwrite(fid,(angles./(180/pi))','single');
% fclose(fid);
% fid = fopen('Output_ART_1.bin','W');
% fwrite(fid,indata,'single');
% fclose(fid);

%Case 2:
rng(123);
dimx=16;
dimy=16;
dimz=16;
%indata = tom_spheremask(single(ones(dimx, dimy)), 3, 0);
indata = single(rand(dimx/4,dimy/4,dimz/4));
indata = tom_rescale(indata,[dimx, dimy, dimz]);
indata = tom_spheremask(indata, dimx/2-3, 1);
angles = tom_av2_equal_angular_spacing([0 90],[0 360],1,'xmipp_doc');
outdata = [];
for a=1:size(angles,1)
    outdata(:,:,a) = tom_proj3d2c(indata,angles(a,:));
end;
fid = fopen('Input_ARTProj_2.bin','W');
fwrite(fid,outdata,'single');
fclose(fid);
fid = fopen('Input_ARTAngles_2.bin','W');
fwrite(fid,(angles./(180/pi))','single');
fclose(fid);
fid = fopen('Output_ART_2.bin','W');
fwrite(fid,indata,'single');
fclose(fid);