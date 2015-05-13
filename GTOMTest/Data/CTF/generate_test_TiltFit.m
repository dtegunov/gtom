%Case 1:
% image = tom_mrcread('tilt25.mrc');
% image = image.Value;
% stack = tom_mrcread('L3Tomo3_plus.st');
% stack = stack.Value;
for z=1:size(stack,3)
    fid = fopen(['Input_TiltFitMinus' num2str(z-1) '.bin'],'W');
    fwrite(fid,stack(:,:,z),'single');
    fclose(fid);
end;