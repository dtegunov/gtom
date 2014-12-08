rng(121);
vsize = [64 64 64];
vsize = [64 64 16];
angles = -60:2:60;
angles = angles';
angles = angles./180.*pi;
% angles = zeros(21,1);
maxshift = 0;
maxrot = 0;


vdouble = zeros(256, 256, 32);
% vdouble = zeros(128, 128, 128);

for i=1:30
    ball = tom_spheremask(ones(size(vdouble)), rand(1)^2*60, rand(1)^2*0, [rand(1)*224+16, rand(1)*224+16, rand(1)*16+8]).*rand(1);
    vdouble = vdouble + ball;
end;

% ball = tom_spheremask(ones(size(vdouble)), 30, 2, [65 65 65]);
% vdouble = vdouble + ball;

% vdouble = tom_emread('rubisco.em');
% vdouble = vdouble.Value;
% vdouble = tom_rescale(vdouble,[64 64 64]);


vdouble = tom_norm(vdouble, 'mean0+1std');
noise = tom_rescale(rand(size(vdouble)./8), size(vdouble));
noise = tom_norm(noise, 'mean0+1std');
vdouble = vdouble + noise.*0.8;
vdouble = vdouble - min(vdouble(:));
vdouble = padarray(vdouble, [0 0 (size(vdouble,1)-size(vdouble,3))/2], 0);

vdouble = tom_rescale(vdouble,[128 128 128]);

%tom_emwrite('DummyVolume1.em', v);

proj = [];

proj = ForwardProj(vdouble, [zeros(1, size(angles,1)); angles'; zeros(1, size(angles,1))]);
% for a=1:size(angles,1)
%     proj(:,:,a) = tom_proj3d(single(vdouble),[0 angles(a)/pi*180]);
% end;
proj = tom_cut_out(proj,'center',[64 64 size(angles,1)]);
% proj = proj-min(proj(:));
% proj = proj./max(proj(:));
for a=1:size(angles,1)
%     proj(:,:,a) = tom_BandpassNeat(proj(:,:,a),0,8);
end;
% for a=1:size(angles,1)
%     proj(:,:,a) = reshape(1:(64*64),64,64);
% end;

maskproj = zeros(size(proj));
for a=1:size(angles,1)
%     depthmap = boxdepth([128 128], [128 128 32], [0; angles(a,1); 0]./pi.*180);
%     depthmapall = boxdepth([128 128], [128 256 32], [0; angles(a,1); 0]./pi.*180);
%     depthmap = depthmap./depthmapall;
%     depthmap = tom_shift(padarray(depthmap,[0 32],0), [0, (rand(1)-0.5)*8]);
%     depthmap = tom_cut_out(depthmap,'center',[128 128]);
%     maskproj(:,:,a) = depthmap;
end;

shifts = zeros(size(angles,1), 2);
rotations = zeros(size(angles,1), 1);

% for a=1:size(angles,1);
%     shifts(a,:) = [(rand(1) - 0.5) * 2 * maxshift, (rand(1) - 0.5) * 2 * maxshift];
%     rotations(a) = (rand(1) - 0.5) * 2 * maxrot;
%     
%     %proj(:,:,a) = imrotate(proj(:,:,a), rotations(a),'bicubic','crop');    
%     proj(:,:,a) = tom_shift(proj(:,:,a), shifts(a,:));
% end;

%proj=padarray(proj,[32 32 0],'symmetric');
fid = fopen('SIRTvsWBP.bin', 'W');
fwrite(fid,proj,'single');
fclose(fid);

vdouble = tom_cut_out(vdouble,'center',[64 64 16]);
fid = fopen('SIRTvsWBPvolume.bin', 'W');
fwrite(fid,vdouble,'single');
fclose(fid);