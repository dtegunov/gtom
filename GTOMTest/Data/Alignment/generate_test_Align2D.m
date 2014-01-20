rng(123);
dim = 32;
count = 20;
maxtrans = 5;
maxrot = 0;

targets = zeros(dim * 2, dim * 2, count);
for t=1:count
    targets(:,:,t) = tom_rescale(rand(dim / 1), [dim * 2, dim * 2])...
                     + tom_rescale(rand(dim / 2), [dim * 2, dim * 2])...
                     + tom_rescale(rand(dim / 4), [dim * 2, dim * 2]);
end;

fid = fopen('Input_Align2DTargets_1.bin', 'W');
for t=1:count
    fwrite(fid, tom_cut_out(targets(:,:,t), 'center', [dim dim]), 'single');
end;
fclose(fid);

transformed = targets;
trans = zeros(count, 2);
rot = zeros(count, 1);
for t=1:count
    trans(t, :) = (rand(1, 2) - 0.5) .* (maxtrans * 2);
    rot(t) = (rand(1) - 0.5) .* (maxrot * 2);
    transformed(:,:,t) = tom_rotate(transformed(:,:,t), rot(t));
    transformed(:,:,t) = tom_shift(transformed(:,:,t), trans(t, :));
end;

fid = fopen('Input_Align2DData_1.bin', 'W');
fidparams = fopen('Input_Align2DParams_1.bin', 'W');
for t=1:count
    fwrite(fid, tom_cut_out(transformed(:,:,t), 'center', [dim dim]), 'single');
    fprintf(fidparams, '%f\t%f\t%f\n', [trans(t, :) rot(t)]);
end;
fclose(fid);
fclose(fidparams);