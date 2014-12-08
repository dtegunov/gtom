function [ reconstruction, samples ] = gtom_rec_fourier_add( volumeft, volumesamples, projections, weights, ctf, angles, volumesize )

if size(angles, 1) ~= 3 && size(angles, 2) == 3
    angles = angles';
elseif size(angles, 1) ~= 3
    error 'Angles must contain 3 values per column.';
end;

if size(ctf, 1) ~= 11 && size(ctf, 2) == 11
    ctf = ctf';
elseif size(ctf, 1) ~= 11
    error 'CTF must contain 11 values per column.';
end;

[reconstruction, samples] = cgtom_rec_fourier_add(volumeft, volumesamples, projections, weights, ctf, angles, volumesize);

end

