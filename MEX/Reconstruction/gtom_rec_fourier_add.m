function [ reconstruction, samples ] = gtom_rec_fourier_add( volumeft, volumesamples, projections, weights, angles, shifts )

if size(angles, 1) ~= 3 && size(angles, 2) == 3
    angles = angles';
elseif size(angles, 1) ~= 3
    error 'Angles must contain 3 values per column.';
end;

[reconstruction, samples] = cgtom_rec_fourier_add(volumeft, volumesamples, projections, weights, angles, shifts);

end

