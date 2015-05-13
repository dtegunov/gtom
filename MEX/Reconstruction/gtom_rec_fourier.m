function [ volume, volumepsf ] = gtom_rec_fourier( projections, weights, angles, shifts )

if size(angles, 1) ~= 3
    angles = angles';
    if size(angles, 1) ~= 3
        error 'Angles must contain 3 values per column.';
    end;
end;

[ volume, volumepsf ] = cgtom_rec_fourier(projections, weights, angles, shifts);

end

