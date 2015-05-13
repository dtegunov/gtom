function [ projections, projectionspsf ] = gtom_proj_forward_fourier( volume, volumepsf, angles, shifts )

if size(angles,1) ~= 3
    angles = angles';
    if size(angles,1) ~= 3
        error('Angles must have 3 values per column.');
    end;
end;

if size(shifts,1) ~= 2
    shifts = shifts';
    if size(shifts,1) ~= 2
        error('Shifts must have 2 values per column.');
    end;
end;

[ projections, projectionspsf ] = cgtom_proj_forward_fourier(volume, volumepsf, angles, shifts);

end