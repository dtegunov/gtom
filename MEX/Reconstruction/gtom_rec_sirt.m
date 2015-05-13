function [ volume ] = gtom_rec_sirt( projections, volumesize, angles, shifts, scales, iterations )

if size(angles, 1) ~= 3
    angles = angles';
    if size(angles, 1) ~= 3
        error 'Angles must contain 3 values per column.';
    end;
end;

if size(shifts, 1) ~= 2
    shifts = shifts';
    if size(shifts, 1) ~= 2
        error 'Shifts must contain 2 values per column.';
    end;
end;

if size(scales, 1) ~= 2
    scales = scales';
    if size(scales, 1) ~= 2
        error 'Scales must contain 2 values per column.';
    end;
end;

volume = cgtom_rec_sirt(projections, volumesize, angles, shifts, scales, iterations);

end

