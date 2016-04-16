function [ projections ] = gtom_proj_forward_fourier( volume, angles )

if size(angles,1) ~= 3
    angles = angles';
    if size(angles,1) ~= 3
        error('Angles must have 3 values per column.');
    end;
end;

angles = angles./180.*pi;

[ projections ] = cgtom_proj_forward_fourier(volume, angles);

end