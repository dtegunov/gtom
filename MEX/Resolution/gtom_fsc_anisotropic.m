function [ resmap, filter ] = gtom_fsc_anisotropic( map1, map2, coneangle, minshell, threshold )

[resmap, filter] = cgtom_fsc_anisotropic(fftshift(fftn(map1)), fftshift(fftn(map2)), coneangle, minshell, threshold);

end

