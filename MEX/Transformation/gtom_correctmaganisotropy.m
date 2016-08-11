function [ corrected ] = gtom_correctmaganisotropy( images, majorscale, minorscale, angle, supersample )

corrected = cgtom_correctmaganisotropy(images, [majorscale, minorscale, angle / 180 * pi], supersample);

end

