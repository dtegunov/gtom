function [ transformed ] = gtom_rotate( image, angles )

transformed = cgtom_rotate(image, angles./180.*pi);

end

