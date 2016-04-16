function [ ccf ] = gtom_pick( map, ref, ctf, doctf, mask, ismaskcircular, lowpassfreq, anglestep )

ccf = cgtom_pick(single(map), single(ref), single(ctf), single(doctf), single(mask), single(ismaskcircular), single(lowpassfreq), anglestep / 180 * pi);

end