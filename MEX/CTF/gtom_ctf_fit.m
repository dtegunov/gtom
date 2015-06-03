function bestfit = gtom_ctf_fit( image, fitbrackets, kernelsize, startparams )

if size(fitbrackets, 1) ~= 3 && size(fitbrackets, 2) == 3
    fitbrackets = fitbrackets';
end;
if size(kernelsize, 1) ~= 3 && size(kernelsize, 2) == 3
    kernelsize = kernelsize';
end;
if size(startparams, 1) ~= 10 && size(startparams, 2) == 10
    startparams = startparams';
end;

bestfit = cgtom_ctf_fit(image, fitbrackets, kernelsize, startparams);

end