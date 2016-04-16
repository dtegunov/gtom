% mex 'Projection\cgtom_proj_raysum.cu' Helper.cu -g;
mex 'Projection\cgtom_proj_forward_fourier.cu' Helper.cu;
% mex 'Projection\cgtom_proj_forward_raytrace.cu' Helper.cu;
% mex 'Projection\cgtom_proj_backward.cu' Helper.cu;
% mex 'Projection\cgtom_proj_weighting.cu' Helper.cu;