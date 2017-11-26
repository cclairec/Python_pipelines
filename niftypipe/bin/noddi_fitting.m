function [neural_density, orientation_dispersion_index, csf_volume_fraction, objective_function, kappa_concentration, error, fibre_orientations] = noddi_fitting(dwis, mask, bvals, bvecs, b0threshold, fname, varargin)

disp('******************************');
disp('   NODDI fitting pipeline     ');
disp('******************************');

%default settings
set_scaling=0;
set_tissuetype=0;
%set_multistart=0;
poolsize=1;

%get all vargin option
c=1;
while (~isempty(varargin) & c<numel(varargin))
    option=varargin{c};
    switch(option)
        case 'noisescaling'
            set_scaling=1;
            scaling_factor=varargin{c+1};
            c=c+1;
        case 'tissuetype'
            set_tissuetype=1;
            tissuetype=varargin{c+1};
            c=c+1;
        case 'matlabpoolsize'
            poolsize=varargin{c+1};
            c=c+1;
    %  MULTISTART not implemented yet
    %            case {'multistart'}
    %                set_multistart=1;
    %                   ....
    %             
    otherwise         
        disp(['Invalid optional argument - skipping option eval', option]);        
        break;
    end % switch
    c=c+1;   
end % while
   

[~, ~, dwis_ext] = fileparts(dwis);
[~, ~, mask_ext] = fileparts(mask);

if (strcmp(dwis_ext,'.gz'))
    disp('nifti gzip detected. gunzip DWIs');
    dwis = gunzip(dwis, pwd);
    dwis = dwis{1};
end
if (strcmp(mask_ext,'.gz'))
    disp('nifti gzip detected. gunzip mask');
    mask = gunzip(mask, pwd);
    mask = mask{1};
end

roi = 'NODDI_roi.mat';

disp('Creating ROI...');
CreateROI(dwis, mask, roi)

disp('Creating Protocol...');
protocol = FSL2Protocol(bvals, bvecs, b0threshold);

disp('Creating Model...');
noddi = MakeModel('WatsonSHStickTortIsoV_B0');

disp('Setting Custom Parameters...');
if (set_scaling~=0)
    noddi.sigma.scaling = scaling_factor;    
end

if (set_tissuetype~=0)
    noddi.tissuetype = tissuetype;    
end

output_params = 'FittedParams.mat';

disp('Fitting NODDI model...');
if (poolsize == 1)
  batch_fitting_single(roi, protocol, noddi, output_params);
else
  batch_fitting(roi, protocol, noddi, output_params, poolsize);
end

disp('Saving outputs...');
SaveParamsAsNIfTI(output_params, roi, mask, fname)

neural_density = strcat(fname, '_ficvf.nii');
orientation_dispersion_index = strcat(fname, '_odi.nii');
csf_volume_fraction = strcat(fname, '_fiso.nii');
objective_function = strcat(fname, '_fmin.nii');
kappa_concentration = strcat(fname, '_kappa.nii');
error = strcat(fname, '_error_code.nii');

fibre_orientations{1} = strcat(fname, '_fibredirs_xvec.nii');
fibre_orientations{2} = strcat(fname, '_fibredirs_yvec.nii');
fibre_orientations{3} = strcat(fname, '_fibredirs_zvec.nii');

disp('******************************');
disp(' NODDI fitting pipeline: done');
disp('******************************');
