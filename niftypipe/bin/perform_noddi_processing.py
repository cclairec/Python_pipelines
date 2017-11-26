#! /usr/bin/env python

"""
=================
DTI: NODDI multi-shell fitting
=================

Introduction
============

This script, perform_noddi_groupwise.py, performs a shell by shell diffusion preprocessing, followed by a NODDI fitting
of the input diffusion weighted images from different shells::

    python perform_noddi_groupwise.py --help


Details
============

NODDI estimation using the MATLAB toolbox of the NODDI diffusion model fitting.

NODDI (neurite orientation dispersion and density imaging) is  a practical diffusion MRI technique for estimating
the microstructural complexity of dendrites and axons in vivo on clinical MRI scanners.

Documentation and installation of the MATLAB toolbox can be found here:
http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab

More information about the model can be found in the NeuroImage paper:
http://dx.doi.org/10.1016/j.neuroimage.2012.03.072

Perform Diffusion Model Fitting with all pre-processing steps for each provided shell (dwi-bval-bvec triplets).

    1) Motion / eddy-current correction via affine co-registration between DWIs towards an average B-Null image
    2) [optional] susceptibility correction via phase unwrapping as described in http://dx.doi.org/10.1016/j.media.2014.06.008
    3) Non-Linear Least square tensor fitting using NiftyFit, as described in [REF NEEDED, A. Melbourne et al.]

Most important inputs are the triplets of dwi-bval-bvec files (shells)

A T1 image is needed for reference. Optionally you can provide a T1 mask used to mask the outputs.

Provide field map magnitude and phase images to trigger the susceptibility correction for all shells

Values to use for the susceptibility parameters:
## DRC ## (--ped=-y --etd=2.46 --rot=34.56) and
## 1946 ## (--ped=-y --etd=2.46 --rot=25.92).
Note that these values are indicative.


.. topic:: Shell Outputs

    The pipeline provides a series of outputs related to the processing of diffusion images. It uses the input DWI
    image basename as a prefix for all outputs (here as <fn>)

    - **<fn>_b0.nii.gz**: The average B0 image after b0 groupwise registration (and susceptibility correction)
    - **<fn>_fa.nii.gz**: The Fractional Anisotropy map
    - **<fn>_mask.nii.gz**: The brain mask used for the outputs
    - **<fn>_md.nii.gz**: The mean diffusivity map
    - **<fn>_rgb.nii.gz**: The RGB colour-coded image. Colour-code indicates 1st eigenvector direction
    - **<fn>_tensor_residuals.nii.gz**: The root mean square error after least square tensor fitting
    - **<fn>_v1.nii.gz**: The 1st eigen vector xyz coordinates image
    - **<fn>_dwis.nii.gz**: The motion and eddy-current corrected (and susceptibility corrected) 4D DWI image
    - **<fn>_interslice_cc.png**: The quality control graph describing the interslice cross-correlation
    - **<fn>_matrix_rot.png**: The quality control graph describing the de-meaned subject's rotation per DWI
    - **<fn>_predicted_tensors.nii.gz**: <ask andrew>
    - **<fn>_t1_to_b0.txt**: The affine transformation matrix (with T1 as reference and DWI as floating)
    - **<fn>_tensors.nii.gz**: The fitted tensor image in a lower triangular fashion: xx-xy-yy-xz-yz-zz

    All output images are produced in the original DWI space. The user can resample these output images manually towards
    the structural T1 reference space using reg_resample::

        reg_resample -ref <T1_input> -flo <any_output> -res <output_in_input_grid>

    The default interpolation style is cubic. **Caution**: cubic interpolation can introduce non-physical (negative)
    image values. Prevent that by using linear interpolation `-inter 1`

.. topic:: NODDI outputs


List of the binaries necessary for this pipeline:
=================

* **FSL**: fslmaths, fslmerge, fslsplit
* **niftyreg**: reg_aladin, reg_transform, reg_f3d, reg_resample
* **niftyseg**: seg_maths
* **niftyfit**: fit_dwi, dwi_tool
* **susceptibility**: pm_scale, gen_fm, gen_pm, phase_unwrap
* **MATLAB**: and the matlab NODDI toolbox (http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab)


.. seealso::

    https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyFit
        NiftyFit is a software for model fitting and in particular diffusion.

    https://sourceforge.net/projects/niftyreg/
        NiftyReg: contains programs to perform rigid, affine and non-linear registration of nifti or analyse images


Pipeline Code
==================

Import the necessary python modules

"""

from argparse import (ArgumentParser, RawDescriptionHelpFormatter, SUPPRESS)
import os
import sys
import textwrap
from niftypipe.workflows.dmri.matlab_noddi_workflow import create_matlab_noddi_workflow
from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)

"""
Create the help message
"""

pipelineDescription = textwrap.dedent('''
    NODDI estimation using the MATLAB toolbox of the NODDI diffusion model fitting.

    NODDI (neurite orientation dispersion and density imaging) is  a practical diffusion MRI technique for estimating
    the microstructural complexity of dendrites and axons in vivo on clinical MRI scanners.

    Documentation and installation of the MATLAB toolbox can be found here:
    http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab

    More information about the model can be found in the NeuroImage paper:
    http://dx.doi.org/10.1016/j.neuroimage.2012.03.072

    Perform Diffusion Model Fitting with all pre-processing steps for each provided shell (dwi-bval-bvec triplets).
    1) Motion / eddy-current correction via affine co-registration between DWIs towards an average B-Null image
    2) [optional] susceptibility correction via phase unwrapping as described in http://dx.doi.org/10.1016/j.media.2014.06.008
    3) Non-Linear Least square tensor fitting using NiftyFit, as described in [REF NEEDED, A. Melbourne et al.]

    Most important inputs are the triplets of dwi-bval-bvec files (shells)

    A T1 image is needed for reference. Optionally you can provide a T1 mask used to mask the outputs.

    Provide field map magnitude and phase images to trigger the susceptibility correction for all shells

    Values to use for the susceptibility parameters:
    ## DRC ## (--ped=-y --etd=2.46 --rot=34.56) and
    ## 1946 ## (--ped=-y --etd=2.46 --rot=25.92).
    Note that these values are indicative.

    List of the binaries necessary for this pipeline:

    * FSL: fslmaths, fslmerge, fslsplit
    * niftyreg: reg_aladin, reg_transform, reg_f3d, reg_resample
    * niftyseg: seg_maths
    * niftyfit: fit_dwi, dwi_tool
    * susceptibility: pm_scale, gen_fm, gen_pm, phase_unwrap
    * MATLAB, and the matlab NODDI toolbox (http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab)
    ''')

"""
Create the arguments parser
"""

parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=pipelineDescription)
parser.add_argument('-i', '--dwis',
                    dest='dwis',
                    metavar='dwis',
                    nargs='+',
                    help='Diffusion Weighted Images in a 4D nifti file',
                    required=True)
parser.add_argument('-a', '--bvals',
                    dest='bvals',
                    metavar='bvals',
                    nargs='+',
                    help='bval file to be associated with the DWIs',
                    required=True)
parser.add_argument('-e', '--bvecs',
                    dest='bvecs',
                    metavar='bvecs',
                    nargs='+',
                    help='bvec file to be associated with the DWIs',
                    required=True)
parser.add_argument('-t', '--t1',
                    dest='t1',
                    metavar='t1',
                    help='T1 file to be associated with the DWIs',
                    required=True)
parser.add_argument('--t1_mask',
                    dest='t1_mask',
                    metavar='FILE',
                    help='T1 mask file associated with the input T1',
                    required=False)
parser.add_argument('-m', '--fieldmapmag',
                    dest='fieldmapmag',
                    metavar='fieldmapmag',
                    help='Field Map Magnitude image file to be associated with the DWIs',
                    required=False)
parser.add_argument('-p', '--fieldmapphase',
                    dest='fieldmapphase',
                    metavar='fieldmapphase',
                    help='Field Map Phase image file to be associated with the DWIs',
                    required=False)
parser.add_argument('-o', '--output_dir',
                    dest='output_dir',
                    type=str,
                    metavar='output_dir',
                    help='Output directory containing the registration result\n' +
                         'Default is a directory called results',
                    default=os.path.abspath('results'),
                    required=False)
parser.add_argument('--nsf',
                    dest='noise_scaling_factor', 
                    type=int,
                    metavar='noise_scaling_factor', 
                    help='Noise scaling factor value.\n' +
                    'Default is 100 or the value NOISE_SCALING_FACTOR environtment variable', 
                    required=False,
                    default=100)
parser.add_argument('--rot',
                    dest='rot',
                    type=float,
                    metavar='rot',
                    help='Diffusion Read-Out time used for susceptibility correction\n' +
                         'Default is 34.56',
                    default=34.56,
                    required=False)
parser.add_argument('--etd',
                    dest='etd',
                    type=float,
                    metavar='etd',
                    help='Echo Time difference used for susceptibility correction\n' +
                         'Default is 2.46',
                    default=2.46,
                    required=False)
parser.add_argument('--ped',
                    nargs='?',
                    const=None,
                    choices=[Q for x in ['x', 'y', 'z'] for Q in (x, '-' + x)],
                    dest='ped',
                    type=str,
                    metavar='ped',
                    help='Phase encoding direction used for susceptibility correction (x, y or z)\n' +
                         '--ped=val form must be used for -ve indices' +
                         'Default is the -y direction (-y)',
                    default='-y',
                    required=False)
parser.add_argument('-x', '-y', '-z',
                    dest='pedwarn',
                    help=SUPPRESS,
                    required=False,
                    action='store_true')
parser.add_argument('--matlabpoolsize',
                    dest='matlabpoolsize',
                    type=int,
                    metavar='matlabpoolsize',
                    help='Parallel processing pool size for matlab (NODDI fitting\n' +
                         'Default is 1 (don\'t use parallel)',
                    default=1,
                    required=False)
pgroup = parser.add_mutually_exclusive_group()
pgroup.add_argument('--with_eddy',
                    dest='with_eddy',
                    type=bool,
                    metavar='with_eddy',
                    help='Use the \'eddy\' tool for eddy correction (default)',
                    default=True,
                    required=False)
pgroup.add_argument('--no-with_eddy',
                    dest='with_eddy',
                    action='store_false',
                    help='Don\'t use the \'eddy\' tool for eddy correction (registration approach)',
                    required=False)



"""
Add default arguments in the parser
"""

default_parser_argument(parser)

"""
Parse the input arguments
"""

args = parser.parse_args()

if args.ped is None:
    print 'argument --ped: expected one argument, make sure to use --ped='
    sys.exit(1)
if args.pedwarn:
    print 'One of -x, -y or -z found, did you mean --ped=-x, --ped=-y --ped=-z?'
    sys.exit(1)

"""
Create the result directory if non existent
"""

result_dir = os.path.abspath(args.output_dir)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

mag = None
phase = None
if args.fieldmapmag is None or args.fieldmapphase is None:
    do_susceptibility_correction = False
else:
    do_susceptibility_correction = True
    mag = os.path.abspath(args.fieldmapmag)
    phase = os.path.abspath(args.fieldmapphase)

"""
Create the NODDI workflow
"""

workflow = create_matlab_noddi_workflow(
    [os.path.abspath(f) for f in args.dwis], [os.path.abspath(f) for f in args.bvals],
    [os.path.abspath(f) for f in args.bvecs], os.path.abspath(args.t1), result_dir,
    in_fm_mag=mag, in_fm_phase=phase, in_nsf=args.noise_scaling_factor, dwi_interp_type='CUB',
    in_susceptibility_params=[args.rot, args.etd, args.ped],
    in_matlabpoolsize = args.matlabpoolsize, with_eddy=args.with_eddy, t1_mask=args.t1_mask)

workflow.base_dir = result_dir

"""
output the graph if required
"""

if args.graph is True:
    generate_graph(workflow=workflow)
    sys.exit(0)

"""
Edit the qsub arguments based on the input arguments
"""

qsubargs_time = '02:00:00'
qsubargs_mem = '2.9G'
if args.use_qsub is True and args.openmp_core > 1:
    qsubargs_mem = str(max(0.95, 2.9/args.openmp_core)) + 'G'

qsubargs = '-l s_stack=10240 -j y -b y -S /bin/csh -V'
qsubargs = qsubargs + ' -l h_rt=' + qsubargs_time
qsubargs = qsubargs + ' -l tmem=' + qsubargs_mem + ' -l h_vmem=' + qsubargs_mem + ' -l vf=' + qsubargs_mem

"""
Run the workflow
"""

run_workflow(workflow=workflow,
             qsubargs=qsubargs,
             parser=args)
