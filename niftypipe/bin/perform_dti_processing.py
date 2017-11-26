#! /usr/bin/env python

"""
=================
DTI: Diffusion Tensor Processing - NiftyFit
=================

Introduction
============

This script, perform_dti_processing.py, demonstrates the ability to perform complex diffusion processing and analysis
using NiftyPipe::

    python perform_dti_processing.py --help

We perform this analysis a set of in-house registration software and diffusion tensor fitting using NiftyFit::

    [ insert Andrew Melbourne paper and download link here ]


Details
============

Perform Diffusion Model Fitting with pre-processing steps:

1) **Motion / eddy-current correction** via affine co-registration between DWIs towards an average B-Null image

2) [optional] **susceptibility correction** via phase unwrapping
as described in http://dx.doi.org/10.1016/j.media.2014.06.008

3) Non-Linear Least square **tensor fitting** using NiftyFit,
as described in [insert Andrew Melbourne paper and download link here]


Most important inputs are the triplets of **dwi-bval-bvec*** files.
If lists are provided [separated by spaces] then the information is cumulated.

A T1 image is needed for reference. Optionally you can provide a T1 mask used to mask the outputs.

Provide field map magnitude and phase images to trigger the susceptibility correction

Values to use for the susceptibility parameters::

    ## DRC ## (--ped=-y --etd=2.46 --rot=34.56) and

    ## 1946 ## (--ped=-y --etd=2.46 --rot=25.92).

*Note that these values are indicative*

.. topic:: Outputs

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


.. topic:: Output images referential

    All output images are produced in the original DWI space. The user can resample these output images manually towards
    the structural T1 reference space using reg_resample::

        reg_resample -ref <T1_input> -flo <any_output> -res <output_in_input_grid>

    The default interpolation style is cubic. **Caution**: cubic interpolation can introduce non-physical (negative)
    image values. Prevent that by using linear interpolation `-inter 1`


List of the binaries necessary for this pipeline:
=================

* **FSL**: fslmaths, fslmerge, fslsplit
* **niftyreg**: reg_aladin, reg_transform, reg_f3d, reg_resample
* **niftyseg**: seg_maths
* **niftyfit**: fit_dwi, dwi_tool
* **susceptibility**: pm_scale, gen_fm, gen_pm, phase_unwrap


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
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
from niftypipe.workflows.dmri.tensor_processing import (create_diffusion_mri_processing_workflow,
                                                        merge_dwi_function,
                                                        remove_dmri_volumes)
from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)

"""
define the help message
"""

help_message = textwrap.dedent('''
    Perform Diffusion Model Fitting with pre-processing steps.
    1) Motion / eddy-current correction via affine co-registration between DWIs towards an average B-Null image
    2) [optional] susceptibility correction via phase unwrapping
        as described in http://dx.doi.org/10.1016/j.media.2014.06.008
    3) Non-Linear Least square tensor fitting using NiftyFit, as described in [REF NEEDED, A. Melbourne et al.]

    Most important inputs are the triplets of dwi-bval-bvec files. If lists are provided [separated by spaces] then the
    information is cumulated.

    A T1 image is needed for reference. Optionally you can provide a T1 mask used to mask the outputs.

    Provide field map magnitude and phase images to trigger the susceptibility correction

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
    ''')

"""
Define the Parser and its arguments
"""

parser = ArgumentParser(description=help_message, formatter_class=RawDescriptionHelpFormatter)

parser.add_argument('-i', '--dwis',
                    dest='dwis',
                    metavar='FILE',
                    nargs='+',
                    help='Diffusion Weighted Images in a 4D nifti file',
                    required=True)
parser.add_argument('-a', '--bvals',
                    dest='bvals',
                    metavar='FILE',
                    nargs='+',
                    help='bval file to be associated with the DWIs',
                    required=True)
parser.add_argument('-e', '--bvecs',
                    dest='bvecs',
                    metavar='FILE',
                    nargs='+',
                    help='bvec file to be associated with the DWIs',
                    required=True)
parser.add_argument('-t', '--t1',
                    dest='t1',
                    metavar='FILE',
                    help='T1 file to be associated with the DWIs',
                    required=True)
parser.add_argument('--t1_mask',
                    dest='t1_mask',
                    metavar='FILE',
                    help='T1 mask file associated with the input T1',
                    required=False)
parser.add_argument('-m', '--fieldmapmag',
                    dest='fieldmapmag',
                    metavar='FILE',
                    help='Field Map Magnitude image file to be associated with the DWIs',
                    required=False)
parser.add_argument('-p', '--fieldmapphase',
                    dest='fieldmapphase',
                    metavar='FILE',
                    help='Field Map Phase image file to be associated with the DWIs',
                    required=False)
parser.add_argument('-o', '--output_dir',
                    dest='output_dir',
                    type=str,
                    metavar='DIR',
                    help='Output directory containing the registration result\n' +
                         'Default is a directory called results',
                    default=os.path.abspath('results'),
                    required=False)
parser.add_argument('--rot',
                    dest='rot',
                    type=float,
                    metavar='FLOAT',
                    help='Diffusion Read-Out time used for susceptibility correction\n' +
                         'Default is 34.56',
                    default=34.56,
                    required=False)
parser.add_argument('--etd',
                    dest='etd',
                    type=float,
                    metavar='FLOAT',
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
                    metavar='LETTER',
                    help='Phase encoding direction used for susceptibility correction (x, y or z)\n' +
                         '--ped=val form must be used for -ve indices' +
                         'Default is the -y direction (-y)',
                    default='-y',
                    required=False)
parser.add_argument('--rigid',
                    dest='rigid_only',
                    action='store_true',
                    help='Only use rigid registration for DWI (no eddy current correction)',
                    required=False)
parser.add_argument('-x', '-y', '-z',
                    dest='pedwarn',
                    help=SUPPRESS,
                    required=False,
                    action='store_true')
parser.add_argument('-r', '--remove_volume',
                    dest='remove_volume',
                    metavar='vol',
                    nargs='+',
                    type=int,
                    help='Index of the volume to remove from the data',
                    default=[],
                    required=False)
parser.add_argument('--susc_corr',
                    dest='force_susc_correction',
                    action='store_true',
                    help='Force susceptibility correction even in the absence of field maps.' +
                         'This is done using non-linear registration only',
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
    print 'ERROR: argument --ped: expected one argument, make sure to use --ped='
    sys.exit(1)
if args.pedwarn:
    print 'ERROR: One of -x, -y or -z found, did you mean --ped=-x, --ped=-y --ped=-z?'
    sys.exit(1)

if (len(args.dwis) != len(args.bvals)) or (len(args.dwis) != len(args.bvecs)):
    print 'ERROR: The number of BVAL and BVEC files should match the number of DWI files'
    sys.exit(1)

result_dir = os.path.abspath(args.output_dir)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

do_susceptibility_correction_with_fm = True
do_susceptibility_correction_without_fm = False
if args.fieldmapmag is None or args.fieldmapphase is None:
    do_susceptibility_correction_with_fm = False
    if args.force_susc_correction is True:
        do_susceptibility_correction_without_fm = True

"""
Use a merging node to accumulate datasets together if the user inputs more than one (dwi-bval-bvec) triplet
"""

merge_initial_dwis = pe.Node(interface=niu.Function(input_names=['in_dwis',
                                                                 'in_bvals',
                                                                 'in_bvecs'],
                                                    output_names=['out_dwis', 'out_bvals', 'out_bvecs'],
                                                    function=merge_dwi_function),
                             name='merge_initial_dwis')
merge_initial_dwis.inputs.in_dwis = [os.path.abspath(f) for f in args.dwis]
merge_initial_dwis.inputs.in_bvals = [os.path.abspath(f) for f in args.bvals]
merge_initial_dwis.inputs.in_bvecs = [os.path.abspath(f) for f in args.bvecs]


"""
Create the workflow using :func create_diffusion_mri_processing_workflow() method and parsing all arguments
"""

r = create_diffusion_mri_processing_workflow(
    t1_mask_provided=args.t1_mask is not None,
    susceptibility_correction_with_fm=do_susceptibility_correction_with_fm,
    susceptibility_correction_without_fm=do_susceptibility_correction_without_fm,
    in_susceptibility_params=[args.rot, args.etd, args.ped],
    name='dmri_workflow',
    resample_in_t1=False,
    log_data=True,
    dwi_interp_type='CUB',
    wls_tensor_fit=True,
    rigid_only=args.rigid_only)
r.base_dir = result_dir

if len(args.remove_volume) > 0:
    clean_data = pe.Node(interface=niu.Function(input_names=['in_dwi',
                                                             'in_bval',
                                                             'in_bvec',
                                                             'volume_to_remove'],
                                                output_names=['out_dwi', 'out_bval', 'out_bvec'],
                                                function=remove_dmri_volumes),
                         name='clean_data')
    r.connect(merge_initial_dwis, 'out_dwis', clean_data, 'in_dwi')
    r.connect(merge_initial_dwis, 'out_bvals', clean_data, 'in_bval')
    r.connect(merge_initial_dwis, 'out_bvecs', clean_data, 'in_bvec')
    clean_data.inputs.volume_to_remove = args.remove_volume
    r.connect(clean_data, 'out_dwi', r.get_node('input_node'), 'in_dwi_4d_file')
    r.connect(clean_data, 'out_bval', r.get_node('input_node'), 'in_bval_file')
    r.connect(clean_data, 'out_bvec', r.get_node('input_node'), 'in_bvec_file')
else:
    r.connect(merge_initial_dwis, 'out_dwis', r.get_node('input_node'), 'in_dwi_4d_file')
    r.connect(merge_initial_dwis, 'out_bvals', r.get_node('input_node'), 'in_bval_file')
    r.connect(merge_initial_dwis, 'out_bvecs', r.get_node('input_node'), 'in_bvec_file')
r.inputs.input_node.in_t1_file = os.path.abspath(args.t1)
if args.t1_mask:
    r.inputs.input_node.in_t1_mask_file = os.path.abspath(args.t1_mask)
if do_susceptibility_correction_with_fm:
    r.inputs.input_node.in_fm_magnitude_file = os.path.abspath(args.fieldmapmag)
    r.inputs.input_node.in_fm_phase_file = os.path.abspath(args.fieldmapphase)

"""
Sink all outputs of the renamer into the output directory.

NOTA BENE:
The renamer actually creates symbolic links. In most file systems, the DataSink will then hard copy the files by
following the symbolic links. But in some cases, the DataSink will only copy the links, and the output directory
will therefore contain symlinks instead of the actual files.
"""

ds = pe.Node(nio.DataSink(parameterization=False, base_directory=result_dir), name='ds')
r.connect(r.get_node('renamer'), 'out_file', ds, '@outputs')
r.connect(r.get_node('reorder_transformations'), 'out', ds, 'transformations')


"""
output the graph if required
"""

if args.graph is True:
    generate_graph(workflow=r)
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
Run the workflow with all arguments
"""

run_workflow(workflow=r,
             qsubargs=qsubargs,
             parser=args)
