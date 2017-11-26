#!/opt/local/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python

"""
=================
rs-fMRI: Resting State fMRI preprocessing - AFNI
=================

Introduction
============

This script, perform_restingstate_preprocessing.py::

    python perform_restingstate_preprocessing.py --help


Details
============

This pipeline is dedicated to the pre-processing of Resting State fMRI images.

It is based on the AFNI (Analysis of Functional NeuroImages) implementation. More information can be
found here: https://afni.nimh.nih.gov/afni/

The main inputs required for processing are the following:

    1) resting state fMRI image in a 4D nifti format
    2) Structural T1 in a nifti file
    3) Segmentation image (in GIF output format) in a nifti 4D file - (background, CSF, GM, WM, deep GM, brainstem)
    4) Parcellation image (in GIF output format) in a nifti 4D file - 208 labels


.. topic:: Outputs

    The pipeline provides outputs

    - ****:



List of the binaries necessary for this pipeline:
=================

* **FSL**: fslmaths, fslmerge, fslsplit
* **niftyreg**: reg_aladin, reg_transform, reg_f3d, reg_resample
* **AFNI**: [all] e.g. 3dvolreg


.. seealso::

    https://afni.nimh.nih.gov/afni/
        AFNI: Analysis of Functional NeuroImages - Set of software dedicated to functional image analysis

    https://sourceforge.net/projects/niftyreg/
        NiftyReg: contains programs to perform rigid, affine and non-linear registration of nifti or analyse images



Pipeline Code
==================

Import the necessary python modules

"""

import os
import sys
import textwrap
from argparse import (ArgumentParser, RawDescriptionHelpFormatter)
from niftypipe.workflows.rsfmri.afni_restingstate_fmri import create_restingstatefmri_preprocessing_pipeline
from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)

"""
Create tge help message
"""

pipeline_description=textwrap.dedent('''
    This pipeline is dedicated to the pre-processing of Resting State fMRI images.

    It is based on the AFNI (Analysis of Functional NeuroImages) implementation. More information can be
    found here: https://afni.nimh.nih.gov/afni/

    The main inputs required for processing are the following:

    1) resting state fMRI image in a 4D nifti format
    2) Structural T1 in a nifti file
    3) Segmentation image (in GIF output format) in a nifti 4D file - (background, CSF, GM, WM, deep GM, brainstem)
    4) Parcellation image (in GIF output format) in a nifti 4D file - 208 labels

    List of the binaries necessary for this pipeline:

    * FSL: fslmaths, fslsplit
    * niftyreg: reg_aladin, reg_f3d, reg_resample
    * AFNI: [all] (e.g. 3dvolreg)
    ''')

"""
Create the arguments parser
"""

parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=pipeline_description)

parser.add_argument('-i', '--fmri',
                    dest='fmri',
                    metavar='FILE',
                    help='Input resting state fMRI 4D image',
                    required=True)
parser.add_argument('-t', '--t1',
                    dest='t1',
                    metavar='FILE',
                    help='Input T1 image',
                    required=True)
parser.add_argument('-s', '--segmentation',
                    dest='segmentation',
                    metavar='FILE',
                    help='Input Tissue Segmentation image (from GIF pipeline)',
                    required=True)
parser.add_argument('-p', '--parcellation',
                    dest='parcellation',
                    metavar='parcellation',
                    help='Input Parcellation image (from GIF pipeline)',
                    required=True)
parser.add_argument('-o', '--output_dir',
                    dest='output_dir',
                    metavar='output_dir',
                    help='output directory to which the average and the are stored',
                    default=os.path.abspath('results'),
                    required=False)

parser.add_argument('--susceptibility',
                    dest='susceptibility',
                    help='Correct for magnetic field susceptibility (requires fieldmapmag and fieldmapphase)',
                    action='store_true',
                    default=False)
parser.add_argument('--fieldmapmag',
                    dest='fieldmapmag',
                    metavar='fieldmapmag',
                    help='Field Map Magnitude image file to be associated with the DWIs',
                    required=False)
parser.add_argument('--fieldmapphase',
                    dest='fieldmapphase',
                    metavar='fieldmapphase',
                    help='Field Map Phase image file to be associated with the DWIs',
                    required=False)
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

"""
Add default arguments in the parser
"""

default_parser_argument(parser)

"""
Parse the input arguments
"""

args = parser.parse_args()

result_dir = os.path.abspath(args.output_dir)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

mag = None
phase = None
if args.susceptibility:
    if args.fieldmapmag is None or args.fieldmapphase is None:
        print ('ERROR: Susceptibility requires Field map magnitude and phase inputs.')
        sys.exit(1)
    mag = os.path.abspath(args.fieldmapmag)
    phase = os.path.abspath(args.fieldmapphase)
workflow = create_restingstatefmri_preprocessing_pipeline(os.path.abspath(args.fmri),
                                                          os.path.abspath(args.t1),
                                                          os.path.abspath(args.segmentation),
                                                          os.path.abspath(args.parcellation),
                                                          result_dir,
                                                          mag,
                                                          phase,
                                                          [args.rot, args.etd, args.ped])
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

qsubargs_time = '05:00:00'
qsubargs_mem = '1.9G'
if args.use_qsub is True and args.openmp_core > 1:
    qsubargs_mem = str(max(0.95, 1.9/args.openmp_core)) + 'G'

qsubargs = '-l s_stack=10240 -j y -b y -S /bin/csh -V'
qsubargs = qsubargs + ' -l h_rt=' + qsubargs_time
qsubargs = qsubargs + ' -l tmem=' + qsubargs_mem + ' -l h_vmem=' + qsubargs_mem + ' -l vf=' + qsubargs_mem

"""
Run the workflow
"""

run_workflow(workflow=workflow,
             qsubargs=qsubargs,
             parser=args)
