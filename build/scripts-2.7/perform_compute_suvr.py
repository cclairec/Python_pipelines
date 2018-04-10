#!/home/claicury/Code/Install/niftypipe/bin/python

"""
=================
PET-MRI: SUVR analysis - Regional PET Uptake Value Ratio
=================

Introduction
============

This script, perform_compute_suvr.py, performs ::

    python perform_compute_suvr.py --help

Details
============

Compute SUVR: standard uptake value ratio

The SUVR of a PET image **--pet** is computed using a GIF parcelation **--par** and its associated structural image
**--mri**.

Using a parcelation image and a scalar value image **--img**, the mean intensity values are computed for each label of
the desired region of interest **--roi**.


.. topic:: Outputs


.. topic:: Output images referential


List of the binaries necessary for this pipeline:
=================

* **FSL**: fslmaths, fslmerge, fslsplit
* **niftyreg**: reg_aladin, reg_transform, reg_f3d, reg_resample


.. seealso::

    https://sourceforge.net/projects/niftyreg/
        NiftyReg: contains programs to perform rigid, affine and non-linear registration of nifti or analyse images



Pipeline Code
==================

Import the necessary python modules

"""

import os
import sys
import textwrap
import argparse
from niftypipe.workflows.petmri.petmr_suvr import create_compute_suvr_pipeline
from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)

"""
create the help message
"""

pipeline_description = textwrap.dedent('''
    Compute SUVR.
    The SUVR of a PET image (--pet) is computed using a GIF parcelation (--par) and its
    associated structural image (--mri)
    Using a parcelation image (--par) and a scalar value image (--img), the mean intensity
    values are computed for each label.
    ''')

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=pipeline_description)
"""
Input images
"""

parser.add_argument('--pet',
                    dest='input_pet',
                    type=str,
                    metavar='image',
                    help='Function image or list of function images',
                    required=True)
parser.add_argument('--mri',
                    dest='input_mri',
                    type=str,
                    metavar='image',
                    help='MRI image or list of MRI images',
                    required=True)
parser.add_argument('--par',
                    dest='input_par',
                    type=str,
                    metavar='image',
                    help='Parcelation (NeuroMorphometrics from GIF) image or list of parcelation images',
                    required=True)
"""
Output argument
"""

parser.add_argument('--output_dir',
                    dest='output_dir',
                    type=str,
                    metavar='directory',
                    help='Output directory containing the pipeline result',
                    default='.',
                    required=False)
"""
Others argument
"""

roi_choices = ['cereb', 'gm_cereb', 'pons', 'wm_subcort']
parser.add_argument('--roi',
                    metavar='roi',
                    nargs=1,
                    type=str,
                    choices=roi_choices,
                    default=roi_choices[0],
                    help='ROI to use to perform the function image intensities normalisation. ' +
                         'Choices are: '+str(roi_choices)+' without quotes.' +
                         'The default value is ' + str(roi_choices[0]))

parser.add_argument('--erode_ref', action='store_true',
                    help='Perform one erosion on reference region before doing SUVR calculation')

"""
Add default arguments in the parser
"""

default_parser_argument(parser)

"""
Parse the input arguments
"""

args = parser.parse_args()

"""
Create the output folder if it does not exists
"""

result_dir = os.path.abspath(args.output_dir)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

workflow = create_compute_suvr_pipeline(
    os.path.abspath(args.input_pet),
    os.path.abspath(args.input_mri),
    os.path.abspath(args.input_par),
    args.erode_ref,
    result_dir,
    name='compute_suvr',
    norm_region=args.roi[0])


"""
output the graph if required
"""

if args.graph is True:
    generate_graph(workflow=workflow)
    sys.exit(0)

"""
Edit the qsub arguments based on the input arguments
"""

qsubargs_time = '01:00:00'
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
