#!/opt/local/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python

"""
=================
Brain Parcellation: Full brain parcellation - Geodesic Flow Information
=================

Introduction
============

This script, perform_gif_propagation.py, performs a full cortical and subcortical grey matter parcellation
using NiftyPipe::

    python perform_gif_propagation.py --help

We perform this segmentation using a set of in-house registration and segmentation software
and the geodesic flow information propagation strategy, GIF: http://dx.doi.org/10.1109/TMI.2015.2418298 ::


    @article { CardosoTMI2015,
    title={Geodesic information flows: spatially-variant graphs and their application to segmentation and fusion},
    author={Cardoso, M Jorge and Modat, Marc and Wolz, Robin and Melbourne,
            Andrew and Cash, David and Rueckert, Daniel and Ourselin, Sebastien},
    journal={Medical Imaging, IEEE Transactions on},
    volume={34},
    number={9},
    pages={1976--1988},
    year={2015},
    publisher={IEEE} }


Details
============

This pipeline performs a full brain parcellation using the Geodesic Information Flow strategy,
as described in http://dx.doi.org/10.1109/TMI.2015.2418298

.. topic:: gif database

    The algorithm uses a database of subjects where parcellations are known. The databse should be described in
    an xml file, that can be found in the following paths

    - **CS network**: `/cluster/project0/GIF/template-database/db.xml`

    - **DRC network**: `/var/drc/software/64bits/gif-template-database/db.xml`


**The input should be a structural T1 image file in nifti format.**

.. topic:: Outputs

    The pipeline provides a series of outputs related to the parcellation of the input T1, using the file basename
    as a prefix for all outputs (here as <fn>)

    - **<fn>_bias_corrected.nii.gz**: The bias corrected image.
    - **<fn>_brain.nii.gz**: Binary mask image of the brain tissue
    - **<fn>_labels.nii.gz**: Label image that includes the 208 grey matter structures (see **_volumes.csv**)
    - **<fn>_prior.nii.gz**: The prior image for each tissue
    - **<fn>_seg.nii.gz**: The tissue segmentation image (background, CSF, GM, WM, deep GM, brainstem)
    - **<fn>_tiv.nii.gz**: Binary mask image of the total intracranial volume
    - **<fn>_volumes.csv**: A comma separated table containing the description and volume
    for each label structure in cubic millimeters.


.. topic:: Output images referential

    For computation time reasons, the input image is cropped to accelerate the registrations. As a consequence, all
    outputs are cropped with respect to the input image. They are however in the same `space`. The q- and s-forms of
    the outputs are corrected in order to be in adequation with the input image.

    If the user would necessitate to have the outputs in the exact same grid as the input, please use reg_resample::

        reg_resample -ref <gif_input> -flo <gif_output> -res <output_in_input_grid> -inter 0



List of the binaries necessary for this pipeline:
============

    * **FSL**: fslmaths, fslmerge, fslsplit
    * **niftyreg**: reg_aladin, reg_transform, reg_f3d, reg_resample
    * **niftyseg**: seg_maths
    * **niftyfit**: fit_dwi, dwi_tool
    * **susceptibility**: pm_scale, gen_fm, gen_pm, phase_unwrap
    * **gif_extras**: seg_GIF
    * **niftk**: niftkCropImage, niftkN4BiasCorrection


Pipeline Code
==================

Import the necessary python modules

"""

from argparse import (ArgumentParser, RawDescriptionHelpFormatter)
import os
import sys
import textwrap
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
from niftypipe.workflows.structuralmri.gif_parcellation import (create_gif_propagation_workflow,
                                                                create_gif_pseudoct_workflow)
from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)


"""
define the help message
"""

pipeline_description = textwrap.dedent('''
    This pipeline performs a full brain parcellation using the Geodesic Information Flow strategy,
    as described in http://dx.doi.org/10.1109/TMI.2015.2418298

    The algorithm uses a database of subjects where parcellations are known. The databse should be described in
    an xml file, that can be found in the following paths
    - CS network: /cluster/project0/GIF/template-database/db.xml
    - DRC network: /var/drc/software/64bits/gif-template-database/db.xml

    The input should be a structural T1 image file in nifti format.

    List of the binaries necessary for this pipeline:

    * FSL: fslmaths, fslsplit
    * niftyreg: reg_aladin, reg_f3d, reg_resample
    * niftyseg: seg_maths
    * gif_extras: seg_GIF
    ''')

"""
Define the Parser and arguments
"""

parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=pipeline_description)

parser.add_argument('-i', '--input_t1_file',
                    dest='input_t1_file',
                    metavar='FILE',
                    help='Input T1 image file to propagate labels in',
                    required=False)
parser.add_argument('--input_t2_file',
                    dest='input_t2_file',
                    metavar='FILE',
                    help='Input T2 image file to propagate labels in',
                    required=False)
parser.add_argument('-d', '--database',
                    dest='database',
                    metavar='FILE',
                    help='gif-based database xml file describing the inputs',
                    required=True)
parser.add_argument('-o', '--output',
                    dest='output',
                    metavar='DIR',
                    help='output directory to which the gif outputs are stored',
                    required=True)
parser.add_argument('--pct',
                    dest='pct',
                    help='Use the Pseudo CT parameters',
                    action='store_true',
                    default=False)
parser.add_argument('--ute_echo2_file',
                    dest='ute_echo2_file',
                    metavar='FILE',
                    help='Input ute_echo2 file',
                    required=False)
parser.add_argument('--ute_umap_dir',
                    dest='ute_umap_dir',
                    metavar='DIR',
                    help='Input ute_umap DICOM folder',
                    required=False)
parser.add_argument('--nac_pet_dir',
                    dest='nac_pet_dir',
                    metavar='DIR',
                    help='Input Non Attenuation Corrected Dynamic PET DICOM folder',
                    required=False)
parser.add_argument('--use_lncc',
                    dest='lncc',
                    help='Use LNCC as a measure of similarity for the non linear registration. NMI by default',
                    action='store_true',
                    default=False)

"""
Add default arguments in the parser
"""

default_parser_argument(parser)

"""
Parse the input arguments
"""

args = parser.parse_args()

result_dir = os.path.abspath(args.output)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

t1 = os.path.abspath(args.input_t1_file) if args.input_t1_file else None
t2 = os.path.abspath(args.input_t2_file) if args.input_t2_file else None
nac_pet_dir = os.path.abspath(args.nac_pet_dir) if args.nac_pet_dir else None

if not t1 and not t2:
    print('ERROR: no input image provided')
    sys.exit(1)

if args.pct:
    if not args.ute_echo2_file or not args.ute_umap_dir:
        print('ERROR: no UTE inputs provided')
        sys.exit(1)
    cpp_dir = os.path.join(result_dir, 'cpps')
    if not os.path.exists(cpp_dir):
        os.mkdir(cpp_dir)
    r = create_gif_pseudoct_workflow(os.path.abspath(args.ute_echo2_file),
                                     os.path.abspath(args.ute_umap_dir),
                                     os.path.abspath(args.database),
                                     cpp_dir,
                                     in_t1_file=t1,
                                     in_t2_file=t2,
                                     in_nac_pet_dir=nac_pet_dir)
else:
    r = create_gif_propagation_workflow(t1 if t1 else t2,
                                        os.path.abspath(args.database),
                                        result_dir,
                                        use_lncc=args.lncc)
r.base_dir = result_dir

"""
Create a data sink
"""
ds = pe.Node(nio.DataSink(parameterization=False,
                          base_directory=result_dir),
             name='ds')
r.connect(r.get_node('renamer'), 'out_file', ds, '@outputs')


"""
output the graph if required
"""

if args.graph is True:
    generate_graph(workflow=r)
    sys.exit(0)

"""
Edit the qsub arguments based on the input arguments
"""

qsubargs_time = '05:00:00'
qsubargs_mem = '3.9G'
if args.use_qsub is True and args.openmp_core > 1:
    qsubargs_mem = str(max(0.95, 3.9/args.openmp_core)) + 'G'

qsubargs = '-l s_stack=10240 -j y -b y -S /bin/csh -V'
qsubargs = qsubargs + ' -l h_rt=' + qsubargs_time
qsubargs = qsubargs + ' -l tmem=' + qsubargs_mem + ' -l h_vmem=' + qsubargs_mem + ' -l vf=' + qsubargs_mem

"""
Run the workflow with the different arguments
"""

run_workflow(workflow=r,
             qsubargs=qsubargs,
             parser=args)
