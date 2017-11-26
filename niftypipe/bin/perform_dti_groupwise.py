#! /usr/bin/env python

"""
=================
DTI: Tensor Groupwise - White Matter Regional Analysis
=================

Introduction
============

This script, perform_dti_groupwise.py, performs a 3-step cross-sectional groupwise registration between tensor images,
followed by the estimation of diffusion biomarker's in the groupwise space in regions of interest of white matter tissue
using NiftyPipe::

    python perform_dti_groupwise.py --help

The Groupwise follows the strategy of DTITK (http://dti-tk.sourceforge.net/).

In a second step, the groupwise average Fractional Anisotropy map is non-lienarly registered to the MNI JHU template
in order to extract relevant diffusion biomarkers for the different White Matter regions in the groupwise space.
The default biomarkers extracted are the following:
    - Fractional Anisotropy
    - Mean Diffusivity (Trace)
    - Axial Diffusivity
    - Radial Diffusivity


Details
============

Perform diffusion tensor groupwise registration and white matter feature extraction with the following steps:

1) **Tensor groupwise registration** using (default) 3 rigid, 3 affine, and 3 non-linear registration steps towards the
average, using the deviatoric part of the tensor as similarity metric

2) **Feature Extraction**: The groupwise Fractional Anisotropy map is non-linearly registered towards the JHU MNI
template using niftyreg. The JHU template in the groupwise space then helps determining biomarkers values in each of the
White Matter regions.

.. topic:: Outputs

    The pipeline provides outputs of the groupwise registration and the biomarker's maps and statistics in the subject
    space.

    - **groupwise_fa.nii.gz**: The groupwise average Fractional Anisotropy map
    - **groupwise_labels.nii.gz**: The JHU labels in the groupwise space
    - **groupwise_tensors.nii.gz**: The groupwise average tensor image in the DTI-TK format (see http://dti-tk.sourceforge.net/ for details).
    - **tensors/<subject_id>_tensors.nii.gz**: The the subject's tensor image mapped into the groupwise space in the DTI-TK format (see http://dti-tk.sourceforge.net/ for details).
    - **biomakers**: for each bm in (fa, ad, rd, tr)
        * **<subject_id>_<bm>.nii.gz**: The subject's biomarker's map in the groupwise space
        * **<subject_id>_<bm>.csv**: The subject's biomarker's text file describing the mean biomarker's value in each region of interest


.. topic:: Output images referential

    The output images come from the DTI-TK strategy, which means they are automatically re-gridded to 128-128-64 grid.




List of the binaries necessary for this pipeline:
=================

* **FSL**: fslmaths, fslmerge, fslsplit
* **niftyreg**: reg_aladin, reg_transform, reg_f3d, reg_resample
* **DTITK**: [all]


.. seealso::

    http://dti-tk.sourceforge.net/
        Diffusion Tensor ToolKit by UCL

    https://sourceforge.net/projects/niftyreg/
        NiftyReg: contains programs to perform rigid, affine and non-linear registration of nifti or analyse images



Pipeline Code
==================

Import the necessary python modules

"""

from argparse import (ArgumentParser, RawDescriptionHelpFormatter)
import textwrap
import os
import sys
from niftypipe.workflows.dmri.dtitk_tensor_groupwise import create_tensor_groupwise_and_feature_extraction_workflow
from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)


def main():

    """
    Create the help messahe
    """

    pipeline_description = textwrap.dedent('''
    Pipeline to perform a groupwise between tensor encoded files and extract subject level regional feature
    extraction.

    It uses the DTITK framework (http://dti-tk.sourceforge.net/) to perform rigid/affine/non-linear registration
    between tensor images, using the deviatoric part of the tensor as similarity measure.

    At a later stage, it uses the MNI JHU FA template in order to extract subject specific diffusion features
    (FA-MD-AD-RD) in White matter regions of interest. See the following citation for details about the atlas:
    Oishi, Kenichi, et al. MRI atlas of human white matter. Academic Press, 2010.

    List of the binaries necessary for this pipeline:

    * FSL: fslmaths, fslsplit
    * niftyreg: reg_aladin, reg_f3d, reg_resample
    * dtitk: [all]
    ''')

    """
    Create the parser
    """

    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=pipeline_description)

    # Input images
    parser.add_argument('--input_img',
                        dest='input_img',
                        metavar='FILE',
                        help='List of Nifti file(s) to include in the processing',
                        nargs='+',
                        required=True)

    # Other inputs
    parser.add_argument('--rigid_it',
                        type=int,
                        dest='rigid_iteration',
                        metavar='INT',
                        help='Number of iteration to perform for the rigid step (default is 3)',
                        default=3)
    parser.add_argument('--affine_it',
                        type=int,
                        dest='affine_iteration',
                        metavar='INT',
                        help='Number of iteration to perform for the affine step (default is 3)',
                        default=3)
    parser.add_argument('--nonrigid_it',
                        type=int,
                        dest='nonrigid_iteration',
                        metavar='INT',
                        help='Number of iteration to perform for the nonrigid step (default is 6)',
                        default=6)
    parser.add_argument('--biomarkers',
                        nargs='+',
                        metavar='STRING',
                        dest='biomarkers',
                        default=['fa', 'tr', 'ad', 'rd'],
                        help='Optional: indicate what biomarkers you want extracted, ' +
                             'The choices are fa tr ad rd. Indicate as a list separated ' +
                             'with space. e.g. --biomarkers fa md')
    # Output directory
    parser.add_argument('-o', '--output_dir',
                        dest='output_dir',
                        metavar='DIR',
                        help='Output directory where to save the results',
                        default=os.getcwd())

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

    """
    Create the workflow that generates a cross sectional groupwise and extract diffusion features subject-wise
    """

    workflow = create_tensor_groupwise_and_feature_extraction_workflow([os.path.abspath(f) for f in args.input_img],
                                                                       result_dir,
                                                                       rig_iteration=args.rigid_iteration,
                                                                       aff_iteration=args.affine_iteration,
                                                                       nrr_iteration=args.nonrigid_iteration,
                                                                       biomarkers=args.biomarkers)

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

"""
Main function called with arguments
"""

if __name__ == "__main__":
    main()
