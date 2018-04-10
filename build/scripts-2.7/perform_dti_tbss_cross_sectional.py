#!/home/claicury/Code/Install/niftypipe/bin/python


"""
=================
DTI: TBSS - Cross-sectional TBSS analysis
=================

Introduction
============

This script, perform_dti_tbss_cross_sectional.py, performs 1) a cross-sectional groupwise from tensor image inputs and
2) the Cross-section Tract-Based Spatial Statistics (TBSS) pipeline as described in
http://10.1371/journal.pone.0045996::
    python perform_dti_tbss_cross_sectional.py --help

The required input are the tensor images. If the design matrix and contrast file are provided,
the permutation test is performed, otherwise not.


Details
============

The input tensor images can be specified using the `--input_img` argument as a list.

The design matrix (--mat) and contrast (--con) can be generated as described on this page:
    http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/GLM with the Glm_gui executable.

Note that the file order has to match the design matrix.
The user also need to specify an output directory (--output_dir) where all the result files will be saved.

.. note::
    The Groupwise follows the strategy of DTITK (http://dti-tk.sourceforge.net/).
    The **Tensor groupwise registration** uses 3 rigid, 3 affine, and 3 non-linear registration steps towards the
    average, using the deviatoric part of the tensor as similarity metric

.. topic:: Outputs

    The pipeline provides outputs of the tract base statistics analysis, ready for statistical analysis:

    - **tbss_mean_fa**:
    - **tbss_mean_fa_skeleton**:
    - **tbss_mean_fa_skeleton_mask**:
    - **tbss_all_fa_skeletonised**:
    - **tbss_all_md_skeletonised**:
    - **tbss_all_rd_skeletonised**:
    - **tbss_all_ad_skeletonised**:


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
import os
import sys
import textwrap
from niftypipe.workflows.dmri.dtitk_tbss import create_cross_sectional_tbss_pipeline
from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)

"""
Main method
"""


def main():

    """
    Main function call
    """

    pipeline_description = textwrap.dedent('''

    Cross-section Tract-Based Spatial Statistics (TBSS) pipeline as described in http://10.1371/journal.pone.0045996

    The required input are the tensor images. If the design matrix and contrast file are provided, the permutation
    test is performed, otherwise not.

    The input tensor images can be specified using the --input_img argument as a list.

    The design matrix (--mat) and contrast (--con) can be generated as described on
    this page: http://fsl.fmrib.ox.ac.uk/fsl/fslwiki/GLM with the Glm_gui executable.
    Note that the file order has to match the design matrix.

    The user also need to specify an output directory (--output_dir) where all the
    result files will be saved.

    Using the --g argument you can generate the pipeline graph without running the
    actual pipeline.

    List of the binaries necessary for this pipeline:

    * FSL: fslmaths, fslsplit
    * niftyreg: reg_aladin, reg_f3d, reg_resample
    * dtitk: [all]
    ''')

    """
    Create the arguments parser
    """

    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=pipeline_description)

    # Input
    parser.add_argument('--input_img',
                        dest='input_img',
                        metavar='input_img',
                        help='List of Nifti file(s) to include in the analysis',
                        nargs='+',
                        required=True)
    parser.add_argument('--mat',
                        dest='design_mat',
                        metavar='design_mat',
                        help='Design matrix file to be used by randomised on the skeletonised FA maps.' +
                             ' Required to run the statistical analysis')
    parser.add_argument('--con',
                        dest='design_con',
                        metavar='design_con',
                        help='Design contrast file to be used by randomised on the skeletonised FA maps.' +
                             ' Required to run the statistical analysis')
    parser.add_argument('--skeleton_thr',
                        type=float,
                        dest='skeleton_threshold',
                        metavar='thr',
                        help='Threshold value to use to binarise the TBSS skeleton (default is 0.2)',
                        default=0.2)

    # Output directory
    parser.add_argument('-o', '--output_dir',
                        dest='output_dir',
                        metavar='output_dir',
                        help='Output directory where to save the results',
                        default=os.getcwd(),
                        required=False)

    # Others
    parser.add_argument('--no_randomise',
                        dest='run_randomise',
                        help='Do not perform the randomise test on the skeletonised FA maps (permutation test)',
                        action='store_false',
                        default=True)

    """
    Add default arguments in the parser
    """

    default_parser_argument(parser)

    """
    Parse the input arguments
    """

    args = parser.parse_args()

    """
    Read the input images
    """

    input_images = [os.path.abspath(f) for f in args.input_img]

    """
    Read the design files
    """

    if args.run_randomise is True:
        # Assign the design matrix variable
        if args.design_mat is None:
            print('No design matrix has been specified. Exit')
            sys.exit()
        # Assign the design contrast variable
        design_matrix = os.path.abspath(args.design_mat)
        if args.design_con is None:
            print('No design contrast has been specified. Exit')
            sys.exit()

        # Check that the number of file agrees with the design matrix and contrast
        with open(design_matrix, 'r') as f:
            for line in f:
                if '/NumPoints' in line:
                    if int(line.split()[1]) is not len(input_images):
                        print('Incompatible image number and design matrix file')
                        print(line.split()[1]+' images are expected and '+str(len(input_images))+' are specified')
                        exit()

    """
    Create the workflow
    """

    workflow = create_cross_sectional_tbss_pipeline(in_files=input_images,
                                                    output_dir=args.output_dir,
                                                    name='tbss_cross_sectional',
                                                    skeleton_threshold=args.skeleton_threshold,
                                                    design_mat=args.design_mat,
                                                    design_con=args.design_con,
                                                    )

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
Main function call
"""

if __name__ == "__main__":
    main()
