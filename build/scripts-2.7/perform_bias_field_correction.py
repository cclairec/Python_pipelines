#!/home/claicury/Code/Install/niftypipe/bin/python

import sys
import os
import textwrap
from argparse import (ArgumentParser, RawDescriptionHelpFormatter)
from niftypipe.workflows.structuralmri.n4_bias_correction import create_n4_bias_correction_workflow
from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)


def main():
    # Create the parser
    pipeline_description = textwrap.dedent('''
    Pipeline to perform a bias field correction on an input image
    or a list of input images.

    The pipeline uses the N4 algorithm, implemented within an Insight ToolKit framework.
    It takes a structural image as input, and optionally an input mask.

    The implementation comes from a contribution from Nicholas J. Tustison, James C. Gee
    in the Insight Journal paper: https://hdl.handle.net/10380/3053

    List of the binaries necessary for this pipeline:

    * FSL: fslmaths
    * niftyreg: reg_aladin, reg_resample
    * NifTK: niftkN4BiasCorrection

    ''')
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=pipeline_description)
    """ Input images """
    parser.add_argument('-i', '--img', dest='input_img',
                        type=str, nargs='+',
                        metavar='input_img', 
                        help='Image file or list of input images',
                        required=True) 
    parser.add_argument('-m', '--mask', dest='input_mask',
                        type=str, nargs='+',
                        metavar='input_mask', 
                        help='Mask image or list of mask images (optional)', 
                        required=False)
    """ Output argument """
    parser.add_argument('-o', '--output_dir',
                        dest='output_dir',
                        type=str,
                        metavar='directory', 
                        help='Output directory containing the registration result\n' +
                        'Default is the current directory',
                        default=os.path.abspath('.'), 
                        required=False)
    parser.add_argument('--output_pre',
                        dest='output_pre', 
                        type=str,
                        metavar='prefix', 
                        help='Output result prefix',
                        default='', 
                        required=False)
    parser.add_argument('--output_suf',
                        dest='output_suf',
                        type=str,
                        metavar='suffix', 
                        help='Output result suffix',
                        default='', 
                        required=False)

    # Add default arguments in the parser
    default_parser_argument(parser)

    # Parse the input arguments
    args = parser.parse_args()

    # Check the parsed arguments
    if args.input_mask is not None:
        if not len(args.input_img) == len(args.input_mask):
            print('ERROR: The number of input and mask images are expected to be the same.')
            print(str(len(args.input_img))+' image(s) versus ' + str(len(args.input_mask)) + ' mask(s). Exit.')
            sys.exit(1)
    
    result_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    workflow = create_n4_bias_correction_workflow([os.path.abspath(f) for f in args.input_img],
                                                  result_dir,
                                                  [os.path.abspath(f) for f in args.input_mask] if args.input_mask else None)

    # output the graph if required
    if args.graph is True:
        generate_graph(workflow=workflow)
        sys.exit(0)

    # Edit the qsub arguments based on the input arguments
    qsubargs_time = '05:00:00'
    qsubargs_mem = '2.9G'
    if args.use_qsub is True and args.openmp_core > 1:
        qsubargs_mem = str(max(0.95, 2.9/args.openmp_core)) + 'G'

    qsubargs = '-l s_stack=10240 -j y -b y -S /bin/csh -V'
    qsubargs = qsubargs + ' -l h_rt=' + qsubargs_time
    qsubargs = qsubargs + ' -l tmem=' + qsubargs_mem + ' -l h_vmem=' + qsubargs_mem + ' -l vf=' + qsubargs_mem

    run_workflow(workflow=workflow,
                 qsubargs=qsubargs,
                 parser=args)

if __name__ == "__main__":
    main()
