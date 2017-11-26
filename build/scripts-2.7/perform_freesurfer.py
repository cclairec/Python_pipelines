#!/opt/local/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python

"""
=================
Freesurfer: Single timepoint analysis
=================


Pipeline Code
==================

Import the necessary python modules

"""

from argparse import (ArgumentParser, RawDescriptionHelpFormatter)
import textwrap
import os
import sys
from nipype.interfaces.freesurfer import ReconAll
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)


def main():

    """
    Create the help message
    """

    pipeline_description = textwrap.dedent('''
    Pipeline to run the complte freesurfer pipeline on structural images
    ''')

    """
    Create the parser
    """

    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=pipeline_description)

    # Input images
    parser.add_argument('--input_t1w',
                        dest='input_t1w',
                        metavar='FILE',
                        help='List of T1w Nifti file(s) to process',
                        nargs='+',
                        required=True)
    parser.add_argument('--input_t2w',
                        dest='input_t2w',
                        metavar='FILE',
                        help='Optional list of T2w Nifti file(s) to process',
                        nargs='+')
    parser.add_argument('--input_sid',
                        dest='input_sid',
                        metavar='FILE',
                        help='Optional list subject ID',
                        nargs='+')

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
    Check the length of the input lists
    """
    if args.input_t2w is not None:
        if len(args.input_t1w) is not len(args.input_t2w):
            raise Exception('The numbers of T1w and T2w files differ')
    if args.input_sid is not None:
        if len(args.input_t1w) is not len(args.input_sid):
            raise Exception('The numbers of T1w files and subject ID differ')

    """
    Create the workflow that generates a cross sectional groupwise and extract diffusion features subject-wise
    """
    workflow = pe.Workflow(name='freesurfer')
    workflow.base_output_dir = 'freesurfer'
    input_node = pe.Node(interface=niu.IdentityInterface(fields=['T1_files',
                                                                 'T2_files',
                                                                 'subject_id']),
                         name='input_node')

    input_node.inputs.T1_files = [os.path.abspath(f) for f in args.input_t1w]
    if args.input_t2w is not None:
        input_node.inputs.T2_files = [os.path.abspath(f) for f in args.input_t2w]
    if args.input_sid is not None:
        input_node.inputs.subject_id = args.input_sid
    recon = None
    if args.input_t2w is not None and args.input_sid is not None:
        recon = pe.MapNode(interface=ReconAll(),
                           iterfield=['T1_files',
                                      'T2_file',
                                      'subject_id'],
                           name='recon')
        workflow.connect(input_node, 'T2_files', recon, 'T2_file')
        workflow.connect(input_node, 'subject_id', recon, 'subject_id')
        recon.inputs.use_T2 = True
    elif args.input_t2w is not None:
        recon = pe.MapNode(interface=ReconAll(),
                           iterfield=['T1_files',
                                      'T2_file'],
                           name='recon')
        workflow.connect(input_node, 'T2_files', recon, 'T2_file')
        recon.inputs.use_T2 = True
    elif args.input_sid is not None:
        recon = pe.MapNode(interface=ReconAll(),
                           iterfield=['T1_files',
                                      'subject_id'],
                           name='recon')
        workflow.connect(input_node, 'subject_id', recon, 'subject_id')
    workflow.connect(input_node, 'T1_files', recon, 'T1_files')
    recon.inputs.subjects_dir = result_dir
    recon.inputs.openmp = args.openmp_core

    """
    output the graph if required
    """

    if args.graph is True:
        generate_graph(workflow=workflow)
        sys.exit(0)

    """
    Edit the qsub arguments based on the input arguments
    """

    qsubargs_time = '48:00:00'
    qsubargs_mem = '5.9G'
    if args.use_qsub is True and args.openmp_core > 1:
        qsubargs_mem = str(max(0.95, 5.9/args.openmp_core)) + 'G'

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
