#! /usr/bin/env python
import os
import sys
import textwrap
from argparse import (ArgumentParser, RawDescriptionHelpFormatter)
from niftypipe.workflows.misc.utils import create_regional_average_pipeline
from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)

pipeline_description = textwrap.dedent('''
Regional average value computation.
Using a parcelation image (-par) and a scalar value image (-img), the mean intensity
values are computed for each label.
''')

parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=pipeline_description)

"""
Input images
"""
parser.add_argument('--img',
                    dest='input_img',
                    type=str,
                    nargs='+',
                    metavar='image',
                    help='Function image or list of function images',
                    required=True)
parser.add_argument('--par',
                    dest='input_par',
                    type=str,
                    nargs='+',
                    metavar='image',
                    help='Parcelation image or list of parcelation images',
                    required=True)
parser.add_argument('--weights',
                    dest='input_weights',
                    type=str,
                    nargs='+',
                    metavar='file',
                    help='Weight file or list of weight files (0 to 1)',
                    required=False)
parser.add_argument('--trans',
                    dest='input_trans',
                    type=str,
                    nargs='+',
                    metavar='file',
                    help='Transformation file or list of transformation files (ref=img, flo=par)',
                    required=False)
parser.add_argument('--threshold',
                    dest='threshold',
                    type=float,
                    metavar='float',
                    help='Threshold to use on weights in the average computation',
                    required=False)
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
parser.add_argument('-l', '--label',
                    dest='input_label',
                    type=int,
                    nargs='+',
                    metavar='input_label',
                    help='Specify Label value(s) to extract',
                    required=False)
parser.add_argument('--neuromorph',
                    dest='neuromorphometrics',
                    help='Update the csv file with the neuromorphometrics label names',
                    action='store_true',
                    default=False)
parser.add_argument('--fs',
                    dest='freesurfer',
                    help='Update the csv file with the freesurfer label names',
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

"""
Create the output folder if it does not exists
"""
result_dir = os.path.abspath(args.output_dir)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

"""
parse inputs
"""
in_files = [os.path.abspath(f) for f in args.input_img]
in_rois = [os.path.abspath(f) for f in args.input_par]
in_trans = None
if args.input_trans:
    in_trans = [os.path.abspath(f) for f in args.input_trans]
in_weights = None
if args.input_weights:
    in_trans = [os.path.abspath(f) for f in args.input_weights]

workflow = create_regional_average_pipeline(
    output_dir=result_dir,
    name='regional_average',
    in_trans=in_trans,
    in_weights=in_weights,
    freesurfer=args.freesurfer,
    neuromorphometrics=args.neuromorphometrics,
    in_label=args.input_label,
    in_threshold=args.threshold)

workflow.inputs.input_node.in_files = in_files
workflow.inputs.input_node.in_rois = in_rois

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

run_workflow(workflow=workflow,
             qsubargs=qsubargs,
             parser=args)
