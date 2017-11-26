#!/opt/local/Library/Frameworks/Python.framework/Versions/2.7/Resources/Python.app/Contents/MacOS/Python
import os
import sys
import textwrap
from argparse import (ArgumentParser, RawDescriptionHelpFormatter)
from niftypipe.workflows.structuralmri.gradient_unwarp import create_gradient_unwarp_workflow
from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)

pipelineDescription = textwrap.dedent('''
Pipeline to perform a gradwarp correction on an input image or a list of input images.
The grapwarp algorithm has been implemented by Pankaj Daga, and is hosted in https://cmiclab.cs.ucl.ac.uk/CMIC/gradwarp



''')
parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=pipelineDescription)

"""
Input images
"""
parser.add_argument('-i', '--img', dest='input_img',
                    type=str,
                    metavar='input_img',
                    help='Input image file to correct',
                    required=True)
"""
Input coefficient file
"""
parser.add_argument('-c', '--coeff', dest='input_coeff',
                    type=str, metavar='coeff_file',
                    help='File containing the spherical harmonic coefficient',
                    required=True)
"""
Interpolation order input
"""
parser.add_argument('--inter', metavar='inter', type=str,
                    dest='inter',
                    choices=['NN', 'LIN', 'CUB', 'SINC'], default='CUB',
                    help='Interpolation order to resample to unwarped image. ' +
                         'Choices are NN, LIN, CUB, SINC. [CUB]',
                    required=False)
"""
Table offset values input
"""
parser.add_argument('--offset_x', metavar='offset_x', type=float,
                    default=0, dest='offset_x',
                    help='Scanner table offset in x direction in mm. [0]',
                    required=False)
parser.add_argument('--offset_y', metavar='offset_y', type=float,
                    default=0, dest='offset_y',
                    help='Scanner table offset in y direction in mm. [0]',
                    required=False)
parser.add_argument('--offset_z', metavar='offset_z', type=float,
                    default=0, dest='offset_z',
                    help='Scanner table offset in z direction in mm. [0]',
                    required=False)
parser.add_argument('--throughplaneonly', dest='throughplaneonly',
                    action='store_true', default=False,
                    help='Do through plane only correction (default: False)', required=False)
parser.add_argument('--inplaneonly', dest='inplaneonly',
                    action='store_true', default=False,
                    help='Do in plane only correction (default: False)', required=False)
"""
Scanner type input input
"""
parser.add_argument('--scanner', metavar='scanner', type=str, dest='scanner',
                    choices=['ge', 'siemens'], default='siemens',
                    help='Scanner type. Choices are ge and siemens. [siemens]',
                    required=False)
"""
Gradwarp radius input
"""
parser.add_argument('--radius', metavar='radius', type=float, dest='radius',
                    default=0.225,
                    help='Gradwarp radius in meter. [0.225]',
                    required=False)
"""
Output argument
"""
parser.add_argument('--output_dir', dest='output_dir',
                    type=str,
                    metavar='directory',
                    help='Output directory containing the unwarped result\n' +
                         'Default is the current directory',
                    default=os.path.abspath('.'),
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

workflow = create_gradient_unwarp_workflow(os.path.abspath(args.input_img),
                                           os.path.abspath(args.input_coeff),
                                           result_dir,
                                           offsets=[args.offset_x, args.offset_y, args.offset_z],
                                           scanner=args.scanner,
                                           radius=args.radius,
                                           interp=args.inter,
                                           throughplaneonly=args.throughplaneonly,
                                           inplaneonly=args.inplaneonly)

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
qsubargs_time = '01:00:00'
qsubargs_mem = '2.9G'
if args.use_qsub is True and args.openmp_core > 1:
    qsubargs_mem = str(max(0.95, 2.9/args.openmp_core)) + 'G'

qsubargs = '-l s_stack=10240 -j y -b y -S /bin/csh -V'
qsubargs = qsubargs + ' -l h_rt=' + qsubargs_time
qsubargs = qsubargs + ' -l tmem=' + qsubargs_mem + ' -l h_vmem=' + qsubargs_mem + ' -l vf=' + qsubargs_mem

run_workflow(workflow=workflow,
             qsubargs=qsubargs,
             parser=args)