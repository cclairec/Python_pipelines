#!/home/claicury/Code/Install/niftypipe/bin/python

import textwrap
from argparse import (ArgumentParser, RawDescriptionHelpFormatter)
import os
import sys
import glob
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
from nipype.workflows.smri.niftyreg.groupwise import create_groupwise_average
from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)


pipelineDescription = textwrap.dedent('''
Pipeline that performs a groupwise co-registration strategy between 3D images.
Input images are listed space separated using '--input_images'

The user can manipulate the rigid-affine-non-linear iterations.


''')

parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=pipelineDescription)
parser_group = parser.add_mutually_exclusive_group(required=True)
parser_group.add_argument('--input_images',
                          dest='input_images',
                          nargs='+',
                          metavar='images',
                          help='Input images to be registered ',
                          default=None)
parser_group.add_argument('--input_folder',
                          dest='input_folder',
                          metavar='folder',
                          help='Input folder containing the images to be registered ',
                          default=None)
parser.add_argument('-o', '--output',
                    dest='output_dir',
                    metavar='output',
                    help='output directory to which the average and the are stored',
                    required=True)

parser.add_argument('-r', '--rigiditerations',
                    dest='rigiditerations',
                    metavar='int',
                    help='Number of rigid iterations',
                    required=False,
                    type=int,
                    default=4)
parser.add_argument('-a', '--affineiterations',
                    dest='affineiterations',
                    metavar='int',
                    help='Number of affine iterations',
                    required=False,
                    type=int,
                    default=4)
parser.add_argument('--nonlineariterations',
                    dest='nonlineariterations',
                    metavar='int',
                    help='Number of nonlinear iterations',
                    required=False,
                    type=int,
                    default=4)
parser.add_argument('--use_mni152',
                    dest='use_mni',
                    action='store_const',
                    help='Use MNI152_1mm to define the discretisation space. ' +
                    'The first image is used by default',
                    default=False, 
                    const=True, 
                    required=False)
parser.add_argument('--template',
                    dest='initial_template',
                    metavar='filename',
                    help='Specify an image to define the discretisation space' +
                    'The first image is used by default',
                    default=None,
                    required=False)

"""
Add default arguments in the parser
"""
default_parser_argument(parser)

"""
Parse the input arguments
"""
args = parser.parse_args()

if args.initial_template is not None and args.use_mni is True:
    raise Exception('Only one image can be specified as initial template')

"""
Create the output folder if it does not exists
"""
result_dir = os.path.abspath(args.output_dir)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

in_files = ''
if args.input_images is not None:
    in_files = [os.path.abspath(f) for f in args.input_images]
elif args.input_folder is not None:
    in_files = [os.path.abspath(f) for f in glob.glob(os.path.abspath(args.input_folder + os.sep + '*.nii'))]
    in_files = in_files + [os.path.abspath(f) for f in
                           glob.glob(os.path.abspath(args.input_folder + os.sep + '*.nii.gz'))]
    in_files = in_files + [os.path.abspath(f) for f in
                           glob.glob(os.path.abspath(args.input_folder + os.sep + '*.hdr'))]
    in_files = in_files + [os.path.abspath(f) for f in
                           glob.glob(os.path.abspath(args.input_folder + os.sep + '*.img'))]
    in_files = in_files + [os.path.abspath(f) for f in
                           glob.glob(os.path.abspath(args.input_folder + os.sep + '*.img.gz'))]

"""
Create the workflow
"""
r = create_groupwise_average(name="atlas_creation",
                             itr_rigid=args.rigiditerations,
                             itr_affine=args.affineiterations,
                             itr_non_lin=args.nonlineariterations)

"""
Set the output folder
"""
r.base_dir = result_dir

"""
Set to input files
"""
r.inputs.input_node.in_files = in_files

"""
Set the initial template to be used for discretisation
"""
if args.initial_template is not None:
    r.inputs.input_node.ref_file = os.path.abspath(args.initial_template)
elif args.use_mni is True:
    r.inputs.input_node.ref_file = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_1mm.nii.gz')
else:
    r.inputs.input_node.ref_file = in_files[0]


ds = pe.Node(nio.DataSink(parameterization=False, base_directory=args.output_dir), name='ds')
r.connect(r.get_node('output_node'), 'average_image', ds, '@average_image')
r.connect(r.get_node('output_node'), 'trans_files', ds, '@trans_files')

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

run_workflow(workflow=r,
             qsubargs=qsubargs,
             parser=args)
