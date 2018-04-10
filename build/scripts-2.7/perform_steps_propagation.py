#!/home/claicury/Code/Install/niftypipe/bin/python
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
import argparse
import os
import sys
from niftypipe.workflows.misc.steps_propagation import create_steps_propagation_pipeline
from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)


def get_db_template_alignment(in_db_file):
    import xml.etree.ElementTree as Xml
    database_xml = Xml.parse(in_db_file).getroot()
    data = database_xml.findall('data')[0]
    return data.find('sform').text == '1'


parser = argparse.ArgumentParser(description='STEPS Propagation')

# Input images
parser.add_argument('--input_img',
                    dest='input_img',
                    metavar='input_img',
                    help='List of Nifti file(s) to parcellate',
                    nargs='+',
                    required=False)
# Input database
parser.add_argument('-d', '--database',
                    dest='database',
                    metavar='database',
                    help='gif-based database xml file describing the inputs',
                    required=True)
# Output directory
parser.add_argument('-o', '--output_dir',
                    dest='output_dir',
                    metavar='output_dir',
                    help='Output directory where to put the labels',
                    required=False,
                    default=os.path.abspath(os.getcwd()))

# Add default arguments in the parser
default_parser_argument(parser)

# Parse the input arguments
args = parser.parse_args()

# Create the output folder if it does not exists
result_dir = os.path.abspath(args.output_dir)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

input_images = [os.path.abspath(f) for f in args.input_img]

# Create an iterable to loop over all input images
selector = pe.Node(niu.Select(inlist=input_images),
                   name='selector',
                   iterables=('index', range(len(input_images))))

# Create the workflow
db = os.path.abspath(args.database)
workflow = create_steps_propagation_pipeline(name='steps_propagation',
                                             aligned_templates=get_db_template_alignment(db))
workflow.inputs.input_node.database_file = db
workflow.base_dir = result_dir
workflow.connect(selector, 'out', workflow.get_node('input_node'), 'in_file')

# output the graph if required
if args.graph is True:
    generate_graph(workflow=workflow)
    sys.exit(0)

# Edit the qsub arguments based on the input arguments
qsubargs_time = '05:00:00'
qsubargs_mem = '3.9G'
if args.use_qsub is True and args.openmp_core > 1:
    qsubargs_mem = str(max(0.95, 3.9/args.openmp_core)) + 'G'

qsubargs = '-l s_stack=10240 -j y -b y -S /bin/csh -V'
qsubargs = qsubargs + ' -l h_rt=' + qsubargs_time
qsubargs = qsubargs + ' -l tmem=' + qsubargs_mem + ' -l h_vmem=' + qsubargs_mem + ' -l vf=' + qsubargs_mem

run_workflow(workflow=workflow,
             qsubargs=qsubargs,
             parser=args)
