#! /usr/bin/env python

import nipype.interfaces.io as nio
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
import sys
from nipype.interfaces.niftyseg.stats import UnaryStats as segstats
from nipype.interfaces.fsl.maths import BinaryMaths as fslmaths
import textwrap
from argparse import (ArgumentParser, RawDescriptionHelpFormatter)
import os
import numpy as np
import nibabel as nib
from niftypipe.workflows.misc.profiling import create_dual_structure_1D_profile_pipeline
from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)


def type_transform_function(in_array):
    return float(in_array)


def gen_substitutions(subject_id):
    subs = [('graph', subject_id + '_hippocampus_volumes'), ('profile_1', subject_id + '_profile_left'),
            ('profile_2', subject_id + '_profile_right')]
    return subs


def get_input_extension(in_file):
    import sys
    if in_file[-7:] == '.nii.gz':
        ext = '.nii.gz'
    elif in_file[-4:] == '.nii':
        ext = '.nii'
    else:
        print('ERROR: unrecognized extension: ', in_file)
        sys.exit()
    return ext

pipeline_description = textwrap.dedent('''


''')

parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter, description=pipeline_description)

parser.add_argument('-i', '--input_segmentation',
                    dest='input_segmentation',
                    metavar='FILE',
                    help='Input segmentation / data image file(s).',
                    required=True)
parser.add_argument('-m', '--input_mask',
                    dest='input_mask',
                    metavar='FILE',
                    help='Input mask for normalisation image file (brain or cranium mask).',
                    required=False)
parser.add_argument('-p', '--profilesize',
                    type=int,
                    dest='profilesize',
                    metavar='profilesize',
                    help='profile size (integer, default: 30)',
                    required=False,
                    default=30)
parser.add_argument('-o', '--output',
                    dest='output',
                    metavar='DIRECTORY',
                    help='output directory to which the outputs are stored',
                    required=False)
parser.add_argument('--multiply',
                    type=int,
                    dest='multiply',
                    metavar='INT',
                    help='Multiply the input data by the voxel volume (default: 1)',
                    default=1)

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

# extracting basename of the input file (list)
input_file = os.path.abspath(args.input_segmentation)
voxel_volume = np.prod(np.array(nib.load(input_file).get_header().get_zooms()))
basename = os.path.basename(input_file)
basename = basename.replace(get_input_extension(basename), '')

# The processing pipeline itself is instantiated
workflow_name = 'hippocampus_volume_workflow'

r = create_dual_structure_1D_profile_pipeline(name=workflow_name)
r.base_dir = result_dir

input_multiplier = pe.Node(interface=fslmaths(operation='mul'),
                           name='input_multiplier')
input_multiplier.inputs.in_file = input_file
input_multiplier.inputs.operand_value = voxel_volume

# node to divide the data by the normalisation mask:
brain_volume_integrater = pe.Node(interface=segstats(operation='v'),
                                  name='brain_volume_integrater')
type_transformer = pe.Node(interface=niu.Function(input_names=['in_array'],
                                                  output_names=['out_float'],
                                                  function=type_transform_function),
                           name='type_transformer')
input_divider = pe.Node(interface=fslmaths(operation='div'),
                        name='input_divider')

# Connect all the nodes together
if args.input_mask:
    input_mask = os.path.abspath(args.input_mask)
    brain_volume_integrater.inputs.in_file = input_mask
    if args.multiply >= 1:
        r.connect(input_multiplier, 'out_file', input_divider, 'in_file')
    else:
        input_divider.inputs.in_file = input_file
    r.connect(brain_volume_integrater, 'output', type_transformer, 'in_array')
    r.connect(type_transformer, 'out_float', input_divider, 'operand_value')
    r.connect(input_divider, 'out_file', r.get_node('input_node'), 'in_dual_structure_image')
else:
    if args.multiply >= 1:
        r.connect(input_multiplier, 'out_file', r.get_node('input_node'), 'in_dual_structure_image')
    else:
        r.inputs.input_node.in_dual_structure_image = input_file


r.inputs.input_node.in_profilesize = args.profilesize
r.inputs.input_node.in_axis_of_symmetry = 0
r.inputs.input_node.in_xlabel = 'Normalised hippocampal abscissa ( Anterior $\longleftrightarrow$ Posterior )'
label_suffix = ''
if args.input_mask:
    label_suffix = ' (% of TIV)'
label_information = 'value'
if args.multiply >= 1:
    label_information = 'volume'
r.inputs.input_node.in_ylabel_1 = 'Left hippocampal ' + label_information + label_suffix
r.inputs.input_node.in_ylabel_2 = 'Right hippocampal ' + label_information + label_suffix

subsgen = pe.Node(interface=niu.Function(input_names=['subject_id'],
                                         output_names=['substitutions'],
                                         function=gen_substitutions),
                  name='subsgen')
subsgen.inputs.subject_id = basename
ds = pe.Node(nio.DataSink(), name='ds')
ds.inputs.base_directory = os.path.abspath(result_dir)
ds.inputs.parameterization = False
r.connect(subsgen, 'substitutions', ds, 'regexp_substitutions')
r.connect(r.get_node('output_node'), 'out_profile_1', ds, '@out_profile_l')
r.connect(r.get_node('output_node'), 'out_profile_2', ds, '@out_profile_r')
r.connect(r.get_node('output_node'), 'out_graph', ds, '@out_graph')

# output the graph if required
if args.graph is True:
    generate_graph(workflow=r)
    sys.exit(0)

# Edit the qsub arguments based on the input arguments
qsubargs_time = '01:00:00'
qsubargs_mem = '2.9G'
if args.use_qsub is True and args.openmp_core > 1:
    qsubargs_mem = str(max(0.95, 2.9/args.openmp_core)) + 'G'

qsubargs = '-l s_stack=10240 -j y -b y -S /bin/csh -V'
qsubargs = qsubargs + ' -l h_rt=' + qsubargs_time
qsubargs = qsubargs + ' -l tmem=' + qsubargs_mem + ' -l h_vmem=' + qsubargs_mem + ' -l vf=' + qsubargs_mem

run_workflow(workflow=r,
             qsubargs=qsubargs,
             parser=args)
