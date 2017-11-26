#! /usr/bin/env python

import sys
from argparse import (ArgumentParser, RawDescriptionHelpFormatter)
import os
from niftypipe.workflows.asl import create_asl_processing_workflow
from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)

help_message = '''
Perform Arterial Spin Labelling Fitting with pre-processing steps.

Mandatory Input are:
1. the 4D nifti image ASL sequence
2. The inversion recovery image. It is assumed that the inversion recovery image
is a 4D image with 5 instances, from which instances 0,2,4 are extracted
and are assumed to correspond to inversion times of 4,2 and 1 seconds respectively

List of the binaries necessary for this pipeline:

* FSL: fslmaths, fslmerge, fslsplit
* niftyseg: seg_maths
* niftyfit: fit_asl, fit_qt1

'''

parser = ArgumentParser(description=help_message, formatter_class=RawDescriptionHelpFormatter)

parser.add_argument('--ir',
                    dest='ir',
                    required=True,
                    help='Inversion Recovery 4D source file (4th dimension should be 5)')
parser.add_argument('--t1',
                    dest='t1',
                    required=False,
                    help='OPTIONAL: corresponding T1 file. This will make the outputs to be in the space of the T1')
parser.add_argument('--asl',
                    dest='asl',
                    required=True,
                    help='ASL 4D source file')
parser.add_argument('-o', '--output',
                    dest='output',
                    metavar='output',
                    help='Result directory where the output data is to be stored',
                    required=False,
                    default='results')

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
"""
Create the ASL preprocessing workflow
"""
r = create_asl_processing_workflow(in_inversion_recovery_file=os.path.abspath(args.ir),
                                   in_asl_file=os.path.abspath(args.asl),
                                   output_dir=result_dir,
                                   in_t1_file=os.path.abspath(args.t1) if args.t1 else None,
                                   name='asl_workflow')
r.base_dir = result_dir

"""
output the graph if required
"""
if args.graph is True:
    generate_graph(workflow=r)
    sys.exit(0)

"""
Edit the qsub arguments based on the input arguments
"""
qsubargs_time = '03:00:00'
qsubargs_mem = '2.9G'
if args.use_qsub is True and args.openmp_core > 1:
    qsubargs_mem = str(max(0.95, 2.9/args.openmp_core)) + 'G'

qsubargs = '-l s_stack=10240 -j y -b y -S /bin/csh -V'
qsubargs = qsubargs + ' -l h_rt=' + qsubargs_time
qsubargs = qsubargs + ' -l tmem=' + qsubargs_mem + ' -l h_vmem=' + qsubargs_mem + ' -l vf=' + qsubargs_mem

run_workflow(workflow=r,
             qsubargs=qsubargs,
             parser=args)
