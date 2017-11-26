#! /usr/bin/env python

import os
from nipype                             import config, logging
import argparse
from nipype import config

# modified custom interfaces
import niftk

import nipype.interfaces.niftyreg as niftyreg
import nipype.interfaces.niftyseg as niftyseg

# FFE database
spinal_cord_database = os.path.abspath('/usr2/mrtools/pipelines/database/spinal_cord_segmentation/ffe/spinal-cord-database.txt')
spinal_cord_template_mask = os.path.abspath('/usr2/mrtools/pipelines/database/spinal_cord_segmentation/ffe/spinal-cord-ffe-mask.nii.gz')
spinal_cord_database_pm = os.path.abspath('/usr2/mrtools/pipelines/database/spinal_cord_segmentation/ffe/spinal-cord-database-pm.txt')
spinal_cord_template_mask_pm = os.path.abspath('/usr2/mrtools/pipelines/database/spinal_cord_segmentation/ffe/spinal-cord-full-mask.nii.gz')

# DWI database
spinal_cord_database_dwi = os.path.abspath('/usr2/mrtools/pipelines/database/spinal_cord_segmentation/b0/database.txt')
spinal_cord_template_mask_dwi = os.path.abspath('/usr2/mrtools/pipelines/database/spinal_cord_segmentation/b0/cord-b0-mask.nii.gz')
spinal_cord_database_pm_dwi = os.path.abspath('/usr2/mrtools/pipelines/database/spinal_cord_segmentation/b0/database_pm.txt')
spinal_cord_template_mask_pm_dwi = os.path.abspath('/usr2/mrtools/pipelines/database/spinal_cord_segmentation/b0/full_mask.nii.gz')

help_message = \
'------------------------------------------------------------------------------' + \
'-------------------- Perform spinal cord segmentation ------------------------' + \
'------------------------------------------------------------------------------'
parser = argparse.ArgumentParser(description=help_message)
parser.add_argument('-i', '--input',
                    dest='input_file',
                    metavar='input_file',
                    type=str,
                    help='Input data to be segmented.',
                    required=True)
parser.add_argument('-o', '--output', 
                    dest='output_file', 
                    type=str,
                    metavar='output_file_names', 
                    help='Output file names.\n' + \
                    'Default is spinal_cord_mask.nii.gz, and it generates: spinal_cord_mask_gm.nii.gz and spinal_cord_mask_wm.nii.gz.',
                    default=os.path.abspath('spinal_cord_mask.nii.gz'), 
                    required=False)
parser.add_argument('-p', '--p', 
                    dest='probabilistic_output', 
                    metavar='probabilistic_output', 
                    action='store_const',
                    help='Probabilistic output.\n' + \
                    'Probabilistic/Fuzzy segmentation is provided as output, by default binary mask is provided as output.',
                    default=False, 
                    const=True, 
                    required=False)
parser.add_argument('-d', '--d', 
                    dest='dwi_segmentation', 
                    metavar='dwi_segmentation', 
                    action='store_const',
                    help='DWI cord detection using as input a B0 image or the mean B0 image.',
                    default=False, 
                    const=True, 
                    required=False)
parser.add_argument('-b', '--b', 
                    dest='database', 
                    type=str,
                    metavar='database', 
                    help='Template database used by STEPS. By default: '+spinal_cord_database, 
                    required=False)
parser.add_argument('-c', '--c', 
                    dest='database_pm', 
                    type=str,
                    metavar='database_pm', 
                    help='Template database used by Patchmatch, usually the mask for the searching area is bigger than for STEPS. By default: '+spinal_cord_database_pm, 
                    required=False)
parser.add_argument('-m', '--m', 
                    dest='mask', 
                    type=str,
                    metavar='mask', 
                    help='General mask for the STEPS templates. By default: '+spinal_cord_template_mask, 
                    required=False)
parser.add_argument('-n', '--n', 
                    dest='mask_pm', 
                    type=str,
                    metavar='mask', 
                    help='General mask for defining the PatchMatch searching area. By default: '+spinal_cord_template_mask_pm, 
                    required=False)
parser.add_argument('-g', '--graph',
                    dest='graph',
                    help='Print a graph describing the node connections',
                    action='store_true',
                    default=False)


args = parser.parse_args()

if not os.path.exists(os.path.join(os.getcwd(),'spinal_cord_segmentation')):
    os.mkdir(os.path.join(os.getcwd(),'spinal_cord_segmentation')) 

if args.dwi_segmentation:
    spinal_cord_database=spinal_cord_database_dwi
    spinal_cord_template_mask=spinal_cord_template_mask_dwi
    spinal_cord_database_pm=spinal_cord_database_pm_dwi
    spinal_cord_template_mask_pm=spinal_cord_template_mask_pm_dwi

msg=''
if not args.database == None:
    spinal_cord_database = os.path.abspath(args.database)
    if not os.path.exists(spinal_cord_database):
        msg+='\n\t STEPS database not found at '+spinal_cord_database

if not args.mask == None:
    spinal_cord_template_mask = os.path.abspath(args.mask)
    if not os.path.exists(spinal_cord_template_mask):
        msg+='\n\t STEPS mask not found at '+spinal_cord_template_mask

if not args.database_pm == None:
    spinal_cord_database_pm = os.path.abspath(args.database_pm)
    if not os.path.exists(spinal_cord_database_pm):
        msg+='\n\t PatchMatch database not found at '+spinal_cord_database_pm

if not args.mask_pm == None:
    spinal_cord_template_mask_pm = os.path.abspath(args.mask_pm)
    if not os.path.exists(spinal_cord_template_mask_pm):
        msg+='\n\t PatchMatch database not found at '+spinal_cord_template_mask_pm

# Specify how and where to save the log files
config.update_config({
                       'logging': {
                                  'log_directory': os.path.join(os.getcwd(),'spinal_cord_segmentation'),
                                  'log_to_file': True
                                  }
                     })
logging.update_logging(config)
config.enable_debug_mode()
iflogger = logging.getLogger('interface')

iflogger.info('Input file = '+os.path.abspath(args.input_file))
iflogger.info('Output file = '+os.path.abspath(args.output_file))
iflogger.info('Probabilistic output = '+str(args.probabilistic_output))
iflogger.info('STEPS database = '+spinal_cord_database)
iflogger.info('PatchMatch database = '+spinal_cord_database_pm)
iflogger.info('STEPS mask = '+spinal_cord_template_mask)
iflogger.info('PatchMatch mask = '+spinal_cord_template_mask_pm)

if len(msg)>0:
    iflogger.info('There are some errors:'+msg)
    exit(-1)
 
sc_segmentation = niftk.segmentation.create_spinal_cord_segmentation_based_on_STEPS_workflow(
                                            name='spinal_cord_segmentation',
                                            out_file_name=os.path.abspath(args.output_file),
                                            output_probabilistic_mask=args.probabilistic_output)
sc_segmentation.base_dir=os.getcwd()
sc_segmentation.base_output_dir='spinal_cord_segmentation'
sc_segmentation.inputs.input_node.in_file=os.path.abspath(args.input_file)
sc_segmentation.inputs.input_node.in_database=os.path.abspath(spinal_cord_database)
sc_segmentation.inputs.input_node.in_mask=os.path.abspath(spinal_cord_template_mask)
sc_segmentation.inputs.input_node.in_database_pm=os.path.abspath(spinal_cord_database_pm)
sc_segmentation.inputs.input_node.in_mask_pm=os.path.abspath(spinal_cord_template_mask_pm)

# output the graph if required
if args.graph is True:
    niftk.base.generate_graph(workflow=sc_segmentation)

# Run the workflow
qsubargs = '-l h_rt=02:00:00 -l tmem=2.9G -l h_vmem=2.9G -l vf=2.9G -l s_stack=10240 -j y -b y -S /bin/csh -V'
niftk.base.run_workflow(workflow=sc_segmentation,
                        qsubargs=qsubargs)

iflogger.info('The spinal cord segmentation has been done. Have a nice day!')
