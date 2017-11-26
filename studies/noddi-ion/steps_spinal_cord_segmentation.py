#! /usr/bin/env python

import nipype.interfaces.utility as niu     # utility
import nipype.pipeline.engine as pe          # pypeline engine
from nipype.interfaces.niftyreg import RegAladin, RegResample, RegTransform
from nipype.interfaces.niftyseg import BinaryMaths
from nipype.interfaces                  import Function
from distutils                          import spawn
import nipype.interfaces.niftyseg       as niftyseg
import nipype.interfaces.niftyreg       as niftyreg
import nipype.interfaces.niftyfit       as niftyfit

import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.interfaces.utility import Rename
import nipype.interfaces.io             as nio 
from os.path import basename
import os, math, sys, inspect
import numpy as np
import numpy.random
from string import Template


'''This file provides some common segmentation routines useful for a variety of pipelines. '''

# Compute spinal cord segmentation
def create_spinal_cord_segmentation_workflow(name='spinal_cord_segmentation',
					     spinal_cord_template='/usr2/mrtools/pipelines/database/spinal_cord_segmentation/b0/spinal_cord_template_b0.nii.gz',
                                             spinal_cord_template_mask='/usr2/mrtools/pipelines/database/spinal_cord_segmentation/b0/spinal_cord_template_b0_mask.nii.gz'):

    """
     Creates a workflow for computing spinal cord segmentation in DWI

    """

    workflow = pe.Workflow(name=name)

    # We need to create an input node for the workflow
    input_node = pe.Node(
        niu.IdentityInterface(
            fields=['in_file'], mandatory_inputs=False),
        name='input_node')

    # We calculate the mean intensity
    mean_int_value = pe.Node(interface = fsl.ImageStats(op_string= '-M'), name = 'mean_intensity')

    # trehshold by the mean intensity
    tresh_val = pe.Node(interface=fsl.maths.Threshold(direction = 'below'),
                               name = 'tresh_mean_int')

    # Mask dilation per slice to be sure that we got the spinal cord
    mask_binarization = pe.Node(interface = niftyseg.UnaryMaths(operation='bin'),name = 'binarization')

    # Mask dilation to be sure that we got the spinal cord
    mask_erosion = pe.Node(interface = niftyseg.BinaryMathsInteger(), 
                                 name = 'mask_erosion')
    mask_erosion.inputs.operation = 'ero'
    mask_erosion.inputs.operand_value = 2

    # Mask dilationto be sure that we got the spinal cord
    mask_dilation = pe.Node(interface = niftyseg.BinaryMathsInteger(), 
                                 name = 'mask_dilation')
    mask_dilation.inputs.operation = 'dil'
    mask_dilation.inputs.operand_value = 2

    # For spinal cord we do slice wise operation
    split_slices_dwi = pe.Node(interface = fsl.Split(dimension='z'), name = 'split_slices_dwi')

    # For spinal cord we do slice wise operation
    split_slices_mask = pe.Node(interface = fsl.Split(dimension='z'), name = 'split_slices_mask')

    # We register each slice to our mask
    slice_registration = pe.MapNode(niftyreg.RegAladin(verbosity_off_flag = True,
                                                       rig_only_flag = True,
                                                       cog_flag = True,
                                                       ln_val = 1,
                                                       lp_val = 1,
                                                       smoo_r_val = 6,
                                                       maxit_val = 10), 
                                    name = 'slice_to_mask_registration',
                                    iterfield=['ref_file','rmask_file'])

    # Resample the Mask to input space
    resampling = pe.MapNode(niftyreg.RegResample(verbosity_off_flag = True), name = 'resampling', iterfield=['trans_file', 'ref_file'])

    # Remerge all the slices
    merge_mask = pe.Node(interface = fsl.Merge(dimension = 'z'), name = 'merge_mask')

    # Output node
    output_node = pe.Node( interface=niu.IdentityInterface(fields=['out_mask']),
                           name="output_node" )

    # We start the operations
    slice_registration.inputs.flo_file=spinal_cord_template
    slice_registration.inputs.fmask_file=spinal_cord_template_mask
    resampling.inputs.flo_file=spinal_cord_template_mask
    resampling.inputs.inter_val = 'NN'
    workflow.connect(input_node,        'in_file',    mean_int_value,     'in_file')
    workflow.connect(input_node,        'in_file',    tresh_val,          'in_file')
    workflow.connect(mean_int_value,    'out_stat',   tresh_val,          'thresh')
    workflow.connect(tresh_val,         'out_file',   mask_binarization,  'in_file')
    workflow.connect(mask_binarization, 'out_file',   mask_erosion,       'in_file')
    workflow.connect(mask_erosion,      'out_file',   mask_dilation,      'in_file')
    workflow.connect(mask_dilation,     'out_file',   split_slices_mask,  'in_file')
    workflow.connect(input_node,        'in_file',    split_slices_dwi,   'in_file')
    workflow.connect(split_slices_dwi,  'out_files',  slice_registration, 'ref_file')
    workflow.connect(split_slices_mask, 'out_files',  slice_registration, 'rmask_file')
    workflow.connect(split_slices_dwi,  'out_files',  resampling,         'ref_file')
    workflow.connect(slice_registration,'aff_file',   resampling,         'trans_file')
    workflow.connect(resampling,        'res_file',   merge_mask,         'in_files')
    workflow.connect(merge_mask,        'merged_file',output_node,        'out_mask')

    return workflow


# Compute spinal cord segmentation
def create_spinal_cord_segmentation_based_on_STEPS_workflow(name='spinal_cord_segmentation',
                                                            out_file_name='',
                                                            output_probabilistic_mask=False):

    """
     Creates a workflow for computing spinal cord GM and WM segmentation

    """

    workflow = pe.Workflow(name=name)

    # We need to create an input node for the workflow, steps and patchmatch use a slightly different database,
    # basically the differences are in the masks used for the registration in steps and for delimiting the search area in patch match
    input_node = pe.Node(niu.IdentityInterface(fields=['in_file',
                                                       'in_database',
                                                       'in_mask',
                                                       'in_database_pm',
                                                       'in_mask_pm'], 
                                               mandatory_inputs=False),
                         name='input_node')

    # For spinal cord we do slice wise operation
    split_slices = pe.Node(interface = fsl.Split(dimension='z'), name = 'split_slices')

    # We got one file as reference for further operations
    get_one_template = pe.Node(interface = niu.Function(
                               input_names = ['in_database'], 
                               output_names = ['out_file'],
                               function = get_one_image_from_database), 
                               name = 'get_one_template')

    # We register the input slices to the database
    fit_to_template = pe.MapNode(niftyreg.RegAladin(rig_only_flag = True,
                                                    ln_val=1,
                                                    lp_val=1,
                                                    maxit_val=10,
                                                    verbosity_off_flag=True), 
                                 name = 'register_to_database_space',
                                 iterfield=['flo_file'])

    # Resample images to fit with the database - in that case we don't need transformation matrix
    resample_input = pe.MapNode(niftyreg.RegResample(verbosity_off_flag=True), 
                                name = 'resample_input', 
                                iterfield=['flo_file','trans_file'])

    # We calculate where is the spinal cord using patch match
    sc_detection = pe.MapNode(interface = niftyseg.PatchMatch(),
                              name = 'detect_cord_using_patchmatch',
                              iterfield = ['in_file'])

    # We got the mask
    get_mask_pm = pe.MapNode(interface = niftyseg.BinaryMathsInteger(), 
                             name = 'get_whole_cord_result',
                             iterfield = ['in_file'])
    get_mask_pm.inputs.operation = 'tp'
    get_mask_pm.inputs.operand_value = 1

    # We merge all the mask in different timepoints 
    merge_mask = pe.Node(interface = fsl.Merge(dimension = 't'), 
                         name = 'merge_mask')

    # I fuse all the results
    mean_mask = pe.Node(interface = niftyseg.UnaryMaths(operation='tmean'),
                        name = 'mean_mask')

    # trehshold by the mean intensity
    tresh_val = pe.Node(interface=fsl.maths.Threshold(direction = 'below',thresh=0.5),
                           name = 'tresh_mean_probability')

    # Mask binarization per slice 
    mask_binarization = pe.Node(interface = niftyseg.UnaryMaths(operation='bin'),
                                   name = 'mask_binarization')

    # Mask dilation to be sure that we got the spinal cord
    mask_erosion = pe.Node(interface = niftyseg.BinaryMathsInteger(), 
                                 name = 'mask_erosion')
    mask_erosion.inputs.operation = 'ero'
    mask_erosion.inputs.operand_value = 2

    # Mask dilation to be sure that we got the spinal cord
    mask_dilation = pe.Node(interface = niftyseg.BinaryMathsInteger(), 
                                 name = 'mask_dilation')
    mask_dilation.inputs.operation = 'dil'
    mask_dilation.inputs.operand_value = 10

    # We got the spinal cord database
    get_data_list = pe.Node(interface = niu.Function(
                            input_names = ['in_file', 'in_filename'], 
                            output_names = ['image_list','mask_list','output_list'],
                            function = file_to_list), 
                            name = 'get_spinal_cord_templates_from_database')

    # We got the wm mask
    get_wm_mask = pe.MapNode(interface = niftyseg.BinaryMathsInteger(), 
                             name = 'get_wm_mask_from_template',
                             iterfield = ['in_file'])
    get_wm_mask.inputs.operation = 'tp'
    get_wm_mask.inputs.operand_value = 1

    # We got the gm mask
    get_gm_mask = pe.MapNode(interface = niftyseg.BinaryMathsInteger(), 
                             name = 'get_gm_mask_from_template',
                             iterfield = ['in_file'])
    get_gm_mask.inputs.operation = 'tp'
    get_gm_mask.inputs.operand_value = 0

    # We do the registration of each slice to all the template database and we merge the results
    registration = pe.MapNode(name='slice_registration_to_template_database', 
                              interface=Function(input_names=['in_file','image_list','gm_list','wm_list','mask_list','in_mask','result_dir'], 
                                                 output_names=['dir_list'], 
                                                 function=compute_slice_registration_to_template_database),
                              iterfield=['in_file'])

    # We search the different results and we make the new list
    get_results = pe.MapNode(interface = niu.Function(input_names = ['in_dir'],
                                                      output_names = ['image_template', 'gm_template','wm_template'],
                                                      function = find_results),
                                                      name = 'result_finder',
                                                      iterfield = ['in_dir'])

    # Compute STEPS mask for WM and GM per slice
    steps_wm_mask = pe.MapNode(interface = niftyseg.STEPS(), 
                               name = 'calculate_steps_wm_mask',
                               iterfield = ['in_file','warped_img_file','warped_seg_file'])

    steps_gm_mask = pe.MapNode(interface = niftyseg.STEPS(), 
                               name = 'calculate_steps_gm_mask',
                               iterfield = ['in_file','warped_img_file','warped_seg_file'])

    # We register the resampled input slices to the split input slices for a better resampling 
    fit_back_to_input = pe.MapNode(niftyreg.RegAladin(nosym_flag = True,
                                                      ln_val=2,
                                                      lp_val=2,
                                                      maxit_val=10,
                                                      verbosity_off_flag=True), 
                                   name = 'register_to_input_slices',
                                   iterfield=['ref_file','flo_file'])

    # Resample output to fit with original input size - in that case we don't need transformation matrix
    resample_wm_output = pe.MapNode(niftyreg.RegResample(verbosity_off_flag=True), 
                                    name = 'resample_wm_size', 
                                    iterfield=['ref_file','flo_file','trans_file'])
    if not output_probabilistic_mask:
        resample_wm_output.inputs.inter_val = 'NN'

    resample_gm_output = pe.MapNode(niftyreg.RegResample(verbosity_off_flag=True), 
                                    name = 'resample_gm_size', 
                                    iterfield=['ref_file','flo_file','trans_file'])
    if not output_probabilistic_mask:
        resample_gm_output.inputs.inter_val = 'NN'

    # Remerge all the results for each slice
    merge_steps_wm_mask = pe.Node(interface = fsl.Merge(dimension = 'z',
                                                        merged_file=basename(out_file_name).split(".")[0]+'_wm.nii.gz'), 
                                  name = 'merge_steps_wm_mask')
    
    merge_steps_gm_mask = pe.Node(interface = fsl.Merge(dimension = 'z',
                                                        merged_file=basename(out_file_name).split(".")[0]+'_gm.nii.gz'), 
                                  name = 'merge_steps_gm_mask')

    # Create datasink for output
    data_sink = pe.Node(nio.DataSink(), 
                        name='data_output')
    data_sink.inputs.parameterization = False

    # Output node
    output_node = pe.Node( interface=niu.IdentityInterface(fields=['out_gm_mask','out_wm_mask']),
                           name="output_node" )

    # We start the operations
    workflow.connect(input_node,        'in_file',    split_slices,     'in_file')
    workflow.connect(input_node,        'in_database',get_one_template, 'in_database')
    workflow.connect(get_one_template,  'out_file',   fit_to_template,  'ref_file')
    workflow.connect(split_slices,      'out_files',  fit_to_template,  'flo_file')
    workflow.connect(get_one_template,  'out_file',   resample_input,   'ref_file')
    workflow.connect(split_slices,      'out_files',  resample_input,   'flo_file')
    workflow.connect(fit_to_template,   'aff_file',   resample_input,   'trans_file')
    workflow.connect(resample_input,    'res_file',       sc_detection,     'in_file')
    workflow.connect(input_node,        'in_database_pm', sc_detection,     'database_file')
    workflow.connect(input_node,        'in_mask_pm',     sc_detection,     'mask_file')
    workflow.connect(sc_detection,      'out_file',   get_mask_pm,      'in_file')
    workflow.connect(get_mask_pm,       'out_file',   merge_mask,       'in_files')
    workflow.connect(merge_mask,        'merged_file',mean_mask,        'in_file') 
    workflow.connect(mean_mask,         'out_file',   tresh_val,        'in_file')
    workflow.connect(tresh_val,         'out_file',   mask_binarization,'in_file')
    workflow.connect(mask_binarization, 'out_file',   mask_erosion,     'in_file')
    workflow.connect(mask_erosion,      'out_file',   mask_dilation,    'in_file')
    
    # We put all the templates and mask templates in a list
    workflow.connect(input_node,        'in_database',get_data_list,    'in_file')
    workflow.connect(input_node,        'in_file',    get_data_list,    'in_filename')
    workflow.connect(get_data_list,     'output_list',get_wm_mask,      'in_file')
    workflow.connect(get_data_list,     'output_list',get_gm_mask,      'in_file')

    # We register each one of the slice to each template
    registration.inputs.result_dir=os.path.join(os.getcwd(),name)
    workflow.connect(resample_input,    'res_file',   registration,      'in_file')
    workflow.connect(mask_dilation,     'out_file',   registration,      'in_mask')
    workflow.connect(get_data_list,     'mask_list',  registration,      'mask_list')
    workflow.connect(get_data_list,     'image_list', registration,      'image_list')
    workflow.connect(get_wm_mask,       'out_file',   registration,      'wm_list')
    workflow.connect(get_gm_mask,       'out_file',   registration,      'gm_list')

    # We collect the results
    workflow.connect(registration,      'dir_list',   get_results,      'in_dir')  

    # We compute WM steps mask
    steps_wm_mask.inputs.prob_flag=output_probabilistic_mask
    steps_wm_mask.inputs.prob_update_flag=True
    steps_wm_mask.inputs.kernel_size=1.5
    steps_wm_mask.inputs.template_num=15
    steps_wm_mask.inputs.mrf_value=0.55
    workflow.connect(resample_input,'res_file',       steps_wm_mask,     'in_file')
    workflow.connect(get_results,   'image_template', steps_wm_mask,     'warped_img_file')
    workflow.connect(get_results,   'wm_template',    steps_wm_mask,     'warped_seg_file')
    workflow.connect(mask_dilation, 'out_file',       steps_wm_mask,     'mask_file')

    # We compute GM steps mask	
    steps_gm_mask.inputs.prob_flag=output_probabilistic_mask
    steps_gm_mask.inputs.prob_update_flag=True
    steps_gm_mask.inputs.kernel_size=1.5
    steps_gm_mask.inputs.template_num=15
    steps_gm_mask.inputs.mrf_value=0.55
    workflow.connect(resample_input,'res_file',       steps_gm_mask,     'in_file')
    workflow.connect(get_results,   'image_template', steps_gm_mask,     'warped_img_file')
    workflow.connect(get_results,   'gm_template',    steps_gm_mask,     'warped_seg_file')
    workflow.connect(mask_dilation, 'out_file',       steps_gm_mask,     'mask_file')

    # We resample and merge all the slices for GM and WM mask
    workflow.connect(resample_input,     'res_file',  fit_back_to_input,  'flo_file')
    workflow.connect(split_slices,       'out_files', fit_back_to_input,  'ref_file')
    workflow.connect(steps_wm_mask,      'out_file',  resample_wm_output, 'flo_file')
    workflow.connect(split_slices,       'out_files', resample_wm_output, 'ref_file')
    workflow.connect(fit_back_to_input,  'aff_file',  resample_wm_output, 'trans_file')
    workflow.connect(resample_wm_output, 'res_file',  merge_steps_wm_mask,'in_files')
    workflow.connect(steps_gm_mask,      'out_file',  resample_gm_output, 'flo_file')
    workflow.connect(split_slices,       'out_files', resample_gm_output, 'ref_file')
    workflow.connect(fit_back_to_input,  'aff_file',  resample_gm_output, 'trans_file')
    workflow.connect(resample_gm_output, 'res_file',  merge_steps_gm_mask,'in_files')

    # We copy the results to the output node
    workflow.connect(merge_steps_wm_mask,'merged_file',  data_sink,       '@wm')
    workflow.connect(merge_steps_gm_mask,'merged_file',  data_sink,       '@gm')
    workflow.connect(merge_steps_wm_mask,'merged_file',  output_node,     'out_wm_mask')
    workflow.connect(merge_steps_gm_mask,'merged_file',  output_node,     'out_gm_mask')
    data_sink.inputs.base_directory=os.path.dirname(out_file_name)
    
    return workflow     


def compute_slice_registration_to_template_database(in_file,image_list,gm_list,wm_list,mask_list,in_mask,result_dir):
    from nipype.interfaces.base import (TraitedSpec, File, traits, BaseInterface, BaseInterfaceInputSpec, OutputMultiPath)
    import niftk
    from niftk.base import NIFTKCommand, NIFTKCommandInputSpec, getNiftkPath
    from nipype.interfaces.niftyreg import RegAladin, RegTransform, RegResample, RegF3D, RegJacobian
    from nipype.interfaces.niftyseg import BinaryMaths
    from nipype.interfaces                  import Function
    from distutils                          import spawn
    import nipype.interfaces.niftyseg       as niftyseg
    import nipype.interfaces.niftyreg       as niftyreg
    import nipype.interfaces.fsl as fsl
    import nipype.pipeline.engine as pe
    import nipype.interfaces.utility as niu
    import nipype.interfaces.io             as nio 
    import os, math, sys, inspect
    from os.path import basename
    import numpy as np
    import numpy.random
    from niftk.segmentation import file_to_list, get_best_masks, filename_in_array

    slice_outputdir = os.path.join(result_dir, 'reg_to_template_database_'+basename(in_file).split(".")[0])
    pipeline = pe.Workflow(name='reg_to_template_database_'+basename(in_file).split(".")[0])
    pipeline.base_dir=result_dir
    pipeline.base_output_dir=slice_outputdir

    # We need to create an input node for the workflow
    input_node = pe.Node(niu.IdentityInterface(fields=['in_file',
                                                       'image_list',
                                                       'gm_list',
                                                       'wm_list',
                                                       'mask_list',
                                                       'in_mask'], 
                                               mandatory_inputs=False),
                         name='input_node')

    # We are going to register all the templates of the database to our spinal cord, slice per slice
    # First rigid and then non-rigid
    slice_rigid_reg = pe.MapNode(niftyreg.RegAladin(rig_only_flag = True,
                                                    cog_flag = True,
                                                    ln_val = 1,
                                                    lp_val = 1,
                                                    maxit_val=5,
                                                    smoo_f_val = 1,
                                                    smoo_r_val = 1,                
                                                    verbosity_off_flag=True), 
                                 name = 'slice_to_template_rigid_registration',
                                 iterfield=['flo_file','fmask_file'])

    slice_non_rigid_reg = pe.MapNode(niftyreg.RegF3D(verbosity_off_flag = True,
                                                     lncc_val = -1,
                                                     ln_val = 6,
                                                     lp_val = 6), 
                                      name = 'slice_to_template_nonrigid_registration',
                                      iterfield=['flo_file','fmask_file','aff_file'])

    # We resample GM and WM mask of the template database
    resampling_wm_mask = pe.MapNode(niftyreg.RegResample(verbosity_off_flag=True), 
                                    name = 'resampling_wm_mask',
                                    iterfield=['trans_file', 'flo_file'])
    resampling_wm_mask.inputs.inter_val = 'NN'

    resampling_gm_mask = pe.MapNode(niftyreg.RegResample(verbosity_off_flag=True), 
                                    name = 'resampling_gm_mask',
                                    iterfield=['trans_file', 'flo_file'])
    resampling_gm_mask.inputs.inter_val = 'NN'

    resampling_templates = pe.MapNode(niftyreg.RegResample(verbosity_off_flag=True), 
                                      name = 'resampling_templates',
                                      iterfield=['trans_file', 'flo_file'])

    # Compute the best templates to merge and then we get the corresponding mask
    best_templates = pe.Node(interface = niftyseg.CalcTopNCC(), 
                                name = 'calculate_best_templates')

    best_masks = pe.Node(interface = niu.Function(input_names = ['in_files','in_gm_list','in_wm_list'],
                                                  output_names = ['gm_list', 'wm_list'],
                                                  function = get_best_masks),
                                                  name = 'get_best_masks')

    # Remerge all the results for each slice
    merge_wm_mask = pe.Node(interface = fsl.Merge(dimension = 't',
                                                  merged_file=basename(in_file).split(".")[0]+'_wm_mask.nii.gz'), 
                            name = 'merge_wm_mask')

    merge_gm_mask = pe.Node(interface = fsl.Merge(dimension = 't',
                                                  merged_file=basename(in_file).split(".")[0]+'_gm_mask.nii.gz'), 
                            name = 'merge_gm_mask')
 
    merge_templates = pe.Node(interface = fsl.Merge(dimension = 't',
                                                    merged_file=basename(in_file).split(".")[0]+'_templates.nii.gz'), 
                              name = 'merge_templates')

    # Output node
    ds = pe.Node(nio.DataSink(), name='ds')
    
    # We add all the nodes to the workflow
    pipeline.add_nodes([input_node])
    pipeline.add_nodes([slice_rigid_reg])
    pipeline.add_nodes([slice_non_rigid_reg])
    pipeline.add_nodes([resampling_templates])
    pipeline.add_nodes([resampling_gm_mask])    
    pipeline.add_nodes([resampling_wm_mask])
    pipeline.add_nodes([merge_wm_mask])  
    pipeline.add_nodes([merge_gm_mask])
    pipeline.add_nodes([merge_templates])  
    pipeline.add_nodes([ds])  

    # We are going to register all the templates of the database to each slice
    pipeline.inputs.input_node.in_file=in_file
    pipeline.inputs.input_node.image_list=image_list
    pipeline.inputs.input_node.gm_list=gm_list
    pipeline.inputs.input_node.wm_list=wm_list
    pipeline.inputs.input_node.in_mask=in_mask 
    pipeline.inputs.input_node.mask_list=mask_list 
    pipeline.connect(input_node,         'in_file',    slice_rigid_reg,    'ref_file')
    pipeline.connect(input_node,         'in_mask',    slice_rigid_reg,    'rmask_file')
    pipeline.connect(input_node,         'image_list', slice_rigid_reg,    'flo_file')
    pipeline.connect(input_node,         'mask_list',  slice_rigid_reg,    'fmask_file')
    pipeline.connect(input_node,         'in_file',    slice_non_rigid_reg,'ref_file')
    pipeline.connect(input_node,         'in_mask',    slice_non_rigid_reg,'rmask_file')
    pipeline.connect(input_node,         'image_list', slice_non_rigid_reg,'flo_file')
    pipeline.connect(input_node,         'mask_list',  slice_non_rigid_reg,'fmask_file')
    pipeline.connect(slice_rigid_reg,    'aff_file',   slice_non_rigid_reg,'aff_file')

    # We resample all the templates
    pipeline.connect(input_node,           'in_file',   resampling_templates,'ref_file')
    pipeline.connect(slice_non_rigid_reg,  'cpp_file',  resampling_templates,'trans_file')
    pipeline.connect(input_node,           'image_list',resampling_templates,'flo_file')

    # We resample the WM template mask
    pipeline.connect(input_node,           'in_file',   resampling_wm_mask,   'ref_file')
    pipeline.connect(slice_non_rigid_reg,  'cpp_file',  resampling_wm_mask,   'trans_file')
    pipeline.connect(input_node,           'wm_list',   resampling_wm_mask,   'flo_file')

    # We resample the GM template mask
    pipeline.connect(input_node,           'in_file',   resampling_gm_mask,   'ref_file')
    pipeline.connect(slice_non_rigid_reg,  'cpp_file',  resampling_gm_mask,   'trans_file')
    pipeline.connect(input_node,           'gm_list',   resampling_gm_mask,   'flo_file')

    # We compute the best templates
    best_templates.inputs.top_templates=100
    best_templates.inputs.num_templates=len(image_list)
    pipeline.connect(input_node,           'in_file',   best_templates,      'in_file')
    pipeline.connect(resampling_templates, 'res_file',  best_templates,      'in_templates')
    pipeline.connect(input_node,          'in_mask',   best_templates,      'mask_file')

    # Get the corresponding best masks
    pipeline.connect(best_templates,       'out_files', best_masks,          'in_files')
    pipeline.connect(resampling_wm_mask,   'res_file',  best_masks,          'in_wm_list')
    pipeline.connect(resampling_gm_mask,   'res_file',  best_masks,          'in_gm_list')

    # We merge all the template in one file per slice
    pipeline.connect(best_masks,           'wm_list',   merge_wm_mask,        'in_files')
    pipeline.connect(best_masks,           'gm_list',   merge_gm_mask,        'in_files')
    pipeline.connect(best_templates,       'out_files', merge_templates,      'in_files')

    ds.inputs.base_directory = slice_outputdir
    ds.inputs.parameterization = False
    pipeline.connect(merge_gm_mask,        'merged_file',  ds, '@gm')
    pipeline.connect(merge_wm_mask,        'merged_file',  ds, '@wm')
    pipeline.connect(merge_templates,      'merged_file',  ds, '@template')

    dot_exec=spawn.find_executable('dot')   
    if not dot_exec == None:
        niftk.base.generate_graph(workflow=pipeline)
    
    # Run the workflow
    qsubargs = '-l h_rt=02:00:00 -l tmem=2.9G -l h_vmem=2.9G -l vf=2.9G -l s_stack=10240 -j y -b y -S /bin/csh -V'
    niftk.base.run_workflow(workflow=pipeline,
                            qsubargs=qsubargs,
                            proc_num=2)
    
    return slice_outputdir

def find_results(in_dir):
    import os, glob
    image_template = glob.glob(os.path.join(in_dir, 'vol*_templates.*'))
    gm_template = glob.glob(os.path.join(in_dir,  'vol*_gm_mask.*'))
    wm_template = glob.glob(os.path.join(in_dir,  'vol*_wm_mask.*'))
    return image_template[0], gm_template[0], wm_template[0]


def file_to_list(in_file,in_filename):
    import os, glob
    from os.path import basename
    image_list=[]
    mask_list=[]
    output_list=[]
    with open(in_file) as f:
        for line in f:
            data = line.split()
            image_file=os.path.abspath(data[0])
            mask_file=os.path.abspath(data[1])
            output_file=os.path.abspath(data[2])
            name=basename(in_filename).split('_')
            if line.find(name[0])<0:
                image_list.append(image_file)
                mask_list.append(mask_file)
                output_list.append(output_file)

    return image_list,mask_list,output_list

def get_one_image_from_database(in_database):
    import os, glob
    from os.path import basename
    image_list=[]
    with open(in_database) as f:
        for line in f:
            data = line.split()
            image_file=os.path.abspath(data[0])
            image_list.append(image_file)
    out_file=image_list[0]

    return out_file

def get_best_masks(in_files,in_gm_list,in_wm_list):
    from niftk.segmentation import filename_in_array
    import os, glob
    from os.path import basename
    gm_list=[]
    wm_list=[]
    i=0
    while i < len(in_files):
        image_file=basename(in_files[i]).split('res')[0].replace('image','segmentation')
        pos=filename_in_array(in_gm_list,image_file)
        if pos<len(in_gm_list):
                gm_list.append(in_gm_list[pos])
                wm_list.append(in_wm_list[pos])
        i=i+1

    return gm_list,wm_list

def filename_in_array(in_files,filename):
    i=0
    while i < len(in_files) and in_files[i].find(filename)<0:
	i=i+1

    return i
    
