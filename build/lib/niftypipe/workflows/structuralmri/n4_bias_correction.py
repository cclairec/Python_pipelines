# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
from nipype.utils.filemanip import split_filename
import nipype.interfaces.niftyseg as niftyseg
import nipype.interfaces.niftyreg as niftyreg
from ...interfaces.niftk.filters import N4BiasCorrection

mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
mni_template_mask = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain_mask_dil.nii.gz')


def create_n4_bias_correction_workflow(input_images, output_dir, input_masks=None,
                                       name='n4_bias_correction'):

    subject_ids = [split_filename(os.path.basename(f))[1] for f in input_images]

    # Create a workflow to process the images
    workflow = pe.Workflow(name=name)
    workflow.base_dir = output_dir
    workflow.base_output_dir = name
    # Define the input and output node
    input_node = pe.Node(
        interface=niu.IdentityInterface(
            fields=['in_files',
                    'mask_files'],
            mandatory_inputs=False),
        name='input_node')
    output_node = pe.Node(
        interface=niu.IdentityInterface(
            fields=['out_img_files',
                    'out_bias_files',
                    'out_mask_files']),
        name='output_node')

    input_node.inputs.in_files = input_images
    if input_masks is not None:
        input_node.inputs.mask_files = input_masks

    thresholder = pe.MapNode(interface=fsl.Threshold(),
                             name='thresholder',
                             iterfield=['in_file'])
    thresholder.inputs.thresh = 0

    # Finding masks to use for bias correction:
    bias_correction = pe.MapNode(interface=N4BiasCorrection(),
                                 name='bias_correction',
                                 iterfield=['in_file', 'mask_file'])
    bias_correction.inputs.in_downsampling = 2
    bias_correction.inputs.in_maxiter = 200
    bias_correction.inputs.in_convergence = 0.0002
    bias_correction.inputs.in_fwhm = 0.05

    renamer = pe.MapNode(interface=niu.Rename(format_string="%(subject_id)s_corrected.nii.gz"),
                         name='renamer',
                         iterfield=['in_file', 'subject_id'])
    renamer.inputs.subject_id = subject_ids
    mask_renamer = pe.MapNode(interface=niu.Rename(format_string="%(subject_id)s_corrected_mask.nii.gz"),
                              name='mask_renamer',
                              iterfield=['in_file', 'subject_id'])
    mask_renamer.inputs.subject_id = subject_ids

    if input_masks is None:
        mni_to_input = pe.MapNode(interface=niftyreg.RegAladin(),
                                  name='mni_to_input',
                                  iterfield=['ref_file'])
        mni_to_input.inputs.flo_file = mni_template
        mask_resample = pe.MapNode(interface=niftyreg.RegResample(),
                                   name='mask_resample',
                                   iterfield=['ref_file', 'aff_file'])
        mask_resample.inputs.inter_val = 'NN'
        mask_resample.inputs.flo_file = mni_template_mask
        mask_eroder = pe.MapNode(interface=niftyseg.BinaryMathsInteger(), 
                                 name='mask_eroder',
                                 iterfield=['in_file'])
        mask_eroder.inputs.operation = 'ero'
        mask_eroder.inputs.operand_value = 3
        workflow.connect(input_node, 'in_files', mni_to_input, 'ref_file')
        workflow.connect(input_node, 'in_files', mask_resample, 'ref_file')
        workflow.connect(mni_to_input, 'aff_file', mask_resample, 'aff_file')
        workflow.connect(mask_resample, 'out_file', mask_eroder, 'in_file')
        workflow.connect(mask_eroder, 'out_file', bias_correction, 'mask_file')
        workflow.connect(mask_eroder, 'out_file', mask_renamer, 'in_file')
    else:
        workflow.connect(input_node, 'mask_files', bias_correction, 'mask_file')
        workflow.connect(input_node, 'mask_files', mask_renamer, 'in_file')
    
    workflow.connect(input_node, 'in_files', thresholder, 'in_file')
    workflow.connect(thresholder, 'out_file', bias_correction, 'in_file')
    
    # Gather the processed images
    workflow.connect(bias_correction, 'out_file', renamer, 'in_file')
    workflow.connect(renamer, 'out_file', output_node, 'out_img_files')
    workflow.connect(bias_correction, 'out_biasfield_file', output_node, 'out_bias_files')
    workflow.connect(mask_renamer, 'out_file', output_node, 'out_mask_files')

    # Create a data sink
    ds = pe.Node(nio.DataSink(parameterization=False), name='data_sink')
    ds.inputs.base_directory = output_dir
    workflow.connect(output_node, 'out_img_files', ds, '@img')
    workflow.connect(output_node, 'out_mask_files', ds, '@mask')

    return workflow
