# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
import nipype.interfaces.niftyseg as niftyseg
import nipype.interfaces.niftyreg as niftyreg
from ...interfaces.niftk.labels.neuromorphometricslabels import NeuromorphometricsUpdateCsvFileWithLabels
from ..misc.utils import create_regional_average_pipeline
from ..misc.utils import create_regional_normalisation_pipeline


def create_mask_from_functional():
    workflow = pe.Workflow(name='mask_func')
    workflow.base_dir = os.getcwd()
    workflow.base_output_dir = 'mask_func'
    # Create all the required nodes
    input_node = pe.Node(interface=niu.IdentityInterface(fields=['in_files']),
                         name='input_node')
    otsu_filter = pe.MapNode(interface=niftyseg.UnaryMaths(),
                             name='otsu_filter',
                             iterfield=['in_file'])
    erosion_filter = pe.MapNode(interface=niftyseg.BinaryMathsInteger(),
                                name='erosion_filter',
                                iterfield=['in_file'])
    lconcomp_filter = pe.MapNode(interface=niftyseg.UnaryMaths(),
                                 name='lconcomp_filter',
                                 iterfield=['in_file'])
    dilation_filter = pe.MapNode(interface=niftyseg.BinaryMathsInteger(),
                                 name='dilation_filter',
                                 iterfield=['in_file'])
    fill_filter = pe.MapNode(interface=niftyseg.UnaryMaths(),
                             name='fill_filter',
                             iterfield=['in_file'])
    output_node = pe.Node(interface=niu.IdentityInterface(fields=['mask_files']),
                          name='output_node')
    # Define the node options
    otsu_filter.inputs.operation = 'otsu'
    erosion_filter.inputs.operation = 'ero'
    erosion_filter.inputs.operand_value = 1
    lconcomp_filter.inputs.operation = 'lconcomp'
    dilation_filter.inputs.operation = 'dil'
    dilation_filter.inputs.operand_value = 5
    fill_filter.inputs.operation = 'fill'
    fill_filter.inputs.output_datatype = 'char'
    # Create the connections
    workflow.connect(input_node, 'in_files', otsu_filter, 'in_file')
    workflow.connect(otsu_filter, 'out_file', erosion_filter, 'in_file')
    workflow.connect(erosion_filter, 'out_file', lconcomp_filter, 'in_file')
    workflow.connect(lconcomp_filter, 'out_file', dilation_filter, 'in_file')
    workflow.connect(dilation_filter, 'out_file', fill_filter, 'in_file')
    workflow.connect(fill_filter, 'out_file', output_node, 'mask_files')

    return workflow


def create_compute_suvr_pipeline(input_pet,
                                 input_mri,
                                 input_par,
                                 erode_ref,
                                 output_dir,
                                 name='compute_suvr',
                                 norm_region='cereb'):
    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_dir = output_dir
    workflow.base_output_dir = name

    # Merge all the parcelation into a binary image
    merge_roi = pe.MapNode(interface=niftyseg.UnaryMaths(),
                           name='merge_roi',
                           iterfield=['in_file'])
    merge_roi.inputs.in_file = input_par
    merge_roi.inputs.operation = 'bin'
    dilation = pe.MapNode(interface=niftyseg.BinaryMathsInteger(),
                          name='dilation',
                          iterfield=['in_file'])
    workflow.connect(merge_roi, 'out_file', dilation, 'in_file')
    dilation.inputs.operation = 'dil'
    dilation.inputs.operand_value = 5
    # generate a mask for the pet image
    mask_pet = create_mask_from_functional()
    mask_pet.inputs.input_node.in_files = input_pet

    # The structural image is first register to the pet image
    rigid_reg = pe.MapNode(interface=niftyreg.RegAladin(),
                           name='rigid_reg',
                           iterfield=['ref_file',
                                      'flo_file',
                                      'rmask_file',
                                      'fmask_file'])
    rigid_reg.inputs.rig_only_flag = True
    rigid_reg.inputs.verbosity_off_flag = True
    rigid_reg.inputs.v_val = 80
    rigid_reg.inputs.nosym_flag = False
    rigid_reg.inputs.ref_file = input_pet
    rigid_reg.inputs.flo_file = input_mri
    workflow.connect(mask_pet, 'output_node.mask_files', rigid_reg, 'rmask_file')
    workflow.connect(dilation, 'out_file', rigid_reg, 'fmask_file')
    # Propagate the ROIs into the pet space
    resampling = pe.MapNode(interface=niftyreg.RegResample(),
                            name='resampling',
                            iterfield=['ref_file', 'flo_file', 'trans_file'])
    resampling.inputs.inter_val = 'NN'
    resampling.inputs.verbosity_off_flag = True
    resampling.inputs.ref_file = input_pet
    resampling.inputs.flo_file = input_par
    workflow.connect(rigid_reg, 'aff_file', resampling, 'trans_file')
    # The PET image is normalised
    normalisation_workflow = create_regional_normalisation_pipeline(erode_ref=erode_ref)
    normalisation_workflow.inputs.input_node.input_files = input_pet
    workflow.connect(resampling, 'out_file', normalisation_workflow, 'input_node.input_rois')
    if norm_region == 'pons':
        roi_indices = [35]
    elif norm_region == 'gm_cereb':
        roi_indices = [39, 40,72,73,74]
    elif norm_region == 'wm_subcort':
        roi_indices = [45, 46]
    else:  # full cerebellum
        roi_indices = [39, 40, 41, 42, 72, 73, 74]
    normalisation_workflow.inputs.input_node.label_indices = roi_indices
    # The regional uptake are computed
    regional_average_workflow = create_regional_average_pipeline(output_dir=output_dir, neuromorphometrics=True)
    workflow.connect(normalisation_workflow, 'output_node.out_files',
                     regional_average_workflow, 'input_node.in_files')
    workflow.connect(resampling, 'out_file',
                     regional_average_workflow, 'input_node.in_rois')
    # Create an output node
    output_node = pe.Node(
        interface=niu.IdentityInterface(
            fields=['norm_files',
                    'suvr_files',
                    'tran_files',
                    'out_par_files']),
        name='output_node')
    workflow.connect(normalisation_workflow, 'output_node.out_files',
                     output_node, 'norm_files')
    workflow.connect(regional_average_workflow, 'output_node.out_files', output_node, 'suvr_files')
    workflow.connect(rigid_reg, 'aff_file', output_node, 'tran_files')
    workflow.connect(resampling, 'out_file', output_node, 'out_par_files')


    # Create a data sink
    ds = pe.Node(nio.DataSink(parameterization=False), name='data_sink')
    ds.inputs.base_directory = output_dir
    workflow.connect(output_node, 'norm_files', ds, '@norm_files')
    workflow.connect(output_node, 'tran_files', ds, '@tran_files')

    # Return the created workflow
    return workflow
