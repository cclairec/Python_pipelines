# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.interfaces.niftyreg as niftyreg
import nipype.interfaces.niftyseg as niftyseg
import nipype.interfaces.fsl as fsl
from ...interfaces.niftk.labels.neuromorphometricslabels import NeuromorphometricsUpdateCsvFileWithLabels
from ...interfaces.niftk.labels.freesurferlabels import FreesurferUpdateCsvFileWithLabels
from ...interfaces.niftk.utils import (WriteArrayToCsv, MergeLabels, NormaliseImageWithROI)
from ...interfaces.niftk.stats import ExtractRoiStatistics


def create_regional_average_pipeline(output_dir,
                                     name='regional_average',
                                     in_trans=None,
                                     in_weights=None,
                                     in_label=None,
                                     in_threshold=None,
                                     freesurfer=False,
                                     neuromorphometrics=False):
    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_dir = output_dir
    workflow.base_output_dir = name
    # Create the input node interface
    input_node = pe.Node(
        interface=niu.IdentityInterface(
            fields=['in_files',
                    'in_rois',
                    'in_trans',
                    'in_weights'],
            mandatory_inputs=False),
        name='input_node')
    input_node.inputs.in_trans = in_trans
    input_node.inputs.in_weights = in_weights

    # Warp the parcelation in the input image space
    if in_trans is not None:
        resample = pe.MapNode(interface=niftyreg.RegResample(),
                              name='resampling',
                              iterfield=['ref_file',
                                         'flo_file',
                                         'trans_file'])
        workflow.connect(input_node, 'in_files', resample, 'ref_file')
        workflow.connect(input_node, 'in_rois', resample, 'flo_file')
        workflow.connect(input_node, 'in_trans', resample, 'trans_file')
        resample.inputs.inter_val = 'NN'
        resample.inputs.verbosity_off_flag = True
    # Compute the regional statistic
    average_iterfield = ['in_file', 'roi_file']
    if in_weights is not None:
        average_iterfield = ['in_file', 'roi_file', 'weight_file']
    compute_average = pe.MapNode(interface=ExtractRoiStatistics(),
                                 name='compute_average',
                                 iterfield=average_iterfield)
    workflow.connect(input_node, 'in_files', compute_average, 'in_file')
    if in_threshold is not None:
        compute_average.inputs.in_threshold = in_threshold

    if in_label is not None:
        compute_average.inputs.in_label = in_label
    if in_trans is not None:
        workflow.connect(resample, 'out_file', compute_average, 'roi_file')
    else:
        workflow.connect(input_node, 'in_rois', compute_average, 'roi_file')
    if in_weights is not None:
        workflow.connect(input_node, 'in_weights', compute_average, 'weight_file')
    # Save the result in a csv file
    generate_csv = pe.MapNode(interface=WriteArrayToCsv(),
                              name='generate_csv',
                              iterfield=['in_array', 'in_name'])
    workflow.connect(compute_average, 'out_array', generate_csv, 'in_array')
    workflow.connect(input_node, 'in_files', generate_csv, 'in_name')

    # The Freesurfer labels are added
    if freesurfer is True:
        add_fs_labels = pe.MapNode(interface=FreesurferUpdateCsvFileWithLabels(),
                                   name='add_fs_labels',
                                   iterfield=['in_file'])
        workflow.connect(generate_csv, 'out_file', add_fs_labels, 'in_file')
    elif neuromorphometrics is True:
        add_neuromorphometrics_labels = pe.MapNode(interface=NeuromorphometricsUpdateCsvFileWithLabels(),
                                                   name='add_neuromorphometrics_labels',
                                                   iterfield=['in_file'])
        workflow.connect(generate_csv, 'out_file', add_neuromorphometrics_labels, 'in_file')
    # Create the output node interface
    output_node = pe.Node(
        interface=niu.IdentityInterface(
            fields=['out_files',
                    'out_par_files']),
        name='output_node')
    if freesurfer is True:
        workflow.connect(add_fs_labels, 'out_file', output_node, 'out_files')
    elif neuromorphometrics is True:
        workflow.connect(add_neuromorphometrics_labels, 'out_file', output_node, 'out_files')
    else:
        workflow.connect(generate_csv, 'out_file', output_node, 'out_files')

    if in_trans is not None:
        workflow.connect(resample, 'out_file', output_node, 'out_par_files')
    else:
        workflow.connect(input_node, 'in_rois', output_node, 'out_par_files')

    # Create a data sink
    ds = pe.Node(nio.DataSink(parameterization=False), name='data_sink')
    ds.inputs.base_directory = output_dir
    workflow.connect(output_node, 'out_files', ds, '@out_files')
    workflow.connect(output_node, 'out_par_files', ds, '@out_par_files')

    # Return the created workflow
    return workflow



def create_regional_normalisation_pipeline(name='regional_normalisation',
                                           out_base_dir=os.getcwd(),erode_ref=False):
    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_dir = out_base_dir
    workflow.base_output_dir = name
    # Create the input node interface
    input_node = pe.Node(
        interface=niu.IdentityInterface(
            fields=['input_files',
                    'input_rois',
                    'label_indices']),
        name='input_node')
    # Combine the label into a single binary
    merge_labels = pe.MapNode(interface=MergeLabels(),
                              name='merge_labels',
                              iterfield=['in_file'])
    workflow.connect(input_node, 'input_rois', merge_labels, 'in_file')
    workflow.connect(input_node, 'label_indices', merge_labels, 'roi_list')


    # Normalise the input image
    normalise_image = pe.MapNode(interface=NormaliseImageWithROI(),
                                 name='normalise_image',
                                 iterfield=['in_file', 'roi_file'])
    workflow.connect(input_node, 'input_files', normalise_image, 'in_file')
    # If erode flag than stick an erosion between merge_labels and normalise_image
    if erode_ref:
        erosion = pe.MapNode(interface=niftyseg.BinaryMathsInteger(),
                             name='erosion',
                             iterfield=['in_file'])
        erosion.inputs.operation = 'ero'
        erosion.inputs.operand_value = 1
        workflow.connect(merge_labels, 'out_file', erosion, 'in_file')
        workflow.connect(erosion,'out_file', normalise_image, 'roi_file')
    else:
        workflow.connect(merge_labels, 'out_file', normalise_image, 'roi_file')
    # Create the output node interface
    output_node = pe.Node(
        interface=niu.IdentityInterface(
            fields=['out_files']),
        name='output_node')
    workflow.connect(normalise_image, 'out_file', output_node, 'out_files')
    # Return the created workflow
    return workflow


# Take an input mask and a list of transformations to be applied to it
# separately, apply each of them and combine the output. The intent is
# to allow things like having the union of affinely and nonlinearly
# registered versions of the same mask
def create_multiple_resample_and_combine_mask(name='resample_and_combine',
                                              out_base_dir=os.getcwd(),
                                              inter_val='LIN',
                                              psf_flag = True):
    # Inputs::
    #
    #     ::param inter_val:      niftyreg resampling parameter to use
    #     input_node.input_mask:             Mask to resample
    #     input_node.input_transforms:       Input transforms for reg_resample
    #     input_node.target_image:           Resampling ref image
    #
    # Outputs::
    #
    #     output_node.out_file:           Output combined resampled masks

    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_dir = out_base_dir
    workflow.base_output_dir = name
    # Create the input node interface
    input_node = pe.Node(
        interface=niu.IdentityInterface(
            fields=['input_mask',
                    'input_transforms',
                    'target_image']),
        name='input_node')

    # Resample to multiple outputs
    resample_node = pe.MapNode(interface=niftyreg.RegResample(), name="resample",
                               iterfield=['trans_file'])
    resample_node.inputs.inter_val = inter_val
    resample_node.inputs.psf_flag = psf_flag
    workflow.connect([(input_node,resample_node,
                       [('input_transforms','trans_file'),
                        ('input_mask','flo_file'),
                        ('target_image','ref_file')])])

    # Merge at this stage for easier processing of the resampled masks, then
    # do a reduce (max) after binarising.
    merge_node = pe.Node(interface=fsl.utils.Merge(dimension="t"), name="merge")
    workflow.connect(resample_node, 'out_file', merge_node, 'in_files')

    # N.B. if adding thresholding to this pipeline, only seg_maths seems safe
    # with floating point values
    # Using abs with no threshold is quite generous, but this is the aim of
    # this masking approach. Ideally should consider PSF too, but needs added
    # to the niftyreg interface
    abs_node = pe.Node(interface=fsl.maths.UnaryMaths(), name="abs")
    abs_node.inputs.nan2zeros = True
    abs_node.inputs.operation = 'abs'
    workflow.connect(merge_node, 'merged_file', abs_node, 'in_file')
    bin_node = pe.Node(interface=fsl.maths.UnaryMaths(), name="bin")
    bin_node.inputs.nan2zeros = True
    bin_node.inputs.operation = 'bin'
    workflow.connect(abs_node, 'out_file', bin_node, 'in_file')
    max_node = pe.Node(interface=fsl.maths.MaxImage(dimension="T"), name="max")
    workflow.connect(bin_node, 'out_file', max_node, 'in_file')

    # Output node interface
    output_node = pe.Node(interface=niu.IdentityInterface(fields=['out_file']),
                          name="output_node")
    workflow.connect(max_node, 'out_file', output_node, 'out_file')
    return workflow
