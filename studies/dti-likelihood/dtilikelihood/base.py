# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
    Interfaces for the Diffusion Likelihood study
"""

import nipype.interfaces.utility as niu  # utility
import nipype.pipeline.engine as pe  # pipeline engine
import nipype.interfaces.niftyseg as niftyseg
import nipype.interfaces.niftyreg as niftyreg
import nipype.interfaces.niftyfit as niftyfit
import nipype.interfaces.fsl as fsl
import nipype.interfaces.io as nio

import niftk.diffusion as dmri
import dtilikelihood.postprocessing as dmripostproc
import dtilikelihood.statistics as add_noise

'''
This file provides the creation of the whole workflow necessary for 
processing diffusion MRI images.
'''


def create_dti_likelihood_study_workflow(name='create_dti_reproducibility_study', log_data=False, dwi_interp_type='CUB',
                                         result_dir=None):
    """
    Example
    -------

    # >>> dti_repr_st = create_dti_reproducibility_study_workflow(name='dti_repr_st')
    # >>> dti_repr_st.inputs.


    Inputs::

        input_node.
    Outputs::

        output_node.


    Optional arguments::
        
    """

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    # We need to create an input node for the workflow
    input_node = pe.Node(
        niu.IdentityInterface(
            fields=['in_tensors_file',
                    'in_b0_file',
                    'in_bvec_file',
                    'in_bval_file',
                    'in_t1_file',
                    'in_mask_file',
                    'in_labels_file',
                    'in_stddev_translation',
                    'in_stddev_rotation',
                    'in_stddev_shear',
                    'in_noise_sigma']),
        name='input_node')

    distortion_generator = pe.Node(interface=dmri.DistortionGenerator(),
                                   name='distortion_generator')

    tensor_resampling = pe.MapNode(interface=niftyreg.RegResample(inter_val='LIN'),
                                   name='tensor_resampling',
                                   iterfield=['trans_file'])
    tensor_resampling.inputs.tensor_flag = True

    b0_resampling = pe.MapNode(interface=niftyreg.RegResample(),
                               name='b0_resampling',
                               iterfield=['trans_file'])
    mask_resampling = pe.MapNode(interface=niftyreg.RegResample(),
                                 name='mask_resampling',
                                 iterfield=['trans_file'])
    mask_resampling.inputs.inter_val = 'NN'

    tensor_2_dwi = pe.MapNode(interface=niftyfit.DwiTool(dti_flag2=True),
                              name='tensor_2_dwi',
                              iterfield=['source_file', 'b0_file', 'bval_file', 'bvec_file'])
    tensor_2_dwi.inputs.bvallowthreshold = 0.5

    noise_adder = pe.MapNode(interface=add_noise.NoiseAdder(noise_type='rician'),
                             name='noise_adder',
                             iterfield=['in_file', 'mask_file'])

    merge_dwis = pe.Node(interface=fsl.Merge(dimension='t'),
                         direction='t',
                         name='merge_dwis')

    r = dmri.create_diffusion_mri_processing_workflow(name='dmri_workflow',
                                                      correct_susceptibility=False,
                                                      t1_mask_provided=True,
                                                      ref_b0_provided=True,
                                                      wls_tensor_fit=False,
                                                      log_data=log_data,
                                                      dwi_interp_type=dwi_interp_type)
    r.get_node('diffusion_model_fitting_tensor').inputs.bvallowthreshold = 0.5
    r.get_node('find_B0s').inputs.b0_threshold = 0.5
    r.get_node('find_DWIs').inputs.b0_threshold = 0.5
    r.get_node('reorder_transformations').inputs.b0_threshold = 0.5

    inv_estimated_distortions = pe.MapNode(interface=niftyreg.RegTransform(),
                                           name='inv_estimated_distortions',
                                           iterfield=['inv_aff_input'])

    tensor_resampling_2 = pe.MapNode(interface=niftyreg.RegResample(),
                                     name='tensor_resampling_2',
                                     iterfield=['trans_file'])
    tensor_resampling_2.inputs.tensor_flag = True

    b0_resampling_2 = pe.MapNode(interface=niftyreg.RegResample(),
                                 name='b0_resampling_2',
                                 iterfield=['trans_file'])

    tensor_2_dwi_2 = pe.MapNode(interface=niftyfit.DwiTool(dti_flag2=True),
                                name='tensor_2_dwi_2',
                                iterfield=['source_file', 'b0_file', 'bval_file', 'bvec_file'])
    tensor_2_dwi_2.inputs.bvallowthreshold = 0.5

    merge_dwis_2 = pe.Node(interface=fsl.Merge(dimension='t'),
                           direction='t',
                           name='merge_dwis_2')

    threshold = pe.Node(interface=fsl.Threshold(direction='above', thresh=1),
                        name='threshold')

    postproc = dmripostproc.create_dti_likelihood_post_proc_workflow(name='postproc')

    # Output node
    output_node = pe.Node(interface=niu.IdentityInterface(
        fields=['tensor_metric_map',
                'tensor_metric_ROI_statistics',
                'dwi_metric_map',
                'dwi_metric_ROI_statistics',
                'affine_distances',
                'proc_residual_metric_ROI_statistics']),
        name="output_node")

    # Generate Distortions
    workflow.connect(input_node, 'in_stddev_translation', distortion_generator, 'stddev_translation_val')
    workflow.connect(input_node, 'in_stddev_rotation', distortion_generator, 'stddev_rotation_val')
    workflow.connect(input_node, 'in_stddev_shear', distortion_generator, 'stddev_shear_val')
    workflow.connect(input_node, 'in_bval_file', distortion_generator, 'bval_file')
    workflow.connect(input_node, 'in_bvec_file', distortion_generator, 'bvec_file')

    # Resample tensors
    workflow.connect(input_node, 'in_tensors_file', tensor_resampling, 'ref_file')
    workflow.connect(input_node, 'in_tensors_file', tensor_resampling, 'flo_file')
    workflow.connect(distortion_generator, 'aff_files', tensor_resampling, 'trans_file')

    # Resample B0s the same way as the distorted tensor
    workflow.connect(input_node, 'in_b0_file', b0_resampling, 'flo_file')
    workflow.connect(input_node, 'in_b0_file', b0_resampling, 'ref_file')
    workflow.connect(distortion_generator, 'aff_files', b0_resampling, 'trans_file')

    # Make distortedDWI using the the affine distorted tensors and B0s
    workflow.connect(tensor_resampling, 'res_file', tensor_2_dwi, 'source_file')
    workflow.connect(distortion_generator, 'bval_files', tensor_2_dwi, 'bval_file')
    workflow.connect(distortion_generator, 'bvec_files', tensor_2_dwi, 'bvec_file')
    workflow.connect(b0_resampling, 'res_file', tensor_2_dwi, 'b0_file')

    workflow.connect(input_node, 'in_mask_file', mask_resampling, 'flo_file')
    workflow.connect(input_node, 'in_mask_file', mask_resampling, 'ref_file')
    workflow.connect(distortion_generator, 'aff_files', mask_resampling, 'trans_file')

    # Add noise
    workflow.connect(input_node, 'in_noise_sigma', noise_adder, 'sigma_val')
    workflow.connect(tensor_2_dwi, 'syn_file', noise_adder, 'in_file')
    workflow.connect(mask_resampling, 'res_file', noise_adder, 'mask_file')

    # Merge noisy distorted DWI
    workflow.connect(noise_adder, 'out_file', merge_dwis, 'in_files')

    # Now perform the diffusion pre-processing pipeline
    workflow.connect(merge_dwis, 'merged_file', r, 'input_node.in_dwi_4d_file')
    workflow.connect(input_node, 'in_bvec_file', r, 'input_node.in_bvec_file')
    workflow.connect(input_node, 'in_bval_file', r, 'input_node.in_bval_file')
    workflow.connect(input_node, 'in_t1_file', r, 'input_node.in_t1_file')
    workflow.connect(input_node, 'in_mask_file', r, 'input_node.in_t1_mask')
    workflow.connect(input_node, 'in_b0_file', r, 'input_node.in_ref_b0')

    # Take the final estimated transformation of the DWI, and invert the transformation
    workflow.connect(r, 'output_node.transformations', inv_estimated_distortions, 'inv_aff_input')

    # Resample the tensor to the space of each of the observed DWI
    workflow.connect(r, 'output_node.tensor', tensor_resampling_2, 'ref_file')
    workflow.connect(r, 'output_node.tensor', tensor_resampling_2, 'flo_file')
    workflow.connect(inv_estimated_distortions, 'out_file', tensor_resampling_2, 'trans_file')

    # Resample the averageB0 to the sapce of the observed DWI for prediction    
    workflow.connect(r, 'output_node.average_b0', b0_resampling_2, 'ref_file')
    workflow.connect(r, 'output_node.average_b0', b0_resampling_2, 'flo_file')
    workflow.connect(inv_estimated_distortions, 'out_file', b0_resampling_2, 'trans_file')

    # Predict the DWI for this particular b-vector/b-value pair
    workflow.connect(tensor_resampling_2, 'res_file', tensor_2_dwi_2, 'source_file')
    workflow.connect(distortion_generator, 'bval_files', tensor_2_dwi_2, 'bval_file')
    workflow.connect(distortion_generator, 'bvec_files', tensor_2_dwi_2, 'bvec_file')
    workflow.connect(b0_resampling_2, 'res_file', tensor_2_dwi_2, 'b0_file')

    # Merge back the predicted DWI
    workflow.connect(tensor_2_dwi_2, 'syn_file', merge_dwis_2, 'in_files')

    # Perform Post Processing Measurements
    workflow.connect(merge_dwis, 'merged_file', postproc, 'input_node.simulated_dwis')
    workflow.connect(merge_dwis_2, 'merged_file', postproc, 'input_node.estimated_dwis')
    workflow.connect(input_node, 'in_tensors_file', postproc, 'input_node.simulated_tensors')
    workflow.connect(r, 'output_node.tensor', postproc, 'input_node.estimated_tensors')
    workflow.connect(r, 'output_node.residual_image_tensor', threshold, 'in_file')
    workflow.connect(threshold, 'out_file', postproc, 'input_node.proc_residual_image')
    workflow.connect(input_node, 'in_labels_file', postproc, 'input_node.labels_file')

    workflow.connect(inv_estimated_distortions, 'out_file', postproc, 'input_node.estimated_affines')
    workflow.connect(distortion_generator, 'aff_files', postproc, 'input_node.simulated_affines')

    # Propagate results into the output node
    workflow.connect(postproc, 'output_node.tensor_metric_map', output_node, 'tensor_metric_map')
    workflow.connect(postproc, 'output_node.tensor_metric_ROI_statistics', output_node, 'tensor_metric_ROI_statistics')
    workflow.connect(postproc, 'output_node.dwi_metric_map', output_node, 'dwi_metric_map')
    workflow.connect(postproc, 'output_node.dwi_metric_ROI_statistics', output_node, 'dwi_metric_ROI_statistics')
    workflow.connect(postproc, 'output_node.affine_distances', output_node, 'affine_distances')
    workflow.connect(postproc, 'output_node.proc_residual_metric_ROI_statistics', output_node,
                     'proc_residual_metric_ROI_statistics')

    if result_dir is not None:
        ds = pe.Node(nio.DataSink(), name='sinker')
        ds.inputs.base_directory = result_dir
        ds.inputs.container = name
        workflow.connect(output_node, 'tensor_metric_ROI_statistics', ds, '@tensor_stats')
        workflow.connect(output_node, 'tensor_metric_map', ds, '@tensor_residual')
        workflow.connect(output_node, 'dwi_metric_ROI_statistics', ds, '@dwi_stats')
        workflow.connect(output_node, 'dwi_metric_map', ds, '@dwi_residual')
        workflow.connect(output_node, 'affine_distances', ds, '@affine_distances')
        workflow.connect(output_node, 'proc_residual_metric_ROI_statistics', ds, '@proc_residual_stats')
    return workflow
