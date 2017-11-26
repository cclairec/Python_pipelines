# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import nipype.interfaces.niftyreg as niftyreg
from nipype.workflows.smri.niftyreg.groupwise import create_groupwise_average
from nipype.utils.filemanip import split_filename
from .tensor_processing import (merge_dwi_function, create_diffusion_mri_processing_workflow)
from ...interfaces.niftk.nodditoolbox import Noddi
from ...interfaces.niftk.utils import ProduceMask

def multiple_composition_function(in_trans_1, in_trans_2):
    import sys
    import os
    import nipype.pipeline.engine as pe
    import nipype.interfaces.niftyreg as niftyreg

    if len(in_trans_2) % len(in_trans_1) != 0:
        print('ERROR, lengths of lists must agree: %u -- %u' % (len(in_trans_1), len(in_trans_2)))
        sys.exit(1)

    shell_trans_list_size = len(in_trans_2) / len(in_trans_1)

    out_trans = []

    for i in range(len(in_trans_1)):
        in_trans_2_selection = in_trans_2[i * shell_trans_list_size:(i+1) * shell_trans_list_size]
        composer = pe.MapNode(interface=niftyreg.RegTransform(comp_input=in_trans_1[i]),
                              name='composer_'+str(i+1), iterfield=['comp_input2'])
        composer.inputs.comp_input2 = in_trans_2_selection
        composer.base_dir = os.getcwd()
        out_composer = composer.run().outputs
        for f in out_composer.out_file:
            out_trans.append(f)
    return out_trans



def create_matlab_noddi_workflow(in_dwis, in_bvals, in_bvecs, in_t1, output_dir,
                                 name='noddi_processing', in_fm_mag=None, in_fm_phase=None,
                                 in_nsf=100, in_susceptibility_params=[34.56, 2.46, '-y'], dwi_interp_type='CUB', in_matlabpoolsize=1,
                                 with_eddy=True,
                                 t1_mask= None):

    subject_id = split_filename(os.path.basename(in_dwis[0]))[1]
    number_of_shells = len(in_dwis)
    workflow = pe.Workflow(name=name)
    shell_dwis_merger = pe.Node(interface=niu.Merge(numinputs=number_of_shells), name='shell_dwis_merger')
    shell_b0s_merger = pe.Node(interface=niu.Merge(numinputs=number_of_shells), name='shell_b0s_merger')
    shell_trans_merger = pe.Node(interface=niu.Merge(numinputs=number_of_shells), name='shell_trans_merger')

    # for i in range(number_of_shells):
    #     shell_output_dir = os.path.join(output_dir, 'shell_'+str(i+1))
    #     if not os.path.exists(shell_output_dir):
    #         os.mkdir(shell_output_dir)
    #
    #     r = create_diffusion_mri_processing_workflow(
    #         susceptibility_correction_with_fm=(in_fm_mag is not None and in_fm_phase is not None),
    #         in_susceptibility_params=in_susceptibility_params, t1_mask_provided=False,
    #         name='dmri_workflow_shell_' + str(i + 1), resample_in_t1=False, log_data=True,
    #         dwi_interp_type=dwi_interp_type, wls_tensor_fit=False, rigid_only=False)
    #
    #     r.base_dir = shell_output_dir
    #     r.inputs.input_node.in_dwi_4d_file = in_dwis[i]
    #     r.inputs.input_node.in_bval_file = in_bvals[i]
    #     r.inputs.input_node.in_bvec_file = in_bvecs[i]
    #     r.inputs.input_node.in_t1_file = in_t1
    #
    #     if in_fm_mag is not None and in_fm_phase is not None:
    #         r.inputs.input_node.in_fm_magnitude_file = os.path.abspath(in_fm_mag)
    #         r.inputs.input_node.in_fm_phase_file = os.path.abspath(in_fm_phase)
    #
    #     ds = pe.Node(nio.DataSink(parameterization=False, base_directory=shell_output_dir), name='ds_shell_'+str(i+1))
    #     workflow.connect(r.get_node('renamer'), 'out_file', ds, '@outputs')
    #     workflow.connect(r.get_node('reorder_transformations'), 'out', ds, 'transformations')
    #
    #     dwi_splitter = pe.Node(interface=fsl.Split(dimension='t'), name='dwi_splitter'+str(i+1))
    #     workflow.connect(r, 'output_node.dwis', dwi_splitter, 'in_file')
    #     workflow.connect(r, 'output_node.b0', shell_b0s_merger, 'in'+str(i+1))
    #     workflow.connect(r, 'reorder_transformations.out', shell_trans_merger, 'in'+str(i+1))
    #     workflow.connect(dwi_splitter, 'out_files', shell_dwis_merger, 'in'+str(i+1))
    """Merge shells"""
    in_merger = pe.Node(interface=niu.Function(input_names=['in_dwis', 'in_bvals', 'in_bvecs'],
                                                     output_names=['out_dwis', 'out_bvals', 'out_bvecs'],
                                                     function=merge_dwi_function), name='in_merger')
    in_merger.inputs.in_dwis = in_dwis
    in_merger.inputs.in_bvals = in_bvals
    in_merger.inputs.in_bvecs = in_bvecs
    t1_mask_provided = t1_mask is not None
    r = create_diffusion_mri_processing_workflow(
             susceptibility_correction_with_fm=(in_fm_mag is not None and in_fm_phase is not None),
             in_susceptibility_params=in_susceptibility_params, t1_mask_provided=t1_mask_provided,
             name='tensor_proc', resample_in_t1=False, log_data=True,
             dwi_interp_type=dwi_interp_type, wls_tensor_fit=False, rigid_only=False, with_eddy=with_eddy)
    r.base_dir = "tensor"
    workflow.connect([(in_merger, r,
                     [('out_dwis', 'input_node.in_dwi_4d_file'),
                      ('out_bvecs', 'input_node.in_bvec_file'),
                      ('out_bvals', 'input_node.in_bval_file')])])
    r.inputs.input_node.in_t1_file = in_t1
    if t1_mask_provided:
        r.inputs.input_node.in_t1_mask_file = t1_mask
    if in_fm_mag is not None and in_fm_phase is not None:
        r.inputs.input_node.in_fm_magnitude_file = os.path.abspath(in_fm_mag)
        r.inputs.input_node.in_fm_phase_file = os.path.abspath(in_fm_phase)

    # groupwise_b0 = create_groupwise_average('groupwise_b0', itr_rigid=2, itr_affine=0, itr_non_lin=0)
    # workflow.connect(shell_b0s_merger, 'out', groupwise_b0, 'input_node.in_files')
    # ave_ims_b0 = pe.Node(interface=niftyreg.RegAverage(), name="ave_ims_b0")
    # workflow.connect(shell_b0s_merger, 'out', ave_ims_b0, 'avg_files')
    # workflow.connect(ave_ims_b0, 'out_file', groupwise_b0, 'input_node.ref_file')
    #
    # shell_trans_composer = pe.Node(interface=niu.Function(input_names=['in_trans_1', 'in_trans_2'],
    #                                                       output_names=['out_trans'],
    #                                                       function=multiple_composition_function),
    #                                name='shell_trans_composer')
    # workflow.connect(groupwise_b0, 'output_node.trans_files', shell_trans_composer, 'in_trans_1')
    # workflow.connect(shell_trans_merger, 'out', shell_trans_composer, 'in_trans_2')
    # dwi_resample = pe.MapNode(interface=niftyreg.RegResample(inter_val=dwi_interp_type),
    #                           name='dwi_resample', iterfield=['flo_file', 'trans_file'])
    # workflow.connect(groupwise_b0, 'output_node.average_image', dwi_resample, 'ref_file')
    # workflow.connect(shell_dwis_merger, 'out', dwi_resample, 'flo_file')
    # workflow.connect(shell_trans_composer, 'out_trans', dwi_resample, 'trans_file')
    #
    # dwi_merge = pe.Node(interface=fsl.Merge(dimension='t'), name='dwi_merge')
    # workflow.connect(dwi_resample, 'out_file', dwi_merge, 'in_files')
    # shell_bv_merger = pe.Node(interface=niu.Function(input_names=['in_dwis', 'in_bvals', 'in_bvecs'],
    #                                                  output_names=['out_dwis', 'out_bvals', 'out_bvecs'],
    #                                                  function=merge_dwi_function), name='shell_bv_merger')
    # shell_bv_merger.inputs.in_dwis = in_dwis
    # shell_bv_merger.inputs.in_bvals = in_bvals
    # shell_bv_merger.inputs.in_bvecs = in_bvecs
    #
    # b0_mask = pe.Node(interface=ProduceMask(), name='b0_mask')
    # workflow.connect(groupwise_b0, 'output_node.average_image', b0_mask, 'in_file')

    # b0_mask = pe.Node(interface=ProduceMask(), name='b0_mask')
    # workflow.connect(r, 'output_node.b0', b0_mask, 'in_file')

    noddi_fitting = pe.Node(interface=Noddi(in_fname=subject_id+'_noddi', noise_scaling_factor=in_nsf, matlabpoolsize=in_matlabpoolsize),
                            name='noddi_fitting')

    # workflow.connect(dwi_merge, 'merged_file', noddi_fitting, 'in_dwis')
    # workflow.connect(shell_bv_merger, 'out_bvals', noddi_fitting, 'in_bvals')
    # workflow.connect(shell_bv_merger, 'out_bvecs', noddi_fitting, 'in_bvecs')
    # workflow.connect(b0_mask, 'out_file', noddi_fitting, 'in_mask')
    workflow.connect(r, 'output_node.dwis', noddi_fitting, 'in_dwis')
    workflow.connect(r, 'output_node.bval', noddi_fitting, 'in_bvals')
    workflow.connect(r, 'output_node.bvec', noddi_fitting, 'in_bvecs')
    workflow.connect(r, 'output_node.mask', noddi_fitting, 'in_mask')

    ds = pe.Node(nio.DataSink(), name='ds')
    ds.inputs.base_directory = output_dir
    ds.inputs.parameterization = False

    workflow.connect(noddi_fitting, 'out_neural_density', ds, '@out_neural_density')
    workflow.connect(noddi_fitting, 'out_orientation_dispersion_index', ds, '@out_orientation_dispersion_index')
    workflow.connect(noddi_fitting, 'out_csf_volume_fraction', ds, '@out_csf_volume_fraction')
    workflow.connect(noddi_fitting, 'out_objective_function', ds, '@out_objective_function')
    workflow.connect(noddi_fitting, 'out_kappa_concentration', ds, '@out_kappa_concentration')
    workflow.connect(noddi_fitting, 'out_error', ds, '@out_error')
    workflow.connect(noddi_fitting, 'out_fibre_orientations_x', ds, '@out_fibre_orientations_x')
    workflow.connect(noddi_fitting, 'out_fibre_orientations_y', ds, '@out_fibre_orientations_y')
    workflow.connect(noddi_fitting, 'out_fibre_orientations_z', ds, '@out_fibre_orientations_z')

    workflow.connect(r, 'output_node.mask', ds, '@dwi_mask')
    workflow.connect(r, 'output_node.dwis', ds, '@corrected_dwis')
    workflow.connect(r, 'output_node.bval', ds, '@bvals')
    workflow.connect(r, 'output_node.bvec', ds, '@bvecs')

    return workflow
