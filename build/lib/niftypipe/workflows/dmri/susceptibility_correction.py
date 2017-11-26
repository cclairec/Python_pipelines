# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from nipype.interfaces.susceptibility import GenFm, PhaseUnwrap, PmScale
from nipype.interfaces.niftyreg import RegAladin, RegTransform, RegResample, RegF3D, RegJacobian
from nipype.interfaces.niftyseg import BinaryMaths
from niftypipe.interfaces.niftk.utils import ProduceMask
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu


def create_registration_susceptibility_workflow(name='nrr_susceptibility'):

    """
    Create a workflow that perform EPI distortion correction using non-linear registration
    to a T1 image.

    Inputs::

        input_node.epi_image - The EPI distorted diffusion image
        input_node.t1 - A downsampled t1 image in the epi image space
        input_node.t1_mask - Mask image in the epi_image space

    Outputs::

        output_node.out_field - The deformation field that undoes the magnetic susceptibility
        distortion in the space of the epi image
        output_node.out_jac - The thresholded (to remove negative Jacobians) jacobian map of the
        corrective field
        output_node.out_epi - The distortion corrected, and modulated by the thresholded Jacobian, epi image

    """

    # Create the input and output nodes
    input_node = pe.Node(niu.IdentityInterface(
        fields=['epi_image',
                't1',
                't1_mask']),
        name='input_node')

    output_node = pe.Node(niu.IdentityInterface(
        fields=['out_field',
                'out_jac',
                'out_epi']),
        name='output_node')

    # Create the workflow
    pipeline = pe.Workflow(name=name)
    pipeline.base_output_dir = name

    # Apply the t1 mask to the t1 image
    apply_mask_t1 = pe.Node(interface=BinaryMaths(operation='mul'),
                            name='apply_mask_t1')
    pipeline.connect(input_node, 't1', apply_mask_t1, 'in_file')
    pipeline.connect(input_node, 't1_mask', apply_mask_t1, 'operand_file')

    # Apply a mask to the EPI image
    get_mask_epi = pe.Node(interface=ProduceMask(use_nrr=True),
                           name='get_mask_epi')
    apply_mask_epi = pe.Node(interface=BinaryMaths(operation='mul'),
                             name='apply_mask_epi')
    pipeline.connect(input_node, 'epi_image', get_mask_epi, 'in_file')
    pipeline.connect(input_node, 'epi_image', apply_mask_epi, 'in_file')
    pipeline.connect(get_mask_epi, 'out_file', apply_mask_epi, 'operand_file')

    # Registration from the T1 to the EPI
    reg_correction = pe.Node(interface=RegF3D(**{'nox_flag': True, 'noz_flag': True}),
                             name='reg_correction')
    reg_correction.inputs.lncc_val = -1.5
    reg_correction.inputs.maxit_val = 150
    reg_correction.inputs.be_val = 0.1
    reg_correction.inputs.vel_flag = True
    pipeline.connect(apply_mask_t1, 'out_file', reg_correction, 'ref_file')
    pipeline.connect(apply_mask_epi, 'out_file', reg_correction, 'flo_file')

    # Compute the Jacobian map
    reg_jacobian = pe.Node(interface=RegJacobian(), name='calc_transform_jac')
    pipeline.connect(input_node, 'epi_image', reg_jacobian, 'ref_file')
    pipeline.connect(reg_correction, 'cpp_file', reg_jacobian, 'trans_file')

    # Generate the deformation field
    def_field = pe.Node(interface=RegTransform(), name='def_field')
    pipeline.connect(input_node, 't1', def_field, 'ref1_file')
    pipeline.connect(reg_correction, 'cpp_file', def_field, 'def_input')

    # Threshold the Jacobian determinant map
    thr_jac_1 = pe.Node(interface=BinaryMaths(operation='sub', operand_value=0.2), name='thr_jac_1')
    pipeline.connect(reg_jacobian, 'out_file', thr_jac_1, 'in_file')
    thr_jac_2 = pe.Node(interface=fsl.Threshold(thresh=0.0, direction='below'), name='thr_jac_2')
    pipeline.connect(thr_jac_1, 'out_file', thr_jac_2, 'in_file')
    thr_jac_3 = pe.Node(interface=BinaryMaths(operation='add', operand_value=0.2), name='thr_jac_3')
    pipeline.connect(thr_jac_2, 'out_file', thr_jac_3, 'in_file')

    # Modulate the EPI image
    modulate_jac = pe.Node(interface=BinaryMaths(operation='mul'), name='modulate_jac')
    pipeline.connect(reg_correction, 'res_file', modulate_jac, 'in_file')
    pipeline.connect(thr_jac_3, 'out_file', modulate_jac, 'operand_file')

    # Fill out the information in the output node
    pipeline.connect(def_field, 'out_file', output_node, 'out_field')
    pipeline.connect(thr_jac_3, 'out_file', output_node, 'out_jac')
    pipeline.connect(modulate_jac, 'out_file', output_node, 'out_epi')

    # Return the workflow
    return pipeline


def create_fieldmap_susceptibility_workflow(name='fm_susceptibility',
                                            reg_to_t1=False):
    """Creates a workflow that perform EPI distortion correction using field
    maps and possibly T1 images.

    Example
    -------

    >>> susceptibility_correction = create_fieldmap_susceptibility_workflow(name='susceptibility_workflow')
    >>> susceptibility_correction.inputs.input_node.etd = 2.46
    >>> susceptibility_correction.inputs.input_node.rot = 34.56
    >>> susceptibility_correction.inputs.input_node.ped = '-y'


    Inputs::

        input_node.epi_image - The EPI distorted diffusion image
        input_node.phase_image - The phase difference image of the fieldmap
        input_node.mag_image - The magnitude of the fieldmap (must be single image)
        input_node.etd - The echo time difference in msec (generally 2.46 msec for siemens scanners)
        input_node.rot - The read out time in msec (34.56 msec for standard DRC acquisitions)
        input_node.ped - The phase encode direction (-y for standard DRC acquisition)
        input_node.t1 - A T1 image in the epi image physical space space (only used when reg_to_t1 = True)
        input_node.t1_mask - Mask image in the T1 discretised space space (only used where mask_exists = True)


    Outputs::

        output_node.out_field - The deformation field that undoes the magnetic susceptibility
        distortion in the space of the epi image
        output_node.out_jac - The thresholded (to remove negative Jacobians) jacobian map of the
        corrective field
        output_node.out_epi - The distortion corrected, and modulated by the thresholded Jacobian, epi image


    Optional arguments::
        reg_to_t1 - include a step to non-linearly register the field map corrected
        image to the T1 space to refine the correction.


    """
    input_node = pe.Node(niu.IdentityInterface(
        fields=['epi_image',
                'phase_image',
                'mag_image',
                'etd',
                'ped',
                'rot',
                't1',
                't1_mask']),
        name='input_node')

    # create nodes to estimate the deformation field from the field map images
    pm_scale = pe.Node(interface=PmScale(), name='pm_scale')
    pm_unwrap = pe.Node(interface=PhaseUnwrap(), name='phase_unwrap')
    gen_fm = pe.Node(interface=GenFm(), name='gen_fm')

    # Create nodes to register the field map deformation field
    reg_fm_to_b0 = pe.Node(interface=RegAladin(rig_only_flag=True, ln_val=2, verbosity_off_flag=True),
                           name='reg_fm_to_b0')
    reg_b0_to_fm = pe.Node(interface=RegTransform(), name='reg_b0_to_fm')
    resample_mask_in_phase = pe.Node(interface=RegResample(inter_val='NN'), name='resample_mask_in_phase')
    comp_aff_1 = pe.Node(interface=RegTransform(), name='comp_aff_1')
    comp_aff_2 = pe.Node(interface=RegTransform(), name='comp_aff_2')

    resample_epi_in_phase = pe.Node(interface=RegResample(), name='resample_epi_in_phase')
    resample_epi = pe.Node(interface=RegResample(), name='resample_epi')
    reg_jacobian = pe.Node(interface=RegJacobian(), name='calc_transform_jac')
    thr_jac_1 = pe.Node(interface=BinaryMaths(operation='sub', operand_value=0.2), name='thr_jac_1')
    thr_jac_2 = pe.Node(interface=fsl.Threshold(thresh=0.0, direction='below'), name='thr_jac_2')
    thr_jac_3 = pe.Node(interface=BinaryMaths(operation='add', operand_value=0.2), name='thr_jac_3')
    modulate_jac = pe.Node(interface=BinaryMaths(operation='mul'), name='modulate_jac')

    output_node = pe.Node(niu.IdentityInterface(
        fields=['out_field', 'out_epi', 'out_jac']),
        name='output_node')

    pipeline = pe.Workflow(name=name)
    pipeline.base_output_dir = name

    # Registration from the epi to to the magnitude image
    pipeline.connect(input_node, 'mag_image', reg_fm_to_b0, 'ref_file')
    pipeline.connect(input_node, 'epi_image', reg_fm_to_b0, 'flo_file')
    pipeline.connect(reg_fm_to_b0, 'aff_file', reg_b0_to_fm, 'inv_aff_input')

    # Resample the t1 mask in the magnitude image space
    pipeline.connect(input_node, 'mag_image', resample_mask_in_phase, 'ref_file')
    pipeline.connect(input_node, 't1_mask', resample_mask_in_phase, 'flo_file')
    pipeline.connect(reg_fm_to_b0, 'aff_file', resample_mask_in_phase, 'trans_file')

    # Unwrap the phase image
    pipeline.connect(input_node, 'phase_image', pm_scale, 'in_pm')
    pipeline.connect(pm_scale, 'out_pm', pm_unwrap, 'in_fm')
    pipeline.connect(input_node, 'mag_image', pm_unwrap, 'in_mag')
    pipeline.connect(resample_mask_in_phase, 'out_file', pm_unwrap, 'in_mask')

    # Resample the epi in the phase space
    pipeline.connect(input_node, 'epi_image', resample_epi_in_phase, 'flo_file')
    pipeline.connect(input_node, 'mag_image', resample_epi_in_phase, 'ref_file')
    pipeline.connect(reg_fm_to_b0, 'aff_file', resample_epi_in_phase, 'trans_file')

    # Generate the deformation field from the fieldmap
    pipeline.connect(resample_epi_in_phase, 'out_file', gen_fm, 'in_epi')
    pipeline.connect(input_node, 'etd', gen_fm, 'in_etd')
    pipeline.connect(input_node, 'rot', gen_fm, 'in_rot')
    pipeline.connect(input_node, 'ped', gen_fm, 'in_ped')
    pipeline.connect(pm_unwrap, 'out_fm', gen_fm, 'in_ufm')
    pipeline.connect(resample_mask_in_phase, 'out_file', gen_fm, 'in_mask')

    # Move the deformation field from the phase to the b0 image space
    pipeline.connect(reg_b0_to_fm, 'out_file', comp_aff_1, 'comp_input')
    pipeline.connect(input_node, 'mag_image', comp_aff_1, 'ref1_file')
    pipeline.connect(gen_fm, 'out_field', comp_aff_1, 'comp_input2')
    pipeline.connect(comp_aff_1, 'out_file', comp_aff_2, 'comp_input')
    pipeline.connect(reg_fm_to_b0, 'aff_file', comp_aff_2, 'comp_input2')

    # Resample the epi image using the new deformation
    pipeline.connect(input_node, 'epi_image', resample_epi, 'flo_file')
    pipeline.connect(input_node, 'epi_image', resample_epi, 'ref_file')
    pipeline.connect(comp_aff_2, 'out_file', resample_epi, 'trans_file')

    if reg_to_t1:
        reg_refine_fm_correction = pe.Node(interface=RegF3D(**{'nox_flag': True, 'noz_flag': True}),
                                           name='reg_refine_fm_correction')
        reg_refine_fm_correction.inputs.lncc_val = -1.5
        reg_refine_fm_correction.inputs.maxit_val = 150
        reg_refine_fm_correction.inputs.be_val = 0.1
        reg_refine_fm_correction.inputs.vel_flag = True
        comp_def = pe.Node(interface=RegTransform(), name='comp_def')
        resample_epi_2 = pe.Node(interface=RegResample(), name='resample_epi_2')
        reg_jacobian_2 = pe.Node(interface=RegJacobian(), name='reg_jacobian_2')
        resample_jac = pe.Node(interface=RegResample(), name='resample_jac')
        modulate_jac_2 = pe.Node(interface=BinaryMaths(operation='mul'), name='modulate_jac_2')
        apply_mask_t1 = pe.Node(interface=BinaryMaths(operation='mul'),
                                name='apply_mask_t1')
        get_mask_epi = pe.Node(interface=ProduceMask(use_nrr=True),
                               name='get_mask_epi')
        apply_mask_epi = pe.Node(interface=BinaryMaths(operation='mul'),
                                 name='apply_mask_epi')

        # Compute the Jacobian resulting from the field map
        pipeline.connect(comp_aff_2, 'out_file', reg_jacobian_2, 'trans_file')

        # Resample the jacobian in the epi image space
        pipeline.connect(input_node, 'epi_image', resample_jac, 'ref_file')
        pipeline.connect(reg_jacobian_2, 'out_file', resample_jac, 'flo_file')

        # Modulate the intially deformed epi image
        pipeline.connect(resample_epi, 'out_file', modulate_jac_2, 'in_file')
        pipeline.connect(resample_jac, 'out_file', modulate_jac_2, 'operand_file')

        # Apply the mask to the t1 image
        pipeline.connect(input_node, 't1', apply_mask_t1, 'in_file')
        pipeline.connect(input_node, 't1_mask', apply_mask_t1, 'operand_file')

        # Compute the mask of the modulated epi image
        pipeline.connect(modulate_jac_2, 'out_file', get_mask_epi, 'in_file')
        pipeline.connect(modulate_jac_2, 'out_file', apply_mask_epi, 'in_file')
        pipeline.connect(get_mask_epi, 'out_file', apply_mask_epi, 'operand_file')

        # Run the non rigid registration between the T1w and the B0
        pipeline.connect(apply_mask_t1, 'out_file', reg_refine_fm_correction, 'ref_file')
        pipeline.connect(apply_mask_epi, 'out_file', reg_refine_fm_correction, 'flo_file')

        # Compose the FM derived and NRR derived transformations
        pipeline.connect(reg_refine_fm_correction, 'cpp_file', comp_def, 'comp_input')
        pipeline.connect(comp_aff_2, 'out_file', comp_def, 'comp_input2')
        pipeline.connect(input_node, 'epi_image', comp_def, 'ref1_file')

        # Resample the B0 image
        pipeline.connect(input_node, 'epi_image', resample_epi_2, 'flo_file')
        pipeline.connect(input_node, 'epi_image', resample_epi_2, 'ref_file')
        pipeline.connect(comp_def, 'out_file', resample_epi_2, 'trans_file')

        # Compute the Jacobian determinant map from the composed transformation
        pipeline.connect(comp_def, 'out_file', reg_jacobian, 'trans_file')
        pipeline.connect(comp_def, 'out_file', output_node, 'out_field')

        # Set up the Jacobian modulation of the B0
        pipeline.connect(resample_epi_2, 'out_file', modulate_jac, 'in_file')

    else:
        # Set up the Jacobian modulation of the B0
        pipeline.connect(comp_aff_2, 'out_file', output_node, 'out_field')
        pipeline.connect(comp_aff_2, 'out_file', reg_jacobian, 'trans_file')
        pipeline.connect(resample_epi, 'out_file', modulate_jac, 'in_file')

    # Threshold the Jacobian determinant of the transformation
    pipeline.connect(reg_jacobian, 'out_file', thr_jac_1, 'in_file')
    pipeline.connect(thr_jac_1, 'out_file', thr_jac_2, 'in_file')
    pipeline.connect(thr_jac_2, 'out_file', thr_jac_3, 'in_file')

    # Divide the resampled epi image by the Jacobian image
    pipeline.connect(thr_jac_3, 'out_file', modulate_jac, 'operand_file')

    # Fill out the information in the output node
    pipeline.connect(modulate_jac, 'out_file', output_node, 'out_epi')
    pipeline.connect(thr_jac_3, 'out_file', output_node, 'out_jac')

    return pipeline
