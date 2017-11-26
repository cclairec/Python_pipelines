# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import nipype.interfaces.niftyreg as niftyreg
import os
from ...interfaces.niftk.fmri import RestingStatefMRIPreprocess
from ...interfaces.niftk.qc import FmriQcPlot
from ..dmri.susceptibility_correction import create_fieldmap_susceptibility_workflow


def create_restingstatefmri_preprocessing_pipeline(in_fmri,
                                                   in_t1,
                                                   in_segmentation,
                                                   in_parcellation,
                                                   output_dir,
                                                   in_mag=None,
                                                   in_phase=None,
                                                   in_susceptibility_parameters=None,
                                                   name='restingstatefmri'):

    """Perform pre-processing steps for the resting state fMRI using AFNI

    Parameters
    ----------

    ::

      name : name of workflow (default: restingstatefmri)

    Inputs::

        in_fmri : functional runs into a single 4D image in NIFTI format
        in_t1 : The structural T1 image
        in_segmentation : The segmentation image containing 6 volumes (background, CSF, GM, WM, deep GM, brainstem),
        in the space of the fMRI image
        in_parcellation : The parcellation image coming out of the GIF parcellation algorithm
        output_dir : The output directory for the workflow
        in_mag : *OPT*, magnitude image to use for susceptibility correction (default: None)
        in_phase : *OPT*, phase image to use for susceptibility correction (default: None)
        in_susceptibility_parameters : *OPT*, susceptibility parameters
        (in a vector: read-out-time, echo time difference, phase encoding direction], default : None)
        name : *OPT*, name of the workflow (default : restingstatefmri)

    Outputs::


    Example
    -------

    >>> preproc = create_restingstatefmri_preprocessing_pipeline(in_fmri, in_t1, in_segmentation, in_parcellation, output_dir) # doctest: +SKIP
    >>> preproc.base_dir = '/tmp' # doctest: +SKIP
    >>> preproc.run() # doctest: +SKIP

    """

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    # We need to create an input node for the workflow
    input_node = pe.Node(
        niu.IdentityInterface(
            fields=['in_fmri',
                    'in_t1',
                    'in_segmentation',
                    'in_parcellation']),
        name='input_node')
    input_node.inputs.in_fmri = in_fmri
    input_node.inputs.in_t1 = in_t1
    input_node.inputs.in_segmentation = in_segmentation
    input_node.inputs.in_parcellation = in_parcellation

    resting_state_preproc = pe.Node(interface=RestingStatefMRIPreprocess(),
                                    name='resting_state_preproc')

    workflow.connect(input_node, 'in_fmri', resting_state_preproc, 'in_fmri')
    workflow.connect(input_node, 'in_t1', resting_state_preproc, 'in_t1')
    workflow.connect(input_node, 'in_segmentation', resting_state_preproc, 'in_tissue_segmentation')
    workflow.connect(input_node, 'in_parcellation', resting_state_preproc, 'in_parcellation')
    # fMRI QC plot
    plotter = pe.Node(interface=FmriQcPlot(),
                      name='plotter')
    workflow.connect(input_node, 'in_fmri', plotter, 'in_raw_fmri')
    workflow.connect(resting_state_preproc, 'out_raw_fmri_gm', plotter, 'in_raw_fmri_gm')
    workflow.connect(resting_state_preproc, 'out_raw_fmri_wm', plotter, 'in_raw_fmri_wm')
    workflow.connect(resting_state_preproc, 'out_raw_fmri_csf', plotter, 'in_raw_fmri_csf')
    workflow.connect(resting_state_preproc, 'out_mrp_file', plotter, 'in_mrp_file')
    workflow.connect(resting_state_preproc, 'out_spike_file', plotter, 'in_spike_file')
    workflow.connect(resting_state_preproc, 'out_rms_file', plotter, 'in_rms_file')
    # Output node
    output_node = pe.Node(interface=niu.IdentityInterface(fields=['out_corrected_fmri',
                                                                  'out_atlas_fmri',
                                                                  'out_fmri_to_t1_transformation',
                                                                  'out_raw_fmri_gm',
                                                                  'out_raw_fmri_wm',
                                                                  'out_raw_fmri_csf',
                                                                  'out_fmri_qc',
                                                                  'out_motioncorrected_file']),
                          name="output_node")
    workflow.connect(resting_state_preproc, 'out_corrected_fmri', output_node, 'out_corrected_fmri')
    workflow.connect(resting_state_preproc, 'out_atlas_fmri', output_node, 'out_atlas_fmri')
    workflow.connect(resting_state_preproc, 'out_fmri_to_t1_transformation', output_node,
                     'out_fmri_to_t1_transformation')
    workflow.connect(resting_state_preproc, 'out_raw_fmri_gm', output_node, 'out_raw_fmri_gm')
    workflow.connect(resting_state_preproc, 'out_raw_fmri_wm', output_node, 'out_raw_fmri_wm')
    workflow.connect(resting_state_preproc, 'out_raw_fmri_csf', output_node, 'out_raw_fmri_csf')
    workflow.connect(resting_state_preproc, 'out_motioncorrected_file', output_node, 'out_motioncorrected_file')
    workflow.connect(plotter, 'out_file', output_node, 'out_fmri_qc')

    ds = pe.Node(nio.DataSink(), name='ds')
    ds.inputs.base_directory = os.path.abspath(output_dir)
    ds.inputs.parameterization = False

    workflow.connect(output_node, 'out_fmri_to_t1_transformation', ds, '@fmri_to_t1_transformation')
    workflow.connect(output_node, 'out_atlas_fmri', ds, '@atlas_in_fmri')
    workflow.connect(output_node, 'out_raw_fmri_gm', ds, '@gm_in_fmri')
    workflow.connect(output_node, 'out_raw_fmri_wm', ds, '@wm_in_fmri')
    workflow.connect(output_node, 'out_raw_fmri_csf', ds, '@csf_in_fmri')
    workflow.connect(output_node, 'out_fmri_qc', ds, '@fmri_qc')

    if in_mag is None or in_phase is None or in_susceptibility_parameters is None:
        workflow.connect(output_node, 'out_motioncorrected_file', ds, '@motioncorrected_file')
        workflow.connect(output_node, 'out_corrected_fmri', ds, '@corrected_fmri')
    else:
        split_fmri = pe.Node(interface=fsl.Split(dimension='t'),
                             name='split_fmri')
        workflow.connect(output_node, 'out_motioncorrected_file', split_fmri, 'in_file')
        select_1st_fmri = pe.Node(interface=niu.Select(index=0),
                                  name='select_1st_fmri')
        workflow.connect(split_fmri, 'out_files', select_1st_fmri, 'inlist')
        binarise_parcellation = pe.Node(interface=fsl.UnaryMaths(operation='bin'),
                                        name='binarise_parcellation')
        workflow.connect(input_node, 'in_parcellation', binarise_parcellation, 'in_file')

        # Perform susceptibility correction, where we already have a mask in the b0 space
        susceptibility_correction = create_fieldmap_susceptibility_workflow('susceptibility_correction',
                                                                            reg_to_t1=True)
        susceptibility_correction.inputs.input_node.mag_image = os.path.abspath(in_mag)
        susceptibility_correction.inputs.input_node.phase_image = os.path.abspath(in_phase)
        susceptibility_correction.inputs.input_node.t1 = os.path.abspath(in_t1)
        susceptibility_correction.inputs.input_node.rot = in_susceptibility_parameters[0]
        susceptibility_correction.inputs.input_node.etd = in_susceptibility_parameters[1]
        susceptibility_correction.inputs.input_node.ped = in_susceptibility_parameters[2]
        workflow.connect(select_1st_fmri, 'out', susceptibility_correction, 'input_node.epi_image')
        workflow.connect(binarise_parcellation, 'out_file', susceptibility_correction, 'input_node.t1_mask')

        fmri_corrected_resample = pe.Node(interface=niftyreg.RegResample(inter_val='LIN'),
                                          name='fmri_corrected_resample')
        workflow.connect(output_node, 'out_corrected_fmri', fmri_corrected_resample, 'ref_file')
        workflow.connect(output_node, 'out_corrected_fmri', fmri_corrected_resample, 'flo_file')
        workflow.connect(susceptibility_correction.get_node('output_node'), 'out_field',
                         fmri_corrected_resample, 'trans_file')
        workflow.connect(fmri_corrected_resample, 'out_file', ds, '@corrected_fmri')
        fmri_motion_corrected_resample = pe.Node(interface=niftyreg.RegResample(inter_val='LIN'),
                                                 name='fmri_motion_corrected_resample')

        workflow.connect(output_node, 'out_motioncorrected_file', fmri_motion_corrected_resample, 'ref_file')
        workflow.connect(output_node, 'out_motioncorrected_file', fmri_motion_corrected_resample, 'flo_file')
        workflow.connect(susceptibility_correction.get_node('output_node'), 'out_field',
                         fmri_motion_corrected_resample, 'trans_file')
        workflow.connect(fmri_motion_corrected_resample, 'out_file', ds, '@motioncorrected_file')

    return workflow
