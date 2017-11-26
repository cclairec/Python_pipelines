# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.interfaces.utility as niu
import nipype.interfaces.niftyreg as niftyreg
import nipype.interfaces.niftyseg as niftyseg
import nipype.interfaces.niftyfit as niftyfit
import nipype.interfaces.fsl as fsl
from nipype.utils.filemanip import split_filename

'''
This file provides the creation of the whole workflow necessary for
processing ASL MRI images.
'''


def create_asl_processing_workflow(in_inversion_recovery_file,
                                   in_asl_file,
                                   output_dir,
                                   in_t1_file=None,
                                   name='asl_processing_workflow'):

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    subject_id = split_filename(os.path.basename(in_asl_file))[1]

    ir_splitter = pe.Node(interface=fsl.Split(dimension='t', out_base_name='out_', in_file=in_inversion_recovery_file),
                          name='ir_splitter')
    ir_selector = pe.Node(interface=niu.Select(index=[0, 2, 4]), name='ir_selector')
    workflow.connect(ir_splitter, 'out_files', ir_selector, 'inlist')
    ir_merger = pe.Node(interface=fsl.Merge(dimension='t'), name='ir_merger')
    workflow.connect(ir_selector, 'out', ir_merger, 'in_files')
    fitqt1 = pe.Node(interface=niftyfit.FitQt1(TIs=[4, 2, 1], SR=True),
                     name='fitqt1')
    workflow.connect(ir_merger, 'merged_file', fitqt1, 'source_file')
    extract_ir_0 = pe.Node(interface=niftyseg.BinaryMathsInteger(operation='tp', operand_value=0,
                                                                 in_file=in_inversion_recovery_file),
                           name='extract_ir_0')
    ir_thresolder = pe.Node(interface=fsl.Threshold(thresh=250),
                            name='ir_thresolder')
    workflow.connect(extract_ir_0, 'out_file', ir_thresolder, 'in_file')
    create_mask = pe.Node(interface=fsl.UnaryMaths(operation='bin'),
                          name='create_mask')
    workflow.connect(ir_thresolder, 'out_file', create_mask, 'in_file')

    model_fitting = pe.Node(niftyfit.FitAsl(source_file=in_asl_file,
                                            pcasl=True, PLD=1800, LDD=1800, eff=0.614, mul=0.1),
                            name='model_fitting')
    workflow.connect(fitqt1, 'm0map', model_fitting, 'm0map')
    workflow.connect(create_mask, 'out_file', model_fitting, 'mask')

    t1_to_asl_registration = pe.Node(niftyreg.RegAladin(rig_only_flag=True), name='t1_to_asl_registration')
    m0_resampling = pe.Node(niftyreg.RegResample(inter_val='LIN'), name='m0_resampling')
    mc_resampling = pe.Node(niftyreg.RegResample(inter_val='LIN'), name='mc_resampling')
    t1_resampling = pe.Node(niftyreg.RegResample(inter_val='LIN'), name='t1_resampling')
    cbf_resampling = pe.Node(niftyreg.RegResample(inter_val='LIN'), name='cbf_resampling')

    if in_t1_file:
        t1_to_asl_registration.inputs.flo_file = in_asl_file
        t1_to_asl_registration.inputs.ref_file = in_t1_file
        m0_resampling.inputs.ref_file = in_t1_file
        mc_resampling.inputs.ref_file = in_t1_file
        t1_resampling.inputs.ref_file = in_t1_file
        cbf_resampling.inputs.ref_file = in_t1_file
        workflow.connect(fitqt1, 'm0map', m0_resampling, 'flo_file')
        workflow.connect(fitqt1, 'mcmap', mc_resampling, 'flo_file')
        workflow.connect(fitqt1, 't1map', t1_resampling, 'flo_file')
        workflow.connect(model_fitting, 'cbf_file', cbf_resampling, 'flo_file')
        workflow.connect(t1_to_asl_registration, 'aff_file', m0_resampling, 'trans_file')
        workflow.connect(t1_to_asl_registration, 'aff_file', mc_resampling, 'trans_file')
        workflow.connect(t1_to_asl_registration, 'aff_file', t1_resampling, 'trans_file')
        workflow.connect(t1_to_asl_registration, 'aff_file', cbf_resampling, 'trans_file')

    maskrenamer = pe.Node(interface=niu.Rename(format_string=subject_id + '_mask', keep_ext=True), name='maskrenamer')
    m0renamer = pe.Node(interface=niu.Rename(format_string=subject_id + '_m0map', keep_ext=True), name='m0renamer')
    mcrenamer = pe.Node(interface=niu.Rename(format_string=subject_id + '_mcmap', keep_ext=True), name='mcrenamer')
    t1renamer = pe.Node(interface=niu.Rename(format_string=subject_id + '_t1map', keep_ext=True), name='t1renamer')
    workflow.connect(create_mask, 'out_file', maskrenamer, 'in_file')
    if in_t1_file:
        workflow.connect(m0_resampling, 'out_file', m0renamer, 'in_file')
        workflow.connect(mc_resampling, 'out_file', mcrenamer, 'in_file')
        workflow.connect(t1_resampling, 'out_file', t1renamer, 'in_file')
    else:
        workflow.connect(fitqt1, 'm0map', m0renamer, 'in_file')
        workflow.connect(fitqt1, 'mcmap', mcrenamer, 'in_file')
        workflow.connect(fitqt1, 't1map', t1renamer, 'in_file')

    ds = pe.Node(nio.DataSink(parameterization=False, base_directory=output_dir), name='ds')
    workflow.connect(maskrenamer, 'out_file', ds, '@mask_file')
    workflow.connect(m0renamer, 'out_file', ds, '@m0_file')
    workflow.connect(mcrenamer, 'out_file', ds, '@mc_file')
    workflow.connect(t1renamer, 'out_file', ds, '@t1_file')
    if in_t1_file:
        workflow.connect(cbf_resampling, 'out_file', ds, '@cbf_file')
    else:
        workflow.connect(model_fitting, 'cbf_file', ds, '@cbf_file')
    workflow.connect(model_fitting, 'error_file', ds, '@err_file')

    return workflow
