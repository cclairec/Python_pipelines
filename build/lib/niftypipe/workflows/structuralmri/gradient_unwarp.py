# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
import nipype.interfaces.niftyreg as niftyreg
from nipype.utils.filemanip import split_filename
from ...interfaces.niftk.distortion import GradwarpCorrection


def create_gradient_unwarp_workflow(in_file,
                                    in_coeff,
                                    output_dir,
                                    offsets=[0, 0, 0],
                                    scanner='siemens',
                                    radius=0.225,
                                    interp='CUB',
                                    throughplaneonly=False,
                                    inplaneonly=False):

    _, subject_id, _ = split_filename(os.path.basename(in_file))
    # Create a workflow to process the images
    workflow = pe.Workflow(name='gradwarp_correction')
    workflow.base_output_dir = 'gradwarp_correction'
    # The gradwarp field is computed.
    gradwarp = pe.Node(interface=GradwarpCorrection(),
                       name='gradwarp')
    gradwarp.inputs.offset_x = -1 * offsets[0]
    gradwarp.inputs.offset_y = -1 * offsets[1]
    gradwarp.inputs.offset_z = -1 * offsets[2]
    gradwarp.inputs.radius = radius
    gradwarp.inputs.scanner_type = scanner
    gradwarp.inputs.in_file = in_file
    gradwarp.inputs.coeff_file = in_coeff
    if throughplaneonly:
        gradwarp.inputs.throughplaneonly = True
    if inplaneonly:
        gradwarp.inputs.inplaneonly = True

    # The obtained deformation field is used the resample the input image
    resampling = pe.Node(interface=niftyreg.RegResample(inter_val=interp,
                                                        ref_file=in_file,
                                                        flo_file=in_file),
                         name='resampling')
    workflow.connect(gradwarp, 'out_file', resampling, 'trans_file')
    renamer = pe.Node(interface=niu.Rename(format_string=subject_id + "_gradwarp", keep_ext=True),
                      name='renamer')
    workflow.connect(resampling, 'out_file', renamer, 'in_file')
    # Create a data sink
    ds = pe.Node(nio.DataSink(parameterization=False), name='ds')
    ds.inputs.base_directory = output_dir
    workflow.connect(renamer, 'out_file', ds, '@img')
    
    return workflow
