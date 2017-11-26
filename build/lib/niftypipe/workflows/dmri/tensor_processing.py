# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import nibabel as nib
import nipype.interfaces.niftyseg as niftyseg
import nipype.interfaces.niftyreg as niftyreg
import nipype.interfaces.niftyfit as niftyfit
import nipype.interfaces.fsl as fsl
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
from nipype.workflows.smri.niftyreg.groupwise import create_groupwise_average
from ..misc.utils import (create_multiple_resample_and_combine_mask)
from ...interfaces.niftk.utils import (SplitB0DWIsFromFile, SplitAndSelect, ProduceMask, ExtractBaseName, NoRegAladin, IdentityMatrix)
from ...interfaces.niftk.qc import (InterSliceCorrelationPlot, create_MatrixRotationPlot_workflow,
                                    TransRotationPlot, PrepPlotReadParams)
from .susceptibility_correction import (create_fieldmap_susceptibility_workflow,
                                        create_registration_susceptibility_workflow)
from ...interfaces.niftk.dtitk import FSLEddyAcq


# Needs to go somewhere more sensible eventually
# regrettably workflows.misc.utils.get_data_dims is 3D
def nvols_from_img(img):
    import nibabel as nib
    #imglist = imglist if not isinstance(imglist, basestring) else (imglist,)

    def get_n_vols(iname):
        img = nib.load(iname)
        hdr = img.get_header()
        dims = hdr.get_data_shape()
        return 1 if len(dims) < 4 else dims[3]
    return get_n_vols(img)


def merge_dwi_function(in_dwis, in_bvals, in_bvecs):
    import sys
    import errno
    import nipype.interfaces.fsl as fsl
    import nibabel as nib
    import numpy as np

    qdiff_eps = 1.0

    def merge_vector_files(input_files):
        import numpy as np
        import os

        result = np.array([])
        files_base, files_ext = os.path.splitext(os.path.basename(input_files[0]))
        for f in input_files:
            if result.size == 0:
                result = np.loadtxt(f, ndmin=2)
            else:
                result = np.hstack((result, np.loadtxt(f, ndmin=2)))
        output_file = os.path.abspath(files_base + '_merged' + files_ext)
        np.savetxt(output_file, result, fmt='%.3f')
        return output_file

    if len(in_bvals) == 0 or len(in_bvecs) == 0 or len(in_dwis) == 0:
        print 'I/O One of the dwis merge input is empty, exiting.'
        sys.exit(errno.EIO)

    if not type(in_dwis) == list:
        in_dwis = [in_dwis]
        in_bvals = [in_bvals]
        in_bvecs = [in_bvecs]

    qforms = []
    for dwi in in_dwis:
        im = nib.load(dwi)
        qform = im.get_qform()
        qforms.append(qform)

    if len(in_dwis) > 1:
        # Walk backwards through the dwis, assume later scans are more likely to be correctly acquired!
        for file_index in range(len(in_dwis) - 2, -1, -1):
            # If the qform doesn't match that of the last scan, throw it away
            if np.linalg.norm(qforms[len(in_dwis) - 1] - qforms[file_index]) > qdiff_eps:
                print '** WARNING ** : The qform of the DWIs dont match, denoting a re-scan error, throwing ', \
                    in_dwis[file_index]
                in_dwis.pop(file_index)
                in_bvals.pop(file_index)
                in_bvecs.pop(file_index)

    # Set the default values of these variables as the first index,
    # in case we only have one image and we don't do a merge
    out_dwis = in_dwis[0]
    out_bvals = in_bvals[0]
    out_bvecs = in_bvecs[0]
    if len(in_dwis) > 1:
        merger = fsl.Merge(dimension='t')
        merger.inputs.in_files = in_dwis
        res = merger.run()
        out_dwis = res.outputs.merged_file
        out_bvals = merge_vector_files(in_bvals)
        out_bvecs = merge_vector_files(in_bvecs)

    return out_dwis, out_bvals, out_bvecs


def remove_dmri_volumes(in_dwi,
                        in_bval,
                        in_bvec,
                        volume_to_remove):
    import os
    from nipype.utils.filemanip import split_filename
    from dipy.io.gradients import read_bvals_bvecs
    import nibabel as nib
    import numpy as np
    # Set the output filenames
    _, dwi_name, dwi_ext = split_filename(in_dwi)
    _, bvec_name, bvec_ext = split_filename(in_bvec)
    _, bval_name, bval_ext = split_filename(in_bval)
    out_dwi = os.getcwd() + os.sep + dwi_name + '_cleaned' + dwi_ext
    out_bvec = os.getcwd() + os.sep + bvec_name + '_cleaned' + bvec_ext
    out_bval = os.getcwd() + os.sep + bval_name + '_cleaned' + bval_ext
    # Read the input dwi file
    dwi_nii = nib.load(in_dwi)
    dwi_data = dwi_nii.get_data()
    dwi_num = dwi_data.shape[3]
    # Load the bvec and bval files
    bval_data, bvec_data = read_bvals_bvecs(in_bval, in_bvec)
    # Check if the specified volumes actually exist
    for i in volume_to_remove:
        if i < 0 or i >= dwi_num:
            raise Exception('The specified volume (' + str(i) +
                            ' ) is not in the data range (0:' + str(dwi_num-1) + ')')
    # Create the new dwi volume
    new_dwi_num = dwi_num - len(volume_to_remove)
    new_dwi_data = np.zeros((dwi_data.shape[0], dwi_data.shape[1], dwi_data.shape[2], new_dwi_num),
                            dtype=dwi_nii.get_data_dtype())
    new_bval_data = np.zeros(new_dwi_num)
    new_bvec_data = np.zeros((new_dwi_num, 3))
    # Fill the new dwi volume
    j = 0
    for i in range(dwi_num):
        if i not in volume_to_remove:
            new_dwi_data[:, :, :, j] = dwi_data[:, :, :, i]
            new_bval_data[j] = bval_data[i]
            new_bvec_data[j, :] = bvec_data[i, :]
            j += 1
    # Save the new dwi volume
    new_dwi_nii = nib.Nifti1Image(new_dwi_data, dwi_nii.get_affine())
    nib.save(new_dwi_nii, out_dwi)
    # Save the new bvec and bval files
    np.savetxt(out_bval, new_bval_data, '%.1f',)
    np.savetxt(out_bvec, new_bvec_data, '%.6f', '\t')

    return out_dwi, out_bval, out_bvec


def reorient_bvec(in_bvec_file,
                  affine_matrices):
    import os
    from nipype.utils.filemanip import split_filename
    from nipype.interfaces.niftyreg import RegTransform
    from dipy.io.gradients import read_bvals_bvecs
    import numpy as np
    # Generate the output filename
    _, bvec_name, bvec_ext = split_filename(in_bvec_file)
    out_bvec_file = os.getcwd() + os.sep + bvec_name + '_rot' + bvec_ext
    # Read the input vectors
    _, bvec_data = read_bvals_bvecs(None, in_bvec_file)
    volume_number = bvec_data.shape[0]
    # Check that the arguments agree
    if volume_number != len(affine_matrices):
            raise Exception('The number of bvec and affine transformations are different.' +
                            ' (' + str(volume_number) + ' vs' + len(affine_matrices) + ' respectively)')
    # Create the new bvec array
    new_bvec_data = np.zeros((3, volume_number))
    # Reorient each input vector
    for i in range(volume_number):
        # Compute the closest rotation matrix
        aff_2_rig = RegTransform()
        aff_2_rig.inputs.aff_2_rig_input = affine_matrices[i]
        aff_2_rig_res = aff_2_rig.run()
        # Read the rigid matrix
        current_rigid = np.loadtxt(aff_2_rig_res.outputs.out_file)[0:3, 0:3]
        new_bvec_data[:, i] = np.dot(np.transpose(current_rigid), np.transpose(bvec_data[i, :]))
    # Save the re-oriented vectors
    np.savetxt(out_bvec_file, new_bvec_data, '%.6f', '\t')
    # Return the new file
    return out_bvec_file


def create_merge_tensor_B0_workflow(name = "merge_tensor_B0"):
    """
    Create workflow to merge a tensor volume and a B0 volume to produce source volume
     suitable for niftyfit.dwi_tool

     Inputs::

        input_node.in_b0:                     B0 volume
        input_node.in_tensors:                Tensor volume

    Outputs::

        output_node.merged_file:            Combined tensor and log B0 file
    :return:
      returns a workflow
    """
    workflow = pe.Workflow(name=name)
    input_node = pe.Node(interface=niu.IdentityInterface(fields=['in_b0',
                                                                 'in_tensors']),
                         name='input_node')

    log_b0 = pe.Node(interface=fsl.UnaryMaths(operation='log'), name='log_b0')
    workflow.connect(input_node, 'in_b0', log_b0, 'in_file')

    combine = pe.Node(niu.Merge(2), name="combine")
    workflow.connect(input_node, 'in_tensors', combine, 'in1')
    workflow.connect(log_b0, 'out_file', combine, 'in2')

    fslmerge = pe.Node(interface=fsl.utils.Merge(dimension="t"), name="output_node")
    workflow.connect(combine, 'out', fslmerge, 'in_files')

    return workflow




def create_diffusion_mri_processing_workflow(t1_mask_provided=False,
                                             name='diffusion_mri_processing_workflow',
                                             in_bvallowthreshold=10,
                                             susceptibility_correction_with_fm=False,
                                             susceptibility_correction_without_fm=False,
                                             in_susceptibility_params=[34.56, 2.46, '-y'],
                                             resample_in_t1=False, log_data=False, dwi_interp_type='CUB',
                                             wls_tensor_fit=True, rigid_only=False,
                                             with_eddy=False):
    """
    Creates a diffusion processing workflow. This initially performs a groupwise registration
    of all the B=0 images, subsequently each of the DWI is registered to the averageB0.
    If enabled, the averageB0 is corrected for magnetic susceptibility distortion.
    Tensor, and derivative images, are estimated using niftyfit

    Parameters
    ----------

    ::

    Inputs::

        ::param name:                               The workflow name
        ::param in_bvallowthreshold:                The low bvalue threshold
        ::param susceptibility_correction_with_fm:  Plug the susceptibility correction workflow using the field maps
        ::param in_susceptibility_params:           The rot/etc/ped parameters for the susceptibility
        ::param resample_in_t1:                     Resample the outputs in the T1 space
        ::param log_data:                           Log the dwis for registration purposes
        ::param dwi_interp_type:                    Interpolation type for the DWIs (default is cubic 'CUB')
        ::param wls_tensor_fit:                     Fit the tensors using weighted least square
        ::param rigid_only:                         Use rigid only for DWI registration

    Outputs::

        tensors:            Fitted tensor image, stored in a lower-triangular manner in a 6-D nifti file:
                            xx-xy-yy-xz-yz-zz
        fa:                 Fractional Anisotropy map (between 0 and 1)
        md:                 Mean Diffusivity map (in s.mm-3)
        rgb:                RGB colour coded map indicating main eigenvector direction
        v1:                 Image coding the main eigenvector in a 3-D vector image
        predicted_images:   Model-predicted images
        tensor_residuals:   Linear Least Square fitting residual
        dwis:               Diffusion Weighted Image motion (and susceptibility) corrected
        bval:               Diffusion Weighted bval motion corrected
        bvec:               Diffusion Weighted bvec motion corrected
        b0:                 Average B-Null image
        t1_to_b0:           T1 to B0 affine transformation (with T1 as reference)
        mask:               Mask in the DWI space used to mask out background in output images
        interslice_cc:      Quality Control graph indicating interslice correlation
        matrix_rot:         Quality Control graph indication rotation and translation of each DWI


    Example
    -------

    >>> preproc = create_diffusion_mri_processing_workflow() # doctest: +SKIP
    >>> preproc.inputs.inputnode.in_dwi_4d_file = 'diffusion.nii' # doctest: +SKIP
    >>> preproc.inputs.inputnode.in_bval_file = 'diffusion.bval' # doctest: +SKIP
    >>> preproc.inputs.inputnode.in_bvec_file = 'diffusion.bvec' # doctest: +SKIP
    >>> preproc.inputs.inputnode.in_t1_file = 't1.nii' # doctest: +SKIP
    >>> preproc.inputs.inputnode.in_t1_mask_file = 't1_mask.nii' # doctest: +SKIP
    >>> preproc.inputs.inputnode.in_fm_magnitude_file = 'magnitude.nii' # doctest: +SKIP
    >>> preproc.inputs.inputnode.in_fm_phase_file = 'phase.nii' # doctest: +SKIP
    The outputs are listed above. Use a `DataSink` to sink them somewhere:
    >>> ds = pe.Node(nio.DataSink(parameterization=False, base_directory='/tmp/'), name='ds') # doctest: +SKIP
    >>> preproc.connect(preproc.get_node('renamer'), 'out_file', ds, '@outputs') # doctest: +SKIP
    >>> preproc.connect(preproc.get_node('reorder_transformations'), 'out', ds, 'transformations') # doctest: +SKIP
    >>> preproc.run() # doctest: +SKIP


    """

    def merge_list_from_indices(l1, l2, l1_indices):
        ret_val = []
        id_l1, id_l2 = 0, 0
        for t in range(len(l1) + len(l2)):
            if t in l1_indices:
                ret_val.append(l1[id_l1])
                id_l1 += 1
            else:
                ret_val.append(l2[id_l2])
                id_l2 += 1
        return ret_val
    """
    Creating the nipype workflow
    """
    workflow = pe.Workflow(name=name)

    """
    Input node containing all needed inputs for DTI preprocessing
    """
    input_node = pe.Node(interface=niu.IdentityInterface(fields=['in_dwi_4d_file',
                                                                 'in_bval_file',
                                                                 'in_bvec_file',
                                                                 'in_t1_file',
                                                                 'in_t1_mask_file',
                                                                 'in_fm_magnitude_file',
                                                                 'in_fm_phase_file'],
                                                         mandatory_inputs=False),
                         name='input_node')
    """
    The `subject_extractor` node extracts the basename of the DWI file in order to provide
    readable names for the output files
    """
    subject_extractor = pe.Node(interface=ExtractBaseName(), name='subject_extractor')
    workflow.connect(input_node, 'in_dwi_4d_file', subject_extractor, 'in_file')


    connect_in = pe.Node(niu.IdentityInterface(fields=['bvec', 'dwi']), name="connector_in")
    connect_newbvec = pe.Node(niu.IdentityInterface(fields=['bvec']), name="connector_bvec")

    if (with_eddy):
        """Count DWI"""
        in_count = pe.Node(interface=niu.Function(input_names=['img'],
                                                  output_names=['nvol'],
                                                  function = nvols_from_img),
                           name='in_count')
        workflow.connect(input_node, 'in_dwi_4d_file', in_count, 'img')

        """Need B0 mask"""
        eddy_split_dwis = pe.Node(interface=SplitB0DWIsFromFile(),
                                  name='eddy_split_dwis')
        workflow.connect(input_node, 'in_dwi_4d_file', eddy_split_dwis, 'in_file')
        workflow.connect(input_node, 'in_bval_file', eddy_split_dwis, 'in_bval')
        workflow.connect(input_node, 'in_bvec_file', eddy_split_dwis, 'in_bvec')
        eddy_b0_groupwise = create_groupwise_average(itr_rigid=2,
                                                     itr_affine=0,
                                                     itr_non_lin=0,
                                                     name='eddy_groupwise')
        workflow.connect(eddy_split_dwis, 'out_B0s', eddy_b0_groupwise, 'input_node.in_files')
        eddy_ave_ims_b0 = pe.Node(interface=niftyreg.RegAverage(), name="eddy_ave_ims_b0")
        workflow.connect(eddy_split_dwis, 'out_B0s', eddy_ave_ims_b0, 'avg_files')
        workflow.connect(eddy_ave_ims_b0, 'out_file', eddy_b0_groupwise, 'input_node.ref_file')
        eddy_connect_mask = pe.Node(niu.IdentityInterface(fields=['mask_file']),
                                    name="eddy_connect_mask")
        if t1_mask_provided:
            # Use affine reg of T1 and nonlinear reg of *mask* to produce
            # large estimated brain mask
            eddy_t1_to_b0 = pe.Node(interface=niftyreg.RegAladin(rig_only_flag=True,
                                                                 verbosity_off_flag=True),
                                    name='eddy_t1_to_b0')
            workflow.connect(input_node, 'in_t1_file', eddy_t1_to_b0, "flo_file")
            workflow.connect(eddy_b0_groupwise, 'output_node.average_image',
                             eddy_t1_to_b0, "ref_file")

            eddy_t1m_to_b0_nl = pe.Node(interface=niftyreg.RegF3D(), name="eddy_t1m_to_b0_nl")
            workflow.connect(input_node, 'in_t1_mask_file', eddy_t1m_to_b0_nl, "flo_file")
            workflow.connect(eddy_b0_groupwise, 'output_node.average_image',
                             eddy_t1m_to_b0_nl, "ref_file")
            workflow.connect(eddy_t1_to_b0, 'aff_file', eddy_t1m_to_b0_nl, "aff_file")

            eddy_merge_trans = pe.Node(niu.Merge(2), name="eddymergetrans")

            workflow.connect(eddy_t1_to_b0, "aff_file", eddy_merge_trans, 'in1')
            workflow.connect(eddy_t1m_to_b0_nl, 'cpp_file', eddy_merge_trans, 'in2')

            eddy_res_and_comb = create_multiple_resample_and_combine_mask(
                name='eddyresandcomb')

            workflow.connect(eddy_merge_trans, 'out',
                             eddy_res_and_comb, 'input_node.input_transforms')
            workflow.connect(input_node, 'in_t1_mask_file',
                             eddy_res_and_comb, 'input_node.input_mask')

            workflow.connect(eddy_b0_groupwise, 'output_node.average_image',
                             eddy_res_and_comb, 'input_node.target_image')
            workflow.connect(eddy_res_and_comb, 'output_node.out_file',
                             eddy_connect_mask, 'mask_file')
        else:
            # Not well tested, may benefit from the nonlinear mask approach too
            eddy_b0_mask = pe.Node(interface=ProduceMask(), name='eddy_b0_mask')
            workflow.connect(eddy_b0_groupwise, 'output_node.average_image',
                             eddy_b0_mask, 'in_file')
            workflow.connect(eddy_b0_mask, 'out_file',
                             eddy_connect_mask, 'mask_file')

        """Eddy"""
        eddyacq = pe.Node(interface=FSLEddyAcq(), name="eddyacq")
        eddyacq.inputs.rot = in_susceptibility_params[0]
        eddyacq.inputs.ped = in_susceptibility_params[2]
        workflow.connect(in_count, 'nvol', eddyacq, 'dticount')

        eddy_node = pe.Node(interface=fsl.Eddy(), name="eddy")
        eddy_node.inputs.num_threads = 8   # fix later

        workflow.connect(input_node, 'in_dwi_4d_file', eddy_node, 'in_file')
        workflow.connect(input_node, 'in_bvec_file', eddy_node, 'in_bvec')
        workflow.connect(input_node, 'in_bval_file', eddy_node, 'in_bval')
        workflow.connect(eddy_connect_mask, 'mask_file', eddy_node, 'in_mask')
        workflow.connect(eddyacq, 'out_acqp', eddy_node, 'in_acqp')
        workflow.connect(eddyacq, 'out_index', eddy_node, 'in_index')
        workflow.connect(eddy_node, 'out_corrected', connect_in, 'dwi')

        bvec_rename = pe.Node(interface=niu.Rename(format_string="reoriented.bvec"), name="bvec_rename")
        workflow.connect(eddy_node, 'out_bvec', bvec_rename, 'in_file')
        workflow.connect(bvec_rename, 'out_file', connect_in, 'bvec')
    else:
        workflow.connect([(input_node, connect_in,
                           [('in_dwi_4d_file', 'dwi'),
                            ('in_bvec_file', 'bvec')])])

    """
    Node using fslsplit() to split the 4D file, separate the B0 and DWIs
    """
    split_dwis = pe.Node(interface=SplitB0DWIsFromFile(),
                         name='split_dwis')
    workflow.connect(connect_in, 'dwi', split_dwis, 'in_file')
    workflow.connect(input_node, 'in_bval_file', split_dwis, 'in_bval')
    workflow.connect(connect_in, 'bvec', split_dwis, 'in_bvec')
    """
    Perform rigid groupwise registration for the B-Null images if not
    using eddy. If using eddy then just pretend we did.
    """
    ave_ims_b0 = pe.Node(interface=niftyreg.RegAverage(), name="ave_ims_b0")
    workflow.connect(split_dwis, 'out_B0s', ave_ims_b0, 'avg_files')
    b0_groupwise = pe.Node(niu.IdentityInterface(['output_node.average_image',
                                                  'output_node.trans_files']),
                           name = 'groupwise_connector')
    if (with_eddy) :
        b0_id_transforms = pe.MapNode(IdentityMatrix(),
                                   name="b0_id_transforms",
                                   iterfield=['flo_file'])
        workflow.connect(split_dwis, 'out_B0s', b0_id_transforms, 'flo_file')
        workflow.connect(b0_id_transforms, 'aff_file',
                         b0_groupwise, 'output_node.trans_files')
        workflow.connect(ave_ims_b0, 'out_file',
                         b0_groupwise, 'output_node.average_image')
    else:
        b0_do_groupwise = create_groupwise_average(itr_rigid=2,
                                            itr_affine=0,
                                            itr_non_lin=0,
                                            name='groupwise')
        workflow.connect(split_dwis, 'out_B0s', b0_do_groupwise, 'input_node.in_files')
        workflow.connect(ave_ims_b0, 'out_file', b0_do_groupwise, 'input_node.ref_file')
        workflow.connect([(b0_do_groupwise, b0_groupwise,
                          [('output_node.average_image', 'output_node.average_image'),
                           ('output_node.trans_files', 'output_node.trans_files')])])
    """
    As we're trying to estimate an affine transformation, and rotations and shears are confounded
    easier just to optimise an affine directly for the DWI
    """
    if (with_eddy) :
        dwi_to_b0_registration = pe.MapNode(NoRegAladin(),
                                            name='dwi_to_b0_noregistration',
                                            iterfield=['flo_file'])
    else :
        dwi_to_b0_registration = pe.MapNode(niftyreg.RegAladin(verbosity_off_flag=True),
                                        name='dwi_to_b0_registration',
                                        iterfield=['flo_file'])
    dwi_to_b0_registration.inputs.ln_val = 2

    if rigid_only is True:
        dwi_to_b0_registration.inputs.rig_only_flag = True
    else:
        dwi_to_b0_registration.inputs.aff_direct_flag = True
    """
    If required, the log of the DWIs is used to find best transformations between DWIs and the B-Null image
    """
    if log_data:
        log_b0 = pe.Node(interface=fsl.UnaryMaths(operation='log'), name='log_b0')
        workflow.connect(b0_groupwise, 'output_node.average_image', log_b0, 'in_file')
        log_ims = pe.MapNode(interface=fsl.UnaryMaths(operation='log', output_datatype='float'),
                             name='log_ims', iterfield=['in_file'])
        workflow.connect(split_dwis, 'out_DWIs', log_ims, 'in_file')
        smooth_b0 = pe.Node(interface=niftyseg.BinaryMaths(operation='smo', operand_value=0.75),
                            name='smooth_b0')
        workflow.connect(log_b0, 'out_file', smooth_b0, 'in_file')
        smooth_ims = pe.MapNode(interface=niftyseg.BinaryMaths(operation='smo', operand_value=0.75),
                                name='smooth_ims', iterfield=['in_file'])
        workflow.connect(log_ims, 'out_file', smooth_ims, 'in_file')
        workflow.connect(smooth_b0, 'out_file',
                         dwi_to_b0_registration, 'ref_file')
        workflow.connect(smooth_ims, 'out_file',
                         dwi_to_b0_registration, 'flo_file')
    else:
        workflow.connect(b0_groupwise, 'output_node.average_image',
                         dwi_to_b0_registration, 'ref_file')
        workflow.connect(split_dwis, 'out_DWIs',
                         dwi_to_b0_registration, 'flo_file')
    """
    The transformations from `dwi_to_b0_registration` need to be re-implanted in the right order of the original DWI
    4D image
    """
    reorder_transformations = pe.Node(interface=niu.Function(input_names=['l1', 'l2', 'l1_indices'],
                                                             output_names=['out'],
                                                             function=merge_list_from_indices),
                                      name='reorder_transformations')
    workflow.connect(b0_groupwise, 'output_node.trans_files', reorder_transformations, 'l1')
    workflow.connect(dwi_to_b0_registration, 'aff_file', reorder_transformations, 'l2')
    workflow.connect(split_dwis, 'out_indices', reorder_transformations, 'l1_indices')
    """
    Reorient the b-vectors
    """
    if (not with_eddy):
        reorient_vector = pe.Node(interface=niu.Function(input_names=['in_bvec_file',
                                                                      'affine_matrices'],
                                                         output_names=['out'],
                                                         function=reorient_bvec),
                                  name='reorient_vector')
        workflow.connect(input_node, 'in_bvec_file', reorient_vector, 'in_bvec_file')
        workflow.connect(reorder_transformations, 'out', reorient_vector, 'affine_matrices')
        workflow.connect(reorient_vector, 'out', connect_newbvec, 'bvec')
    else :
        workflow.connect(connect_in, 'bvec', connect_newbvec, 'bvec')
    """
    Upsample the average B0 to increase the registration resolution
    """
    upsample_b0 = pe.Node(interface=niftyreg.RegTools(),
                          name='upsample_b0')
    workflow.connect(b0_groupwise, 'output_node.average_image', upsample_b0, 'in_file')
    upsample_b0.inputs.chg_res_val = (1.5, 1.5, 1.5)
    """
    Perform a T1 to B0 registration
    """
    t1_to_b0 = pe.Node(interface=niftyreg.RegAladin(rig_only_flag=True,
                                                    verbosity_off_flag=True),
                       name='t1_to_b0')
    workflow.connect(upsample_b0, 'out_file', t1_to_b0, 'ref_file')
    workflow.connect(input_node, 'in_t1_file', t1_to_b0, 'flo_file')
    """
    Nonlinear mask registration for mask resampling
    """
    t1m_to_b0_nl = pe.Node(interface=niftyreg.RegF3D(), name="t1m_to_b0_nl")
    workflow.connect(upsample_b0, 'out_file', t1m_to_b0_nl, "ref_file")
    workflow.connect(t1_to_b0, 'aff_file', t1m_to_b0_nl, "aff_file")
    if t1_mask_provided:
        workflow.connect(input_node, 'in_t1_mask_file', t1m_to_b0_nl, "flo_file")
    else:
        t1_mask = pe.Node(interface=ProduceMask(), name='t1_mask')
        workflow.connect(input_node, 'in_t1_file', t1_mask, 'in_file')
        workflow.connect(t1_mask, 'out_file',
                         t1m_to_b0_nl, "flo_file")
    """
    Resample the T1 mask into the upsampled DWI space
    """

    merge_mask_trans = pe.Node(niu.Merge(2), name="mergemasktrans")
    workflow.connect(t1_to_b0, "aff_file", merge_mask_trans, 'in1')
    workflow.connect(t1m_to_b0_nl, 'cpp_file', merge_mask_trans, 'in2')

    t1_mask_resampling = create_multiple_resample_and_combine_mask(
        name='maskresandcomb')

    workflow.connect(merge_mask_trans, 'out',
                     t1_mask_resampling, 'input_node.input_transforms')
    workflow.connect(upsample_b0, 'out_file',
                     t1_mask_resampling, 'input_node.target_image')

    if t1_mask_provided:
        workflow.connect(input_node, 'in_t1_mask_file',
            t1_mask_resampling, 'input_node.input_mask')
    else:
        workflow.connect(t1_mask, 'out_file',
            t1_mask_resampling, 'input_node.input_mask')
    """
    Plug the Susceptibility correction workflow if required
    """
    susceptibility_correction = False
    if susceptibility_correction_with_fm or susceptibility_correction_without_fm:
        if susceptibility_correction_with_fm:
            select_fm_mag = pe.Node(interface=SplitAndSelect(), name='select_fm_mag')
            workflow.connect(input_node, 'in_fm_magnitude_file', select_fm_mag, 'in_file')
            susceptibility_correction = create_fieldmap_susceptibility_workflow('susceptibility_correction_with_fm',
                                                                                reg_to_t1=True)
            workflow.connect(select_fm_mag, 'out_file', susceptibility_correction, 'input_node.mag_image')
            workflow.connect(input_node, 'in_fm_phase_file', susceptibility_correction, 'input_node.phase_image')
            susceptibility_correction.inputs.input_node.rot = in_susceptibility_params[0]
            susceptibility_correction.inputs.input_node.etd = in_susceptibility_params[1]
            susceptibility_correction.inputs.input_node.ped = in_susceptibility_params[2]
        else:
            susceptibility_correction = create_registration_susceptibility_workflow('susceptibility_correction_without_fm')
        workflow.connect(b0_groupwise, 'output_node.average_image', susceptibility_correction, 'input_node.epi_image')
        workflow.connect(t1_to_b0, 'res_file', susceptibility_correction, 'input_node.t1')
        workflow.connect(t1_mask_resampling, 'output_node.out_file',
                         susceptibility_correction, 'input_node.t1_mask')
        compose_transformations = pe.MapNode(niftyreg.RegTransform(), name='compose_transformations',
                                             iterfield=['comp_input2'])
        workflow.connect(susceptibility_correction, 'output_node.out_field',
                         compose_transformations, 'comp_input')
        workflow.connect(reorder_transformations, 'out', compose_transformations, 'comp_input2')

    """
    Resample the DWI and B0s with the correct transformations
    """
    resample_dwis = pe.MapNode(niftyreg.RegResample(inter_val=dwi_interp_type, verbosity_off_flag=True),
                               name='resample_dwis',
                               iterfield=['trans_file', 'flo_file'])
    workflow.connect(b0_groupwise, 'output_node.average_image', resample_dwis, 'ref_file')
    workflow.connect(split_dwis, 'out_all', resample_dwis, 'flo_file')
    if susceptibility_correction:
        workflow.connect(compose_transformations, 'out_file', resample_dwis, 'trans_file')
    else:
        workflow.connect(reorder_transformations, 'out', resample_dwis, 'trans_file')

    """
    Remerge all the DWIs in preparation for the tensor fitting algorithm
    """
    merge_dwis = pe.Node(interface=fsl.Merge(dimension='t'), name='merge_dwis')
    workflow.connect(resample_dwis, 'out_file', merge_dwis, 'in_files')
    """
    Threshold the DWIs to 0 to avoid negative values that might have been introduced by cubic interpolation
    of the DWIs
    """
    threshold_dwis = pe.Node(interface=fsl.Threshold(thresh=0.0, direction='below'),
                             name='threshold_dwis')
    if susceptibility_correction:
        """
        Divide the corrected merged DWIs by the distortion Jacobian image to dampen compression effects
        """
        modulate_dwis = pe.Node(interface=fsl.BinaryMaths(operation='mul'), name='modulate_dwis')
        workflow.connect(merge_dwis, 'merged_file', modulate_dwis, 'in_file')
        workflow.connect(susceptibility_correction, 'output_node.out_jac', modulate_dwis, 'operand_file')
        workflow.connect(modulate_dwis, 'out_file', threshold_dwis, 'in_file')
    else:
        workflow.connect(merge_dwis, 'merged_file', threshold_dwis, 'in_file')
    """
    Produce the Quality Control Graph from the registration results
    """
    interslice_qc = pe.Node(interface=InterSliceCorrelationPlot(),
                            name='interslice_qc')
    workflow.connect(input_node, 'in_dwi_4d_file', interslice_qc, 'in_file')
    workflow.connect(input_node, 'in_bval_file', interslice_qc, 'bval_file')
    #matrixprep_qc = pe.Node(interface=MatrixPrepPlot(), name='matrixprep_qc')
    #workflow.connect(reorder_transformations, 'out', matrixprep_qc, 'in_files')
    #transrotplot_qc = pe.Node(interface=TransRotationPlot(), name='transrot_qc')
    #workflow.connect(matrixprep_qc, 'out_xfms', transrotplot_qc, 'in_xfms')

    transrotplot_connector = pe.Node(niu.IdentityInterface(fields=['out_file']), name="transrotplot_connector")
    if with_eddy:
        readparams_qc = pe.Node(interface=PrepPlotReadParams(), name="readparams_qc")
        workflow.connect(eddy_node, 'out_parameter', readparams_qc, 'in_file')
        transrotplot_qc = pe.Node(interface=TransRotationPlot(), name="transrotation_plot")
        workflow.connect(readparams_qc, 'out_xfms', transrotplot_qc, 'in_xfms')
        workflow.connect(transrotplot_qc,'out_file', transrotplot_connector, 'out_file')
    else:
        matrixplot_qc = create_MatrixRotationPlot_workflow()
        workflow.connect(reorder_transformations, 'out', matrixplot_qc, 'input_node.in_files')
        workflow.connect(matrixplot_qc,'output_node.out_file', transrotplot_connector, 'out_file')
    """
    Fit the tensor model
    The reorientation of the tensors is not necessary.
    It is taken care of in the tensor resampling process itself (reg_resample)
    We force the reorientation flag to 0 just to be sure
    """
    diffusion_model_fitting_tensor = pe.Node(interface=niftyfit.FitDwi(dti_flag=True,
                                                                       wls_flag=wls_tensor_fit,
                                                                       #  The bvallowthreshold of niftyfit is not
                                                                       #  yet released in the master.
                                                                       #  Deactivate the option for now
                                                                       #  bvallowthreshold=in_bvallowthreshold,
                                                                       rotsform_flag=0),
                                             name='diffusion_model_fitting_tensor')
    workflow.connect(threshold_dwis, 'out_file', diffusion_model_fitting_tensor, 'source_file')
    workflow.connect(input_node, 'in_bval_file', diffusion_model_fitting_tensor, 'bval_file')
    workflow.connect(connect_newbvec, 'bvec', diffusion_model_fitting_tensor, 'bvec_file')

    """
    Resample the T1 mask into the original DWI space
    """
    t1_mask_resampling2 = create_multiple_resample_and_combine_mask(
        name='maskresandcomb2')
    workflow.connect(merge_mask_trans, 'out',
                     t1_mask_resampling2, 'input_node.input_transforms')
    workflow.connect(b0_groupwise, 'output_node.average_image',
                     t1_mask_resampling2, 'input_node.target_image')

    if t1_mask_provided:
        #workflow.connect(input_node, 'in_t1_mask_file', t1_mask_resampling2, 'flo_file')
        workflow.connect(input_node, 'in_t1_mask_file',
            t1_mask_resampling2, 'input_node.input_mask')
    else:
        # workflow.connect(t1_mask, 'out_file', t1_mask_resampling2, 'flo_file')
        workflow.connect(t1_mask, 'out_file',
            t1_mask_resampling2, 'input_node.input_mask')

    workflow.connect(t1_mask_resampling2, 'output_node.out_file', diffusion_model_fitting_tensor, 'mask_file')

    inv_t1_aff = pe.Node(niftyreg.RegTransform(), name='inv_t1_aff')
    workflow.connect(t1_to_b0, 'aff_file', inv_t1_aff, 'inv_aff_input')
    """
    List the outputs of the DTI processing workflow
    """
    outputs = ['tensors', 'fa', 'md', 'rgb', 'v1', 'predicted_images', 'tensor_residuals',
               'dwis', 'b0', 't1_to_b0', 'mask', 'interslice_cc', 'matrix_rot', 'b0_res', 'fa_res',
               'bvec', 'bval']

    """
    Output node
    """
    output_node = pe.Node(interface=niu.IdentityInterface(fields=outputs,
                                                          mandatory_inputs=False),
                          name='output_node')
    """
    Fill the output node with connections according to arguments
    """
    workflow.connect(threshold_dwis, 'out_file', output_node, 'dwis')
    workflow.connect(diffusion_model_fitting_tensor, 'res_file', output_node, 'tensor_residuals')

    b0_resampling = pe.Node(niftyreg.RegResample(inter_val='LIN',
                                                 verbosity_off_flag=True),
                            name='b0_resampling')
    workflow.connect(input_node, 'in_t1_file', b0_resampling, 'ref_file')
    if susceptibility_correction:
        workflow.connect(susceptibility_correction, 'output_node.out_epi', b0_resampling, 'flo_file')
    else:
        workflow.connect(b0_groupwise, 'output_node.average_image', b0_resampling, 'flo_file')
    workflow.connect(inv_t1_aff, 'out_file', b0_resampling, 'trans_file')
    workflow.connect(b0_resampling, 'out_file', output_node, 'b0_res')

    tensor_resampling = pe.Node(niftyreg.RegResample(tensor_flag=True,
                                                     pad_val=0,  # pad_val is mute with tensors !!
                                                     verbosity_off_flag=True,
                                                     inter_val='LIN'),
                                name='tensor_resampling')
    workflow.connect(input_node, 'in_t1_file', tensor_resampling, 'ref_file')
    workflow.connect(diffusion_model_fitting_tensor, 'tenmap_file', tensor_resampling, 'flo_file')
    workflow.connect(inv_t1_aff, 'out_file', tensor_resampling, 'trans_file')
    nanremover = pe.Node(interface=fsl.UnaryMaths(operation='nan'), name='nanremover')
    workflow.connect(tensor_resampling, 'out_file', nanremover, 'in_file')
    dwi_tool = pe.Node(interface=niftyfit.DwiTool(dti_flag2=True), name='dwi_tool')
    workflow.connect(input_node, 'in_bval_file', dwi_tool, 'bval_file')

    merge_tensor_node = create_merge_tensor_B0_workflow()
    workflow.connect(nanremover, 'out_file', merge_tensor_node, 'input_node.in_tensors')
    workflow.connect(b0_resampling, 'out_file', merge_tensor_node, 'input_node.in_b0')
    workflow.connect(merge_tensor_node, 'output_node.merged_file', dwi_tool, 'source_file')

    if t1_mask_provided:
        workflow.connect(input_node, 'in_t1_mask_file', dwi_tool, 'mask_file')
    else:
        workflow.connect(t1_mask, 'out_file', dwi_tool, 'mask_file')

    workflow.connect(dwi_tool, 'famap_file', output_node, 'fa_res')

    if resample_in_t1:
        if t1_mask_provided:
            workflow.connect(input_node, 'in_t1_mask_file', output_node, 'mask')
        else:
            workflow.connect(t1_mask, 'out_file', output_node, 'mask')
        workflow.connect(dwi_tool, 'mcmap_file', output_node, 'mc')
        workflow.connect(dwi_tool, 'famap_file', output_node, 'fa')
        workflow.connect(dwi_tool, 'mdmap_file', output_node, 'md')
        workflow.connect(dwi_tool, 'v1map_file', output_node, 'v1')
        workflow.connect(dwi_tool, 'rgbmap_file', output_node, 'rgb')
        # Only connect the bvec and b0 inputs to dwi_tool if syn_file output is wanted.
        workflow.connect(connect_newbvec, 'bvec', dwi_tool, 'bvec_file')
        workflow.connect(b0_resampling, 'out_file', dwi_tool, 'b0_file')
        workflow.connect(dwi_tool, 'syn_file', output_node, 'predicted_images')
        workflow.connect(tensor_resampling, 'out_file', output_node, 'tensors')
        workflow.connect(b0_resampling, 'out_file', output_node, 'b0')

    else:
        workflow.connect(diffusion_model_fitting_tensor, 'syn_file', output_node, 'predicted_images')
        workflow.connect(diffusion_model_fitting_tensor, 'famap_file', output_node, 'fa')
        workflow.connect(diffusion_model_fitting_tensor, 'mdmap_file', output_node, 'md')
        workflow.connect(diffusion_model_fitting_tensor, 'v1map_file', output_node, 'v1')
        workflow.connect(diffusion_model_fitting_tensor, 'rgbmap_file', output_node, 'rgb')
        workflow.connect(diffusion_model_fitting_tensor, 'tenmap_file', output_node, 'tensors')

        if susceptibility_correction:
            workflow.connect(susceptibility_correction, 'output_node.out_epi', output_node, 'b0')
        else:
            workflow.connect(b0_groupwise, 'output_node.average_image', output_node, 'b0')
        workflow.connect(t1_mask_resampling2, 'output_node.out_file', output_node, 'mask')

    workflow.connect(inv_t1_aff, 'out_file', output_node, 't1_to_b0')
    workflow.connect(interslice_qc, 'out_file', output_node, 'interslice_cc')
    #workflow.connect(transrotplot_qc, 'out_file', output_node, 'matrix_rot')
    workflow.connect(transrotplot_connector, 'out_file', output_node, 'matrix_rot')

    workflow.connect(input_node, 'in_bval_file', output_node, 'bval')
    workflow.connect(connect_newbvec, 'bvec', output_node, 'bvec')

    """
    In order to have a harmonised naming convention, we use a merger to put all outputs in one list.

    We then use a `renamer` in order to give all the outputs the same basename prefix, using the `subject_extractor`
    node
    """
    output_merger = pe.Node(interface=niu.Merge(numinputs=len(outputs)), name='output_merger')
    for i in range(len(outputs)):
        workflow.connect(output_node, outputs[i], output_merger, 'in'+str(i+1))
    renamer = pe.MapNode(interface=niu.Rename(format_string="%(subject_id)s_%(type)s", keep_ext=True),
                         iterfield=['in_file', 'type'], name='renamer')
    workflow.connect(subject_extractor, 'out_basename', renamer, 'subject_id')
    renamer.inputs.type = outputs
    workflow.connect(output_merger, 'out', renamer, 'in_file')
    """
    Return the workflow
    """
    return workflow
