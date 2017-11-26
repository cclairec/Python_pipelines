# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
from nipype.utils.filemanip import split_filename
import nipype.interfaces.niftyreg as niftyreg
from ...interfaces.niftk import dtitk
from ...interfaces.niftk.utils import FilenamesToTextFile

jhu_atlas_fa = os.path.join(os.environ['FSLDIR'], 'data', 'atlases', 'JHU', 'JHU-ICBM-FA-1mm.nii.gz')
jhu_atlas_labels = os.path.join(os.environ['FSLDIR'], 'data', 'atlases', 'JHU', 'JHU-ICBM-labels-1mm.nii.gz')


def extract_statistics_extended_function(in_file, roi_file):

    from niftypipe.interfaces.niftk.labels import jhulabels as jhu
    import nibabel as nib
    import numpy as np
    import os

    labels_dict = jhu.get_label_dictionary()
    parcellation = np.array(nib.load(roi_file).get_data()).ravel()
    data = np.array(nib.load(in_file).get_data()).ravel()

    header = '"White Matter (sum, l=-1)"'
    for key in sorted(labels_dict, key=lambda label: int(label)):
        header += ', "' + labels_dict[key] + ' (l=' + key + ')"'
    header += '\n'
    whitematter = np.mean(data[parcellation >= 1])
    line = str(whitematter)
    for key in sorted(labels_dict, key=lambda label: int(label)):
        v = np.mean(data[parcellation == int(key)])
        line += ', ' + str(v)
    out_csv_file = os.path.abspath('statistics.csv')
    f = open(out_csv_file, 'w+')
    f.write(header + line)
    f.close()
    return out_csv_file


# Rigid groupwise workflow
def create_dtitk_rigid_groupwise_workflow(name="DTITK_rigid_groupwise_workflow",
                                          sm_option_value="EDS",
                                          ftol_value=0.005,
                                          sep_value=2,
                                          use_trans=0):

    workflow = pe.Workflow(name=name)

    # We need to create an input node for the workflow
    input_node = pe.Node(niu.IdentityInterface(
        fields=['in_template',
                'in_files',
                'in_trans']),
        name='input_node')

    # Rigid registration nodes
    if use_trans > 0:
        rig_reg = pe.MapNode(interface=dtitk.RTVCGM(),
                             name="rig_reg",
                             iterfield=['flo_file',
                                        'in_trans'])
        workflow.connect(input_node, 'in_trans', rig_reg, 'in_trans')
    else:
        rig_reg = pe.MapNode(interface=dtitk.RTVCGM(),
                             name="rig_reg_init",
                             iterfield=['flo_file'])
    workflow.connect(input_node, 'in_template', rig_reg, 'ref_file')
    workflow.connect(input_node, 'in_files', rig_reg, 'flo_file')
    rig_reg.inputs.sm_option_val = sm_option_value
    rig_reg.inputs.ftol_val = ftol_value
    rig_reg.inputs.sep_x_val = sep_value
    rig_reg.inputs.sep_y_val = sep_value
    rig_reg.inputs.sep_z_val = sep_value
    rig_reg.synchronize = True

    # Resampling image nodes
    sym_tensor = pe.MapNode(interface=dtitk.AffineSymTensor3DVolume(),
                            name="sym_tensor",
                            iterfield=['flo_file',
                                       'in_trans'])
    workflow.connect(input_node, 'in_template', sym_tensor, 'ref_file')
    workflow.connect(input_node, 'in_files', sym_tensor, 'flo_file')
    workflow.connect(rig_reg, 'out_trans', sym_tensor, 'in_trans')
    sym_tensor.synchronize = True

    # Create a file list containing all the warped images
    img_list = pe.Node(interface=FilenamesToTextFile(),
                       name="img_list")
    workflow.connect(sym_tensor, 'out_file', img_list, 'in_files')

    # Node creating the average image
    avg_image = pe.Node(interface=dtitk.TVMean(),
                        name="avg_image")
    workflow.connect(img_list, 'out_file', avg_image, 'in_file_list')

    # Rename the average file
    rename_avg_rig = pe.Node(interface=niu.Rename(keep_ext=True,
                                                  format_string='avg_img_rig'),
                             name='rename_avg_rig')
    workflow.connect(avg_image, 'out_file', rename_avg_rig, 'in_file')

    output_node = pe.Node(niu.IdentityInterface(
        fields=['out_template',
                'out_trans',
                'out_res']),
        name='output_node')
    workflow.connect(rename_avg_rig, 'out_file', output_node, 'out_template')
    workflow.connect(rig_reg, 'out_trans', output_node, 'out_trans')
    workflow.connect(sym_tensor, 'out_file', output_node, 'out_res')
    return workflow


# Affine groupwise workflow
def create_dtitk_affine_groupwise_workflow(name="DTITK_affine_groupwise_workflow",
                                           sm_option_value="EDS",
                                           ftol_value=0.001,
                                           sep_value=2,
                                           use_trans=0):

    workflow = pe.Workflow(name=name)

    # We need to create an input node for the workflow
    input_node = pe.Node(niu.IdentityInterface(
        fields=['in_template',
                'in_files',
                'in_trans']),
        name='input_node')

    # Affine registration nodes
    if use_trans > 0:
        aff_reg = pe.MapNode(interface=dtitk.ATVCGM(),
                             name="aff_reg",
                             iterfield=['flo_file',
                                        'in_trans'])
        workflow.connect(input_node, 'in_trans', aff_reg, 'in_trans')
    else:
        aff_reg = pe.MapNode(interface=dtitk.ATVCGM(),
                             name="aff_reg_init",
                             iterfield=['flo_file'])
    workflow.connect(input_node, 'in_template', aff_reg, 'ref_file')
    workflow.connect(input_node, 'in_files', aff_reg, 'flo_file')
    aff_reg.inputs.sm_option_val = sm_option_value
    aff_reg.inputs.ftol_val = ftol_value
    aff_reg.inputs.sep_x_val = sep_value
    aff_reg.inputs.sep_y_val = sep_value
    aff_reg.inputs.sep_z_val = sep_value
    aff_reg.synchronize = True

    # Combine all the affine transformation into a single text file
    combine_aff = pe.Node(interface=FilenamesToTextFile(),
                          name='combine_aff')
    workflow.connect(aff_reg, 'out_trans', combine_aff, 'in_files')

    # Compute the inverse of the average affine transformation
    inv_avg_aff = pe.Node(interface=dtitk.Affine3DShapeAverage(),
                          name='inv_avg_aff')
    inv_avg_aff.inputs.inverse_flag = True
    workflow.connect(combine_aff, 'out_file', inv_avg_aff, 'file_list')
    workflow.connect(input_node, 'in_template', inv_avg_aff, 'ref_file')

    # Compose all the affine transformations
    comp = pe.MapNode(interface=dtitk.Affine3Dtool(),
                      name='comp',
                      iterfield=['in_file'])
    workflow.connect(aff_reg, 'out_trans', comp, 'in_file')
    workflow.connect(inv_avg_aff, 'out_file', comp, 'comp_file')

    # Resample all images
    sym_tensor = pe.MapNode(interface=dtitk.AffineSymTensor3DVolume(),
                            name="sym_tensor",
                            iterfield=['flo_file',
                                       'in_trans'])
    workflow.connect(input_node, 'in_template', sym_tensor, 'ref_file')
    workflow.connect(input_node, 'in_files', sym_tensor, 'flo_file')
    workflow.connect(comp, 'out_file', sym_tensor, 'in_trans')
    sym_tensor.synchronize = True

    # Create a file list containing all the warped images
    img_list = pe.Node(interface=FilenamesToTextFile(),
                       name="img_list")
    workflow.connect(sym_tensor, 'out_file', img_list, 'in_files')

    # Node creating the average image
    avg_image = pe.Node(interface=dtitk.TVMean(),
                        name="avg_image")
    workflow.connect(img_list, 'out_file', avg_image, 'in_file_list')

    # Rename the average file
    rename_avg_aff = pe.Node(interface=niu.Rename(keep_ext=True,
                                                  format_string='avg_img_aff'),
                             name='rename_avg_aff')
    workflow.connect(avg_image, 'out_file', rename_avg_aff, 'in_file')

    output_node = pe.Node(niu.IdentityInterface(
        fields=['out_template',
                'out_trans',
                'out_res']),
        name='output_node')
    workflow.connect(rename_avg_aff, 'out_file', output_node, 'out_template')
    workflow.connect(comp, 'out_file', output_node, 'out_trans')
    workflow.connect(sym_tensor, 'out_file', output_node, 'out_res')
    return workflow


# Non-rigid groupwise workflow
def create_dtitk_nonrigid_groupwise_workflow(name="DTITK_nonrigid_groupwise_workflow",
                                             ftol_value=0.002,
                                             use_trans=0):

    workflow = pe.Workflow(name=name)

    # We need to create an input node for the workflow
    input_node = pe.Node(niu.IdentityInterface(
        fields=['in_template',
                'in_mask',
                'in_files']),
        name='input_node')

    # Non-rigid registration nodes
    nrr_reg = pe.MapNode(interface=dtitk.DtiDiffeomorphicReg(),
                         name="nrr_reg",
                         iterfield=['flo_file'])
    workflow.connect(input_node, 'in_template', nrr_reg, 'ref_file')
    workflow.connect(input_node, 'in_files', nrr_reg, 'flo_file')
    workflow.connect(input_node, 'in_mask', nrr_reg, 'mask_file')
    nrr_reg.inputs.ftol_val = ftol_value
    nrr_reg.inputs.iteration_val = use_trans + 1
    nrr_reg.inputs.initial_val = 1

    # Resample all images
    sym_tensor = pe.MapNode(interface=dtitk.DeformationSymTensor3DVolume(),
                            name="sym_tensor",
                            iterfield=['flo_file',
                                       'in_trans'])
    workflow.connect(nrr_reg, 'out_trans', sym_tensor, 'in_trans')
    workflow.connect(input_node, 'in_files', sym_tensor, 'flo_file')

    # Compute an initial warped template
    war_list = pe.Node(interface=FilenamesToTextFile(),
                       name="war_list")
    workflow.connect(sym_tensor, 'out_file', war_list, 'in_files')
    avg_image = pe.Node(interface=dtitk.TVMean(),
                        name="avg_image")
    workflow.connect(war_list, 'out_file', avg_image, 'in_file_list')

    # Compute the average deformation field
    def_list = pe.Node(interface=FilenamesToTextFile(),
                       name="def_list")
    workflow.connect(nrr_reg, 'out_trans', def_list, 'in_files')

    # Compute the average image
    avg_def = pe.Node(interface=dtitk.VVMean(),
                      name='avg_def')
    workflow.connect(def_list, 'out_file', avg_def, 'in_file_list')

    # Invert the resulting average deformation field
    inv_def = pe.Node(interface=dtitk.DfToInverse(),
                      name='inv_def')
    workflow.connect(avg_def, 'out_file', inv_def, 'in_file')

    # Warp the template image
    war_temp = pe.Node(interface=dtitk.DeformationSymTensor3DVolume(),
                       name='war_temp')
    workflow.connect(inv_def, 'out_file', war_temp, 'in_trans')
    workflow.connect(avg_image, 'out_file', war_temp, 'flo_file')

    # Rename the average file
    rename_avg_nrr = pe.Node(interface=niu.Rename(keep_ext=True,
                                                  format_string='avg_img_nrr'),
                             name='rename_avg_nrr')
    workflow.connect(war_temp, 'out_file', rename_avg_nrr, 'in_file')

    output_node = pe.Node(niu.IdentityInterface(
        fields=['out_template',
                'out_trans',
                'out_res']),
        name='output_node')
    workflow.connect(rename_avg_nrr, 'out_file', output_node, 'out_template')
    workflow.connect(nrr_reg, 'out_trans', output_node, 'out_trans')
    workflow.connect(sym_tensor, 'out_file', output_node, 'out_res')
    return workflow


# Full Groupwise workflow
def create_dtitk_groupwise_workflow(in_files,
                                    name="dtitk_groupwise_workflow",
                                    rig_iteration=3,
                                    aff_iteration=3,
                                    nrr_iteration=6):

    if rig_iteration + aff_iteration + nrr_iteration < 1:
        raise ValueError('Number of iteration in the dtitk groupwise is equal to zero')

    if rig_iteration + aff_iteration < 1:
        raise ValueError('Number of global (rigid and affine) iteration in the dtitk groupwise is equal to zero')

    workflow = pe.Workflow(name=name)
    # Scale the input image intensities
    scale_img = pe.MapNode(interface=dtitk.TVtool(),
                           name='scale_img',
                           iterfield=['in_file'])
    scale_img.inputs.scale_value = 1000
    scale_img.inputs.in_file = in_files
    spd_img = pe.MapNode(interface=dtitk.TVtool(),
                         name='spd_img',
                         iterfield=['in_file'])
    spd_img.inputs.operation = 'spd'
    workflow.connect(scale_img, 'out_file', spd_img, 'in_file')
    # Adjust the header origin
    adjust_origin = pe.MapNode(interface=dtitk.TVAdjustVoxelspace(),
                               name='adjust_origin',
                               iterfield=['in_file'])
    adjust_origin.inputs.orig_val_x = 0
    adjust_origin.inputs.orig_val_y = 0
    adjust_origin.inputs.orig_val_z = 0
    workflow.connect(spd_img, 'out_file', adjust_origin, 'in_file')
    # Resample the image to the given resolution
    resampling = pe.MapNode(interface=dtitk.TVResample(),
                            name='resampling',
                            iterfield=['in_file'])
    resampling.inputs.size_val_x = 128
    resampling.inputs.size_val_y = 128
    resampling.inputs.size_val_z = 64
    resampling.inputs.vsize_val_x = 1.5
    resampling.inputs.vsize_val_y = 1.75
    resampling.inputs.vsize_val_z = 2.25
    workflow.connect(adjust_origin, 'out_file', resampling, 'in_file')
    # Normalise the input image
    norm_img = pe.MapNode(interface=dtitk.TVtool(),
                          name='norm_img',
                          iterfield=['in_file'])
    norm_img.inputs.operation = 'norm'
    workflow.connect(resampling, 'out_file', norm_img, 'in_file')
    # Mask the input image
    mask_img = pe.MapNode(interface=dtitk.DTITKBinaryThresholdImageFilter(),
                          name='mask_img',
                          iterfield=['in_file'])
    mask_img.inputs.lower_val = 0.01
    mask_img.inputs.upper_val = 10
    mask_img.inputs.inside = 1
    mask_img.inputs.outside = 0
    workflow.connect(norm_img, 'out_file', mask_img, 'in_file')
    # Apply the mask
    apply_mask = pe.MapNode(interface=dtitk.TVtool(),
                            name='apply_mask',
                            iterfield=['in_file',
                                       'mask_file'])
    workflow.connect(resampling, 'out_file', apply_mask, 'in_file')
    workflow.connect(mask_img, 'out_file', apply_mask, 'mask_file')
    # Rename the preprocessed tensor files
    string_to_replace = '_tvtool_tvtool_TVAdjustVoxelspace_TVResample_tvtool'
    rename_tensor_image = pe.MapNode(interface=niu.Rename(format_string='%(basename)s_preproc',
                                                          parse_string='(?P<basename>\w*)' + string_to_replace + '.*',
                                                          keep_ext=True),
                                     name='rename_tensor_image',
                                     iterfield=['in_file'])
    workflow.connect(apply_mask, 'out_file', rename_tensor_image, 'in_file')

    # Create a text file containing all the image filename
    img_list = pe.Node(interface=FilenamesToTextFile(),
                       name='img_list')
    workflow.connect(rename_tensor_image, 'out_file', img_list, 'in_files')

    # Average all the pre-processed input images
    init_mean_image = pe.Node(interface=dtitk.TVMean(),
                              name='init_mean_image')
    workflow.connect(img_list, 'out_file', init_mean_image, 'in_file_list')

    # Rename the initial average file
    rename_avg0 = pe.Node(interface=niu.Rename(keep_ext=True,
                                               format_string='initial_avg'),
                          name='rename_avg0')
    workflow.connect(init_mean_image, 'out_file', rename_avg0, 'in_file')

    # Run the rigid registration step
    rig_workflows = []
    for i in range(rig_iteration):
        ftol = 0.01
        sep = 4
        if i is rig_iteration - 1:
            ftol = 0.005
            sep = 2
        w = create_dtitk_rigid_groupwise_workflow(name='rig_gw' + str(i),
                                                  sm_option_value="EDS",
                                                  ftol_value=ftol,
                                                  sep_value=sep,
                                                  use_trans=i)
        workflow.connect(rename_tensor_image, 'out_file', w, 'input_node.in_files')
        if i > 0:
            workflow.connect(rig_workflows[i-1], 'output_node.out_trans', w, 'input_node.in_trans')
            workflow.connect(rig_workflows[i-1], 'output_node.out_template', w, 'input_node.in_template')
        else:
            workflow.connect(rename_avg0, 'out_file', w, 'input_node.in_template')
        rig_workflows.append(w)

    # Run the affine registration step
    aff_workflows = []
    for i in range(aff_iteration):
        ftol = 0.01
        sep = 4
        if i is aff_iteration - 1:
            ftol = 0.001
            sep = 2
        w = create_dtitk_affine_groupwise_workflow(name='aff_gw' + str(i),
                                                   sm_option_value="EDS",
                                                   ftol_value=ftol,
                                                   sep_value=sep,
                                                   use_trans=1)
        workflow.connect(rename_tensor_image, 'out_file', w, 'input_node.in_files')
        if i > 0:
            workflow.connect(aff_workflows[i-1], 'output_node.out_trans', w, 'input_node.in_trans')
            workflow.connect(aff_workflows[i-1], 'output_node.out_template', w, 'input_node.in_template')
        else:
            if rig_iteration > 0:
                workflow.connect(rig_workflows[rig_iteration-1], 'output_node.out_trans', w, 'input_node.in_trans')
                workflow.connect(rig_workflows[rig_iteration-1], 'output_node.out_template',
                                 w, 'input_node.in_template')
            else:
                workflow.connect(rename_avg0, 'out_file', w, 'input_node.in_template')
        aff_workflows.append(w)

    # Run the non-rigid registration step
    nrr_workflows = []
    for i in range(nrr_iteration):
        w = create_dtitk_nonrigid_groupwise_workflow(name='nrr_gw' + str(i),
                                                     ftol_value=0.002,
                                                     use_trans=1)
        # generate a mask for the non-rigid step
        norm_temp = pe.Node(interface=dtitk.TVtool(),
                            name='norm_temp' + str(i))
        norm_temp.inputs.operation = 'norm'
        if i > 0:
            workflow.connect(nrr_workflows[i-1], 'output_node.out_template', norm_temp, 'in_file')
        elif aff_iteration > 0:
            workflow.connect(aff_workflows[aff_iteration-1], 'output_node.out_template', norm_temp, 'in_file')
        elif rig_iteration > 0:
            workflow.connect(rig_workflows[rig_iteration-1], 'output_node.out_template', norm_temp, 'in_file')
        else:
            workflow.connect(rename_avg0, 'out_file', norm_temp, 'in_file')
        bin_filter = pe.Node(interface=dtitk.DTITKBinaryThresholdImageFilter(),
                             name='bin_filter' + str(i))
        workflow.connect(norm_temp, 'out_file', bin_filter, 'in_file')
        bin_filter.inputs.lower_val = 0.01
        bin_filter.inputs.upper_val = 10
        bin_filter.inputs.inside = 1
        bin_filter.inputs.outside = 0
        # Connect the mask to the workflow
        workflow.connect(bin_filter, 'out_file', w, 'input_node.in_mask')
        # Use the result of the affine as an input ... erf
        if aff_iteration > 0:
            workflow.connect(aff_workflows[aff_iteration-1], 'output_node.out_res', w, 'input_node.in_files')
        elif rig_iteration > 0:
            workflow.connect(rig_workflows[rig_iteration-1], 'output_node.out_res', w, 'input_node.in_files')
        else:
            workflow.connect(rename_tensor_image, 'out_file', w, 'input_node.in_files')
        # Set the template image
        if i > 0:
            workflow.connect(nrr_workflows[i-1], 'output_node.out_template', w, 'input_node.in_template')
        else:
            if aff_iteration > 0:
                workflow.connect(aff_workflows[rig_iteration-1], 'output_node.out_template',
                                 w, 'input_node.in_template')
            elif rig_iteration > 0:
                workflow.connect(rig_workflows[rig_iteration-1], 'output_node.out_template',
                                 w, 'input_node.in_template')
            else:
                workflow.connect(init_mean_image, 'out_file', w, 'input_node.in_template')
        nrr_workflows.append(w)

    if nrr_iteration > 0:
        # Combine the global and local transformation to avoid double resampling
        comp_trans = pe.MapNode(interface=dtitk.DfRightComposeAffine(),
                                name='comp_trans',
                                iterfield=['aff_file',
                                           'def_file'])
        if aff_iteration > 0:
            workflow.connect(aff_workflows[aff_iteration-1], 'output_node.out_trans', comp_trans, 'aff_file')
        else:
            workflow.connect(rig_workflows[rig_iteration-1], 'output_node.out_trans', comp_trans, 'aff_file')
        workflow.connect(nrr_workflows[nrr_iteration-1], 'output_node.out_trans', comp_trans, 'def_file')
        # Warp the input images using the final transformations
        war_final = pe.MapNode(interface=dtitk.DeformationSymTensor3DVolume(),
                               name="war_final",
                               iterfield=['flo_file',
                                          'in_trans'])
        workflow.connect(rename_tensor_image, 'out_file', war_final, 'flo_file')
        workflow.connect(comp_trans, 'out_file', war_final, 'in_trans')
        war_final.synchronize = True

        # Compute the final template
        war_list = pe.Node(interface=FilenamesToTextFile(),
                           name="war_list")
        workflow.connect(war_final, 'out_file', war_list, 'in_files')
        final_temp = pe.Node(interface=dtitk.TVMean(),
                             name="final_temp")
        workflow.connect(war_list, 'out_file', final_temp, 'in_file_list')

    # Output node for the workflow
    output_node = pe.Node(niu.IdentityInterface(
        fields=['out_scaled',
                'out_template',
                'out_trans',
                'out_res']),
        name='output_node')

    workflow.connect(rename_tensor_image, 'out_file', output_node, 'out_scaled')
    if nrr_iteration > 0:
        workflow.connect(final_temp, 'out_file', output_node, 'out_template')
        workflow.connect(comp_trans, 'out_file', output_node, 'out_trans')
        workflow.connect(war_final, 'out_file', output_node, 'out_res')
    else:
        if aff_iteration > 0:
            last_workflow = aff_workflows[aff_iteration-1]
        elif rig_iteration > 0:
            last_workflow = rig_workflows[rig_iteration-1]
        workflow.connect(last_workflow, 'output_node.out_template', output_node, 'out_template')
        workflow.connect(last_workflow, 'output_node.out_trans', output_node, 'out_trans')
        workflow.connect(last_workflow, 'output_node.out_res', output_node, 'out_res')

    # Return the generate workflow
    return workflow


def create_tensor_groupwise_and_feature_extraction_workflow(input_tensor_fields, output_dir,
                                                            rig_iteration=3, aff_iteration=3, nrr_iteration=6,
                                                            biomarkers=['fa', 'tr', 'ad', 'rd']):

    subject_ids = [split_filename(os.path.basename(f))[1] for f in input_tensor_fields]

    pipeline_name = 'dti_wm_regional_analysis'
    workflow = create_dtitk_groupwise_workflow(in_files=input_tensor_fields,
                                               name=pipeline_name,
                                               rig_iteration=rig_iteration,
                                               aff_iteration=aff_iteration,
                                               nrr_iteration=nrr_iteration)
    workflow.base_output_dir = pipeline_name
    workflow.base_dir = output_dir

    groupwise_fa = pe.Node(interface=dtitk.TVtool(operation='fa'), name='groupwise_fa')
    workflow.connect(workflow.get_node('output_node'), 'out_template', groupwise_fa, 'in_file')

    aff_jhu_to_groupwise = pe.Node(interface=niftyreg.RegAladin(flo_file=jhu_atlas_fa), name='aff_jhu_to_groupwise')
    workflow.connect(groupwise_fa, 'out_file', aff_jhu_to_groupwise, 'ref_file')

    nrr_jhu_to_groupwise = pe.Node(interface=niftyreg.RegF3D(vel_flag=True, lncc_val=-5, maxit_val=150, be_val=0.025,
                                                             flo_file=jhu_atlas_fa), name='nrr_jhu_to_groupwise')
    workflow.connect(groupwise_fa, 'out_file', nrr_jhu_to_groupwise, 'ref_file')
    workflow.connect(aff_jhu_to_groupwise, 'aff_file', nrr_jhu_to_groupwise, 'aff_file')

    resample_labels = pe.Node(interface=niftyreg.RegResample(inter_val='NN', flo_file=jhu_atlas_labels),
                              name='resample_labels')
    workflow.connect(groupwise_fa, 'out_file', resample_labels, 'ref_file')
    workflow.connect(nrr_jhu_to_groupwise, 'cpp_file', resample_labels, 'trans_file')

    iterator = pe.Node(interface=niu.IdentityInterface(fields=['biomarker']), name='iterator')
    iterator.iterables = ('biomarker', biomarkers)

    tvtool = pe.MapNode(interface=dtitk.TVtool(), name='tvtool', iterfield=['in_file'])
    workflow.connect(workflow.get_node('output_node'), 'out_res', tvtool, 'in_file')
    workflow.connect(iterator, 'biomarker', tvtool, 'operation')

    stats_extractor = pe.MapNode(interface=niu.Function(input_names=['in_file', 'roi_file'],
                                                        output_names=['out_file'],
                                                        function=extract_statistics_extended_function),
                                 name='stats_extractor', iterfield=['in_file'])
    workflow.connect(resample_labels, 'out_file', stats_extractor, 'roi_file')
    workflow.connect(tvtool, 'out_file', stats_extractor, 'in_file')

    tensors_renamer = pe.MapNode(interface=niu.Rename(format_string='%(subject_id)s_tensors', keep_ext=True),
                                 name='tensors_renamer', iterfield=['in_file', 'subject_id'])
    workflow.connect(workflow.get_node('output_node'), 'out_res', tensors_renamer, 'in_file')
    tensors_renamer.inputs.subject_id = subject_ids

    maps_renamer = pe.MapNode(interface=niu.Rename(format_string='%(subject_id)s_%(biomarker)s', keep_ext=True),
                              name='maps_renamer', iterfield=['in_file', 'subject_id'])
    workflow.connect(tvtool, 'out_file', maps_renamer, 'in_file')
    workflow.connect(iterator, 'biomarker', maps_renamer, 'biomarker')
    maps_renamer.inputs.subject_id = subject_ids

    stats_renamer = pe.MapNode(interface=niu.Rename(format_string='%(subject_id)s_%(biomarker)s.csv'),
                               name='stats_renamer', iterfield=['in_file', 'subject_id'])
    workflow.connect(stats_extractor, 'out_file', stats_renamer, 'in_file')
    workflow.connect(iterator, 'biomarker', stats_renamer, 'biomarker')
    stats_renamer.inputs.subject_id = subject_ids

    groupwise_outputs = ['fa', 'labels', 'tensors']
    gw_outputs_merger = pe.Node(interface=niu.Merge(numinputs=len(groupwise_outputs)), name='gw_outputs_merger')
    workflow.connect(groupwise_fa, 'out_file', gw_outputs_merger, 'in1')
    workflow.connect(resample_labels, 'out_file', gw_outputs_merger, 'in2')
    workflow.connect(workflow.get_node('output_node'), 'out_template', gw_outputs_merger, 'in3')

    groupwise_renamer = pe.MapNode(interface=niu.Rename(format_string='groupwise_%(type)s', keep_ext=True),
                                   name='groupwise_renamer', iterfield=['in_file', 'type'])
    workflow.connect(gw_outputs_merger, 'out', groupwise_renamer, 'in_file')
    groupwise_renamer.inputs.type = groupwise_outputs

    # Create a data sink
    ds = pe.Node(nio.DataSink(parameterization=False),
                 name='data_sink')
    ds.inputs.base_directory = os.path.abspath(output_dir)

    workflow.connect(maps_renamer, 'out_file', ds, 'biomarkers.@maps')
    workflow.connect(stats_renamer, 'out_file', ds, 'biomarkers.@stats')
    workflow.connect(tensors_renamer, 'out_file', ds, 'tensors')
    workflow.connect(groupwise_renamer, 'out_file', ds, '@outputs')

    return workflow
