# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
import nipype.interfaces.niftyseg as niftyseg
import nipype.interfaces.niftyreg as niftyreg
import nipype.interfaces.fsl as fsl
from nipype.interfaces.fsl.maths import MaxImage
from ...interfaces.niftk.filters import CropImage


def extract_db_info_function(in_db_file):
    import xml.etree.ElementTree as Xml
    import os
    from glob import glob
    database_xml = Xml.parse(in_db_file).getroot()
    database_root = os.path.dirname(in_db_file)
    temp_template_images = []
    temp_template_labels = []
    for data in database_xml.findall('data'):
        current_path = data.find('path').text
        temp_template_images = glob(database_root + os.sep + current_path + os.sep + '*.nii*')
        temp_template_images = temp_template_images + glob(database_root + os.sep + current_path + os.sep + '*.hdr')
    for data in database_xml.findall('info'):
        if data.find('type').text == 'Label':
            current_path = data.find('path').text
            temp_template_labels = glob(database_root + os.sep + current_path + os.sep + '*.nii*')
            temp_template_labels = temp_template_labels + glob(database_root + os.sep + current_path + os.sep + '*.hdr')
    input_template_images = []
    input_template_labels = []
    for template_num in range(len(temp_template_images)):
        image = temp_template_images[template_num]
        template_basename = os.path.basename(image)
        for label_num in range(len(temp_template_labels)):
            label = temp_template_labels[label_num]
            label_basename = os.path.basename(label)
            if label_basename == template_basename:
                input_template_images.append(image)
                input_template_labels.append(label)
                break
    return input_template_images, input_template_labels


def create_steps_propagation_pipeline(name='steps_propagation',
                                      aligned_templates=False):

    workflow = pe.Workflow(name=name)

    # Create an input node
    input_node = pe.Node(
        interface=niu.IdentityInterface(
            fields=['in_file',
                    'database_file']),
        name='input_node')

    extract_db_info = pe.Node(interface=niu.Function(input_names=['in_db_file'], output_names=['input_template_images',
                                                                                               'input_template_labels'],
                                                     function=extract_db_info_function),
                              name='extract_db_info')
    workflow.connect(input_node, 'database_file', extract_db_info, 'in_db_file')

    # All the template images are affinely registered to the target image
    current_aladin = pe.MapNode(interface=niftyreg.RegAladin(verbosity_off_flag=True),
                                name='aladin',
                                iterfield=['flo_file'])
    workflow.connect(input_node, 'in_file', current_aladin, 'ref_file')
    workflow.connect(extract_db_info, 'input_template_images', current_aladin, 'flo_file')

    # Compute the affine TLS if required
    current_robust_affine = None
    if aligned_templates is True:
        current_robust_affine = pe.Node(interface=niftyreg.RegAverage(), name='robust_affine')
        workflow.connect(current_aladin, 'aff_file', current_robust_affine, 'avg_lts_files')
        current_aff_prop = pe.MapNode(interface=niftyreg.RegResample(verbosity_off_flag=True, inter_val='NN'),
                                      name='resample_aff',
                                      iterfield=['flo_file'])
        workflow.connect(current_robust_affine, 'out_file', current_aff_prop, 'trans_file')
    else:
        current_aff_prop = pe.MapNode(interface=niftyreg.RegResample(verbosity_off_flag=True, inter_val='NN'),
                                      name='resample_aff',
                                      iterfield=['flo_file',
                                                 'trans_file'])
        workflow.connect(current_aladin, 'aff_file', current_aff_prop, 'trans_file')
    workflow.connect(input_node, 'in_file', current_aff_prop, 'ref_file')
    workflow.connect(extract_db_info, 'input_template_labels', current_aff_prop, 'flo_file')

    # Merge all the affine parcellation into one 4D
    current_aff_prop_merge = pe.Node(interface=fsl.Merge(dimension='t'), name='merge_aff_prop')
    workflow.connect(current_aff_prop, 'out_file', current_aff_prop_merge, 'in_files')

    # Combine all the propagated parcellation into a single image
    current_aff_prop_max = pe.Node(interface=MaxImage(dimension='T'), name='max_aff')
    workflow.connect(current_aff_prop_merge, 'merged_file', current_aff_prop_max, 'in_file')

    # Binarise the obtained mask
    current_aff_prop_bin = pe.Node(interface=niftyseg.UnaryMaths(operation='bin'), name='bin_aff')
    workflow.connect(current_aff_prop_max, 'out_file', current_aff_prop_bin, 'in_file')

    # Dilate the obtained mask
    current_aff_prop_dil = pe.Node(interface=niftyseg.BinaryMathsInteger(operation='dil', operand_value=10),
                                   name='dil_aff')
    workflow.connect(current_aff_prop_bin, 'out_file', current_aff_prop_dil, 'in_file')

    # Fill the obtained mask
    current_aff_prop_fill = pe.Node(interface=niftyseg.UnaryMaths(operation='fill'), name='fill_aff')
    workflow.connect(current_aff_prop_dil, 'out_file', current_aff_prop_fill, 'in_file')

    # Crop the target image to speed up the process
    current_crop_target = pe.Node(interface=CropImage(), name='crop_target')
    workflow.connect(input_node, 'in_file', current_crop_target, 'in_file')
    workflow.connect(current_aff_prop_fill, 'out_file', current_crop_target, 'mask_file')

    # Crop the mask image to speed up the process
    current_crop_mask = pe.Node(interface=CropImage(), name='crop_mask')
    workflow.connect(current_aff_prop_fill, 'out_file', current_crop_mask, 'in_file')
    workflow.connect(current_aff_prop_fill, 'out_file', current_crop_mask, 'mask_file')

    # Perform all the non-linear registration
    if aligned_templates is True:
        current_f3d = pe.MapNode(interface=niftyreg.RegF3D(sx_val=-2.5, be_val=0.01, verbosity_off_flag=True),
                                 name='f3d',
                                 iterfield=['flo_file'])
        workflow.connect(current_robust_affine, 'out_file', current_f3d, 'aff_file')
    else:
        current_f3d = pe.MapNode(interface=niftyreg.RegF3D(),
                                 name='f3d',
                                 iterfield=['flo_file',
                                            'aff_file'])
        workflow.connect(current_aladin, 'aff_file', current_f3d, 'aff_file')
    workflow.connect(current_crop_target, 'out_file', current_f3d, 'ref_file')
    workflow.connect(current_crop_mask, 'out_file', current_f3d, 'rmask_file')
    workflow.connect(extract_db_info, 'input_template_images', current_f3d, 'flo_file')

    # Merge all the non-linear warped images into one 4D
    current_f3d_temp_merge = pe.Node(interface=fsl.Merge(dimension='t'), name='merge_f3d_temp')
    workflow.connect(current_f3d, 'res_file', current_f3d_temp_merge, 'in_files')

    # Propagate the obtained mask
    current_f3d_prop = pe.MapNode(interface=niftyreg.RegResample(inter_val='NN', verbosity_off_flag=True),
                                  name='f3d_prop',
                                  iterfield=['flo_file',
                                             'trans_file'])
    workflow.connect(current_crop_target, 'out_file', current_f3d_prop, 'ref_file')
    workflow.connect(extract_db_info, 'input_template_labels', current_f3d_prop, 'flo_file')
    workflow.connect(current_f3d, 'cpp_file', current_f3d_prop, 'trans_file')

    # Merge all the non-linear warped labels into one 4D
    current_f3d_prop_merge = pe.Node(interface=fsl.Merge(dimension='t'), name='merge_f3d_prop')
    workflow.connect(current_f3d_prop, 'out_file', current_f3d_prop_merge, 'in_files')

    # Extract the consensus parcellation using steps
    current_fusion = pe.Node(interface=niftyseg.STEPS(template_num=15, kernel_size=1.5, mrf_value=0.15),
                             name='fusion')
    workflow.connect(current_crop_target, 'out_file', current_fusion, 'in_file')
    workflow.connect(current_f3d_temp_merge, 'merged_file', current_fusion, 'warped_img_file')
    workflow.connect(current_f3d_prop_merge, 'merged_file', current_fusion, 'warped_seg_file')
    workflow.connect(current_aff_prop_fill, 'out_file', current_fusion, 'mask_file')

    # Resample the obtained consensus label into the original image space
    current_prop_orig_res = pe.MapNode(interface=niftyreg.RegResample(inter_val='NN', verbosity_off_flag=True),
                                       name='prop_orig_res',
                                       iterfield=['flo_file'])
    workflow.connect(input_node, 'in_file', current_prop_orig_res, 'ref_file')
    workflow.connect(current_fusion, 'out_file', current_prop_orig_res, 'flo_file')

    # Connect the output to the output node
    output_node = pe.Node(
        interface=niu.IdentityInterface(
            fields=['parcellated_file']),
        name='output_node')
    workflow.connect(current_prop_orig_res, 'out_file', output_node, 'parcellated_file')

    return workflow
