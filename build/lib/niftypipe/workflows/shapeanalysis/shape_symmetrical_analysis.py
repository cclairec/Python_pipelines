# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:


import nipype.interfaces.io as nio
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
import nipype.interfaces.niftyreg as niftyreg
import nipype.interfaces.niftyseg as niftyseg
from nipype.interfaces.utility import Function
from nipype.workflows.smri.niftyreg.groupwise import create_groupwise_average as create_atlas

from .shape_analysis import create_get_shape_distance_from_regression
from .atlas_computation import atlas_computation
from ...interfaces.niftk.utils import MergeLabels, extractSubList, SwapDimImage
from ...interfaces.niftk.io import Image2VtkMesh
from ...interfaces.shapeAnalysis import (VTKPolyDataReader, decimateVTKfile, reorder_lists2, reorder_lists,
                                         WriteXMLFiles, write_age2onset_file, split_list)


def create_symmetrical_images(name='create_symmetrical_images'):
    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    # Create the input node
    input_node = pe.Node(niu.IdentityInterface(
        fields=['input_images',
                'input_parcellations',
                'ages',
                'subject_ids'
                ]),
        name='input_node')

    # Create the output node
    output_node = pe.Node(niu.IdentityInterface(
        fields=['swapped_images',
                'swapped_parcellations',
                'LR_images',
                'LR_parcellations']),
        name='output_node')

    # create a list of LR swapped images:
    #cp_image = pe.MapNode(interface=CopyFile(), iterfield="in_file", name="cp_image")
    #workflow.connect(input_node, 'input_images', cp_image, 'in_file')

    swap_images = pe.MapNode(interface=SwapDimImage(), iterfield="image2reorient", name="swap_images")
    workflow.connect(input_node, 'input_images', swap_images, 'image2reorient')
    swap_images.inputs.axe2flip = "LR"

    # create a list of LR swapped parcellations:
    #cp_parcellations = pe.MapNode(interface=CopyFile(), iterfield="in_file",  name="cp_parcellations")
    #workflow.connect(input_node, 'input_parcellations', cp_parcellations, 'in_file')

    swap_parcellations = pe.MapNode(interface=SwapDimImage(), iterfield="image2reorient", name="swap_parcellations")
    workflow.connect(input_node, 'input_parcellations', swap_parcellations, 'image2reorient')
    swap_parcellations.inputs.axe2flip = "LR"

    # merge the lists
    merge_lists_images = pe.Node(interface=niu.Merge(axis='vstack', numinputs=2),
                                 name='merge_lists_images')
    workflow.connect(input_node, 'input_images', merge_lists_images, 'in1')
    workflow.connect(swap_images, 'flipped_image', merge_lists_images, 'in2')

    merge_lists_parcellations = pe.Node(interface=niu.Merge(axis='vstack', numinputs=2),
                                        name='merge_lists_parcellations')
    workflow.connect(input_node, 'input_parcellations', merge_lists_parcellations, 'in1')
    workflow.connect(swap_parcellations, 'flipped_image', merge_lists_parcellations, 'in2')

    # output node
    workflow.connect(swap_images, 'flipped_image', output_node, 'swapped_images')
    workflow.connect(swap_parcellations, 'flipped_image', output_node, 'swapped_parcellations')
    workflow.connect(merge_lists_images, 'out', output_node, 'LR_images')
    workflow.connect(merge_lists_parcellations, 'out', output_node, 'LR_parcellations')

    return workflow


def create_binary_to_meshes(label,
                            name='gw_binary_to_meshes',
                            reduction_rate=0.3):
    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    # Create the input node
    input_node = pe.Node(niu.IdentityInterface(
        fields=['input_images',
                'input_parcellations',
                'input_reference',
                'trans_files',
                'ref_file']),
        name='input_node')

    # Create the output node
    output_node = pe.Node(niu.IdentityInterface(
        fields=['output_meshes', 'output_meshes_sorted']),
        name='output_node')

    # Extract the relevant label from the GIF parcellation
    extract_label = pe.MapNode(interface=MergeLabels(),
                               iterfield=['in_file', 'roi_list'],
                               name='extract_label')
    extract_label.inputs.roi_list = label
    workflow.connect(input_node, 'input_parcellations', extract_label, 'in_file')

    # Removing parasite segmentation: Erosion.
    erode_binaries = pe.MapNode(interface=niftyseg.BinaryMathsInteger(operation='ero', operand_value=1),
                                iterfield=['in_file'], name='erode_binaries')
    workflow.connect(extract_label, 'out_file', erode_binaries, 'in_file')

    # Removing parasite segmentation: Dilatation.
    dilate_binaries = pe.MapNode(interface=niftyseg.BinaryMathsInteger(operation='dil', operand_value=1),
                                 iterfield=['in_file'], name='dilate_binaries')
    workflow.connect(erode_binaries, 'out_file', dilate_binaries, 'in_file')

    # Apply the relevant transformations to the roi
    apply_affine = pe.MapNode(interface=niftyreg.RegResample(inter_val='NN', verbosity_off_flag=True),
                              iterfield=['flo_file',
                                         'trans_file'],
                              name='apply_affine')
    workflow.connect(input_node, 'trans_files', apply_affine, 'trans_file')
    workflow.connect(input_node, 'ref_file', apply_affine, 'ref_file')
    workflow.connect(dilate_binaries, 'out_file', apply_affine, 'flo_file')

    # compute the large ROI that correspond to the union of all warped label
    extract_union_roi = pe.Node(interface=niftyreg.RegAverage(),
                                name='extract_union_roi')
    workflow.connect(apply_affine, 'out_file', extract_union_roi, 'avg_files')

    # Binarise the average ROI
    binarise_roi = pe.Node(interface=niftyseg.UnaryMaths(operation='bin'),
                           name='binarise_roi')
    workflow.connect(extract_union_roi, 'out_file', binarise_roi, 'in_file')

    # Dilation of the binarise union ROI
    dilate_roi = pe.Node(interface=niftyseg.BinaryMathsInteger(operation='dil', operand_value=7),
                         name='dilate_roi')
    workflow.connect(binarise_roi, 'out_file', dilate_roi, 'in_file')

    # Apply the transformations
    apply_rigid_refinement = pe.MapNode(interface=niftyreg.RegAladin(rig_only_flag=True, ln_val=1,
                                                                     nosym_flag=True, verbosity_off_flag=True),
                                        iterfield=['flo_file', 'in_aff_file'],
                                        name='apply_rigid_refinement')
    workflow.connect(input_node, 'input_images', apply_rigid_refinement, 'flo_file')
    workflow.connect(input_node, 'ref_file', apply_rigid_refinement, 'ref_file')
    workflow.connect(input_node, 'trans_files', apply_rigid_refinement, 'in_aff_file')
    workflow.connect(dilate_roi, 'out_file', apply_rigid_refinement, 'rmask_file')

    # Extract the mesh corresponding to the label
    final_resampling = pe.MapNode(interface=niftyreg.RegResample(inter_val='NN', verbosity_off_flag=True),
                                  iterfield=['flo_file',
                                             'trans_file'],
                                  name='final_resampling')
    workflow.connect(apply_rigid_refinement, 'aff_file', final_resampling, 'trans_file')
    workflow.connect(input_node, 'ref_file', final_resampling, 'ref_file')
    workflow.connect(dilate_binaries, 'out_file', final_resampling, 'flo_file')

    # Extract the mesh corresponding to the label
    extract_mesh = pe.MapNode(interface=Image2VtkMesh(in_reductionRate=reduction_rate),
                              iterfield=['in_file'],
                              name='extract_mesh')
    workflow.connect(final_resampling, 'out_file', extract_mesh, 'in_file')
    # workflow.connect(apply_rigid_refinement, 'aff_file', extract_mesh, 'matrix_file')

    # Create a rename for the average image
    groupwise_renamer = pe.Node(interface=niu.Rename(format_string='atlas', keep_ext=True),
                                name='groupwise_renamer')
    workflow.connect(input_node, 'ref_file', groupwise_renamer, 'in_file')

    workflow.connect(extract_mesh, 'out_file', output_node, 'output_meshes')

    return workflow


def create_lists_symmetrical_subject(name='create_lists_symmetrical_subject'):
    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    # Create the input node
    input_node = pe.Node(niu.IdentityInterface(
        fields=['input_meshes',
                'ages',
                'subject_ids']),
        name='input_node')
    # Create the output node
    output_node = pe.Node(niu.IdentityInterface(
        fields=['meshes_sorted_by_suj',
                'ages_sorted_by_suj',
                'subject_ids_sorted_by_suj']),
        name='output_node')


    # Split the list of meshes into a list of subject with 2 images per subject. rearrange in [[L R] [L R] [L R]]
    reorder_meshes = pe.Node(interface=reorder_lists2(),
                           name='reorder_meshes')
    workflow.connect(input_node, 'input_meshes', reorder_meshes, 'in_list')
    workflow.connect(reorder_meshes, 'sorted_list', output_node, 'meshes_sorted_by_suj')

    arrange_subjects = pe.Node(interface=reorder_lists(),
                               name='arrange_subjects')
    workflow.connect(input_node, 'subject_ids', arrange_subjects, 'in_list_a')
    workflow.connect(input_node, 'subject_ids', arrange_subjects, 'in_list_b')
    workflow.connect(arrange_subjects, 'sorted_list', output_node, 'subject_ids_sorted_by_suj')

    arrange_ages = pe.Node(interface=reorder_lists(),
                               name='arrange_ages')
    workflow.connect(input_node, 'ages', arrange_ages, 'in_list_a')
    workflow.connect(input_node, 'ages', arrange_ages, 'in_list_b')
    workflow.connect(arrange_ages, 'sorted_list', output_node, 'ages_sorted_by_suj')

    return workflow


def create_spatio_temporal_regression_preprocessing(
        label,
        scan_number,
        param_gammaR=1.e-4,
        param_sigmaV=13,
        param_sigmaW=[11, 8, 4, 2],
        param_maxiters=[100, 200, 200, 100],
        param_T=10,
        param_ntries=1,
        param_MPeps=0,
        name='spatio_temporal_regression_preprocessing'
):

    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    # Create the input node
    input_node = pe.Node(niu.IdentityInterface(
        fields=['input_meshes',
                'input_meshes_sorted',
                'ages',
                'ages_per_suj',
                'subject_ids',
                'subject_ids_by_suj',
                'xml_dkw',
                'xml_dkt',
                'xml_dtp',
                'xml_dsk',
                'xml_dcps',
                'xml_dcpp',
                'xml_dfcp',
                'xml_dmi',
                'xml_dat',
                'xml_dls',
                'xml_ods',
                'xml_okw',
                'xml_ot',
                'out_xmlDiffeos',
                'out_xmlObject']),
        name='input_node')

    # Create the output node
    output_node = pe.Node(niu.IdentityInterface(
        fields=['out_xmlDiffeo', 'out_sym_vtk_files', 'out_init_shape_vtk_file',
                'out_xmlObject', 'out_AgeToOnsetNorm_file',
                'structure_data']),
        name='output_node')

    compute_mean_shape_of_subject = atlas_computation(True,
                                                      dkw=8,
                                                      okw=[4],
                                                      dmi=200,
                                                      dtp=10,
                                                      ods=[0.7],
                                                      type_xml_file='All',
                                                      name='compute_mean_shape_of_subject'
                                                      )
    # workflow.connect(islist, 'bool_listOfList', compute_mean_shape_of_subject, 'input_node.islistoflist')
    workflow.connect(input_node, 'input_meshes_sorted', compute_mean_shape_of_subject, 'input_node.input_vtk_meshes')
    workflow.connect(input_node, 'subject_ids', compute_mean_shape_of_subject, 'input_node.subject_ids')
    workflow.connect(input_node, 'subject_ids_by_suj', compute_mean_shape_of_subject, 'input_node.subject_ids_2')

    age2onset_file_age = pe.Node(interface=Function(function=write_age2onset_file,
                                                    input_names=['age', 'sub_id', 'bool_1persuj'],
                                                    output_names='file_name'),
                                 name='age2onset_file_age')
    age2onset_file_age.inputs.bool_1persuj = False
    workflow.connect(input_node, 'ages_per_suj', age2onset_file_age, 'age')
    workflow.connect(input_node, 'subject_ids_by_suj', age2onset_file_age, 'sub_id')
    workflow.connect(age2onset_file_age, 'file_name', output_node, 'out_AgeToOnsetNorm_file')

    k = 10
    extract_youngest_subjects = pe.Node(interface=extractSubList(),
                                        name='extract_youngest_subjects')
    extract_youngest_subjects.inputs.k = k
    workflow.connect(input_node, 'ages_per_suj', extract_youngest_subjects, 'sorting_reference')
    workflow.connect(compute_mean_shape_of_subject, 'output_node.out_template_vtk_file',
                     extract_youngest_subjects, 'in_list')

    compute_initial_shape_regression = atlas_computation(False,
                                                         dkw=8,
                                                         okw=[4],
                                                         dmi=200,
                                                         dtp=10,
                                                         type_xml_file='All',
                                                         name='compute_initial_shape_regression'
                                                         )
    workflow.connect(extract_youngest_subjects, 'out_sublist',
                     compute_initial_shape_regression, 'input_node.input_vtk_meshes')
    workflow.connect(input_node, 'subject_ids_by_suj', compute_initial_shape_regression, 'input_node.subject_ids')

    writeXmlParametersFiles = pe.Node(interface=WriteXMLFiles(),
                                      name='writeXmlParametersFiles')
    workflow.connect(input_node, 'xml_dkw', writeXmlParametersFiles, 'dkw')
    workflow.connect(input_node, 'xml_dkt', writeXmlParametersFiles, 'dkt')
    workflow.connect(input_node, 'xml_dtp', writeXmlParametersFiles, 'dtp')
    workflow.connect(input_node, 'xml_dsk', writeXmlParametersFiles, 'dsk')
    workflow.connect(input_node, 'xml_dcps', writeXmlParametersFiles, 'dcps')
    # workflow.connect(input_node, 'xml_dcpp', writeXmlParametersFiles, 'dcpp')
   #  workflow.connect(compute_initial_shape_regression, 'output_node.out_template_vtk_file', writeXmlParametersFiles, 'dcpp')
    workflow.connect(input_node, 'xml_dfcp', writeXmlParametersFiles, 'dfcp')
    workflow.connect(input_node, 'xml_dmi', writeXmlParametersFiles, 'dmi')
    workflow.connect(input_node, 'xml_dat', writeXmlParametersFiles, 'dat')
    workflow.connect(input_node, 'xml_dls', writeXmlParametersFiles, 'dls')

    workflow.connect(input_node, 'xml_ods', writeXmlParametersFiles, 'ods')
    workflow.connect(input_node, 'xml_okw', writeXmlParametersFiles, 'okw')
    workflow.connect(input_node, 'xml_ot', writeXmlParametersFiles, 'ot')

    # Connect the data to return
    workflow.connect(writeXmlParametersFiles, 'out_xmlDiffeo', output_node, 'out_xmlDiffeo')
    workflow.connect(writeXmlParametersFiles, 'out_xmlObject', output_node, 'out_xmlObject')
    workflow.connect(compute_mean_shape_of_subject, 'output_node.out_template_vtk_file',
                     output_node, 'out_sym_vtk_files')
    workflow.connect(compute_initial_shape_regression, 'output_node.out_template_vtk_file',
                     output_node, 'out_init_shape_vtk_file')

    return workflow


def create_symmetric_spatio_temporal_analysis(labels,
                                              reduction_rate,
                                              rigid_iteration=1,
                                              affine_iteration=2,
                                              scan_number=2,
                                              name='spatio_temporal_analysis'
                                              ):
    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    # Create the input input node
    input_node = pe.Node(niu.IdentityInterface(
        fields=['input_images',
                'input_parcellations',
                'input_ref',
                'label_indices',
                'ages',
                'subject_ids',
                'xml_dkw',
                'xml_dkt',
                'xml_dtp',
                'xml_dsk',
                'xml_dcps',
                'xml_dcpp',
                'xml_dfcp',
                'xml_dmi',
                'xml_dat',
                'xml_dls',
                'xml_ods',
                'xml_okw',
                'xml_ot']),
        name='input_node')

    # Create the output node
    output_node = pe.Node(niu.IdentityInterface(
        fields=['extracted_meshes', 'out_template_vtk_file',
                'param_diffeo_file', 'param_object_file', 'out_AgeToOnsetNorm_file',
                'out_init_shape_vtk_file', 'out_sym_vtk_files',
                'transported_res_mom', 'transported_res_vect', 'out_file_CP', 'out_file_MOM']),
        name='output_node')

    # COMPUTING THE SYMMETRIC OF THE IMAGES BEFORE COMPUTING THE ATLAS
    symmetrisation = create_symmetrical_images()
    workflow.connect(input_node, 'input_images', symmetrisation, 'input_node.input_images')
    workflow.connect(input_node, 'input_parcellations', symmetrisation, 'input_node.input_parcellations')

    # Create a sub-workflow for groupwise registration
    groupwise = create_atlas(itr_rigid=rigid_iteration,
                             itr_affine=affine_iteration,
                             itr_non_lin=0,
                             verbose=False,
                             name='groupwise')
    workflow.connect(symmetrisation, 'output_node.LR_images', groupwise, 'input_node.in_files')
    workflow.connect(input_node, 'input_ref', groupwise, 'input_node.ref_file')

    # Create the workflow to create the meshes in an average space
    meshes_workflow = create_binary_to_meshes(label=labels, reduction_rate=reduction_rate)
    workflow.connect(symmetrisation, 'output_node.LR_images', meshes_workflow, 'input_node.input_images')
    workflow.connect(symmetrisation, 'output_node.LR_parcellations', meshes_workflow, 'input_node.input_parcellations')
    workflow.connect(groupwise, 'output_node.trans_files', meshes_workflow, 'input_node.trans_files')
    workflow.connect(groupwise, 'output_node.average_image', meshes_workflow, 'input_node.ref_file')
    workflow.connect(meshes_workflow, 'output_node.output_meshes',
                     output_node, 'extracted_meshes')

    # order the lists of meshes to be [[L R] [L R] [L R]], and age and subject ID accordingly:
    reorder_lists = create_lists_symmetrical_subject()
    workflow.connect(meshes_workflow, 'output_node.output_meshes',reorder_lists, 'input_node.input_meshes')
    workflow.connect(input_node, 'ages', reorder_lists, 'input_node.ages')
    workflow.connect(input_node, 'subject_ids', reorder_lists, 'input_node.subject_ids')
    # Create the workflow to generate the required data for the regression
    # Done for only one label. Should be doable for a set a label, we would analyse together.

    preprocessing_regression = create_spatio_temporal_regression_preprocessing(label=labels[0],
                                                                               scan_number=scan_number
                                                                               )
    workflow.connect(meshes_workflow, 'output_node.output_meshes',
                     preprocessing_regression, 'input_node.input_meshes')
    workflow.connect(reorder_lists, 'output_node.meshes_sorted_by_suj',
                     preprocessing_regression, 'input_node.input_meshes_sorted')
    workflow.connect(reorder_lists, 'output_node.ages_sorted_by_suj',
                     preprocessing_regression, 'input_node.ages')
    workflow.connect(input_node, 'ages',
                     preprocessing_regression, 'input_node.ages_per_suj')
    workflow.connect(reorder_lists, 'output_node.subject_ids_sorted_by_suj',
                     preprocessing_regression, 'input_node.subject_ids')
    workflow.connect(input_node, 'subject_ids',
                     preprocessing_regression, 'input_node.subject_ids_by_suj')
    workflow.connect(input_node, 'xml_dkw',
                     preprocessing_regression, 'input_node.xml_dkw')
    workflow.connect(input_node, 'xml_dkt',
                     preprocessing_regression, 'input_node.xml_dkt')
    workflow.connect(input_node, 'xml_dtp',
                     preprocessing_regression, 'input_node.xml_dtp')
    workflow.connect(input_node, 'xml_dsk',
                     preprocessing_regression, 'input_node.xml_dsk')
    workflow.connect(input_node, 'xml_dcps',
                     preprocessing_regression, 'input_node.xml_dcps')
    workflow.connect(input_node, 'xml_dcpp',
                     preprocessing_regression, 'input_node.xml_dcpp')
    workflow.connect(input_node, 'xml_dfcp',
                     preprocessing_regression, 'input_node.xml_dfcp')
    workflow.connect(input_node, 'xml_dmi',
                     preprocessing_regression, 'input_node.xml_dmi')
    workflow.connect(input_node, 'xml_dat',
                     preprocessing_regression, 'input_node.xml_dat')
    workflow.connect(input_node, 'xml_dls',
                     preprocessing_regression, 'input_node.xml_dls')
    workflow.connect(input_node, 'xml_ods',
                     preprocessing_regression, 'input_node.xml_ods')
    workflow.connect(input_node, 'xml_okw',
                     preprocessing_regression, 'input_node.xml_okw')
    workflow.connect(input_node, 'xml_ot',
                     preprocessing_regression, 'input_node.xml_ot')
    workflow.connect(preprocessing_regression, 'output_node.out_xmlDiffeo', output_node, 'param_diffeo_file')
    workflow.connect(preprocessing_regression, 'output_node.out_xmlObject', output_node, 'param_object_file')
    workflow.connect(preprocessing_regression, 'output_node.out_AgeToOnsetNorm_file',
                     output_node, 'out_AgeToOnsetNorm_file')
    workflow.connect(preprocessing_regression, 'output_node.out_init_shape_vtk_file',
                     output_node, 'out_template_vtk_file')
    workflow.connect(preprocessing_regression, 'output_node.out_init_shape_vtk_file',
                     output_node, 'out_init_shape_vtk_file')
    workflow.connect(preprocessing_regression, 'output_node.out_sym_vtk_files',
                     output_node, 'out_sym_vtk_files')
    # workflow.connect(preprocessing_regression, 'output_node.out_vertices_centroid_file',
    #                  output_node, 'out_vertices_centroid_file')

    # Create the workflow for the computation of the regression and residual deformations transportation
    computation_regression = create_get_shape_distance_from_regression(scan_number=scan_number)
    workflow.connect(preprocessing_regression, 'output_node.out_sym_vtk_files',
                     computation_regression, 'input_node.input_meshes')
    workflow.connect(preprocessing_regression, 'output_node.out_xmlDiffeo',
                     computation_regression, 'input_node.xmlDiffeo')
    workflow.connect(preprocessing_regression, 'output_node.out_xmlObject',
                     computation_regression, 'input_node.xmlObject')
    workflow.connect(preprocessing_regression, 'output_node.out_init_shape_vtk_file',
                     computation_regression, 'input_node.baseline_vtk_file')
    workflow.connect(preprocessing_regression, 'output_node.out_AgeToOnsetNorm_file',
                     computation_regression, 'input_node.in_AgeToOnsetNorm_file')
    workflow.connect(computation_regression, 'output_node.transported_res_mom',
                     output_node, 'transported_res_mom')
    workflow.connect(computation_regression, 'output_node.transported_res_vect',
                     output_node, 'transported_res_vect')
    workflow.connect(computation_regression, 'output_node.out_file_CP',
                     output_node, 'out_file_CP')
    workflow.connect(computation_regression, 'output_node.out_file_MOM',
                     output_node, 'out_file_MOM')

    return workflow



# This function does only the preprocessing steps at the moment. Should be renamed in the future,
# so the function also does: global template of time point 1,
# linking the template to the time point 1 of each subject,
# computing the individual trajectories,
# transporting the individual evolution to the template.
def create_preprocessing_shape_analysis_epilepsy_flipping(labels,
                                              reduction_rate,
                                              rigid_iteration=1,
                                              affine_iteration=2,
                                              scan_number=2,
                                              name='spatio_temporal_analysis'
                                              ):
    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    # Create the input input node
    input_node = pe.Node(niu.IdentityInterface(
        fields=['input_images',
                'input_ref',
                'flip_id',
                'no_flip_id',
                'no_flip_seg',
                'flip_seg',
                'subject_ids']),
        name='input_node')

    # Create the output node
    output_node = pe.Node(niu.IdentityInterface(
        fields=['extracted_meshes']),
        name='output_node')

    # Extract the sublist of parcelation and T1.
    split_list_to_flip_images = pe.Node(interface=Function(function=split_list,
                                                    input_names=['including_id', 'all_id', 'list_data'],
                                                    output_names='extracted_list'),
                                 name='split_list_to_flip_images')
    workflow.connect(input_node, 'flip_id', split_list_to_flip_images, 'including_id')
    workflow.connect(input_node, 'subject_ids', split_list_to_flip_images, 'all_id')
    workflow.connect(input_node, 'input_images', split_list_to_flip_images, 'list_data')

    split_list_to_not_flip_images = pe.Node(interface=Function(function=split_list,
                                                    input_names=['including_id', 'all_id', 'list_data'],
                                                    output_names='extracted_list'),
                                 name='split_list_to_not_flip_images')
    workflow.connect(input_node, 'no_flip_id', split_list_to_not_flip_images, 'including_id')
    workflow.connect(input_node, 'subject_ids', split_list_to_not_flip_images, 'all_id')
    workflow.connect(input_node, 'input_images', split_list_to_not_flip_images, 'list_data')

    split_list_to_flip_seg = pe.Node(interface=Function(function=split_list,
                                                    input_names=['including_id', 'all_id', 'list_data'],
                                                    output_names='extracted_list'),
                                 name='split_list_to_flip_seg')
    workflow.connect(input_node, 'flip_id', split_list_to_flip_seg, 'including_id')
    workflow.connect(input_node, 'subject_ids', split_list_to_flip_seg, 'all_id')
    workflow.connect(input_node, 'flip_seg', split_list_to_flip_seg, 'list_data')

    split_list_to_not_flip_seg = pe.Node(interface=Function(function=split_list,
                                                    input_names=['including_id', 'all_id', 'list_data'],
                                                    output_names='extracted_list'),
                                 name='split_list_to_not_flip_seg')
    workflow.connect(input_node, 'no_flip_id', split_list_to_not_flip_seg, 'including_id')
    workflow.connect(input_node, 'subject_ids', split_list_to_not_flip_seg, 'all_id')
    workflow.connect(input_node, 'no_flip_seg', split_list_to_not_flip_seg, 'list_data')

    # COMPUTING THE SYMMETRIC OF THE IMAGES BEFORE COMPUTING THE ATLAS
    swap_images = pe.MapNode(interface=SwapDimImage(), iterfield="image2reorient", name="swap_images")
    workflow.connect(split_list_to_flip_images, 'extracted_list', swap_images, 'image2reorient')
    swap_images.inputs.axe2flip = "LR"

    # create a list of LR swapped parcellations:
    swap_parcellations = pe.MapNode(interface=SwapDimImage(), iterfield="image2reorient", name="swap_parcellations")
    workflow.connect(split_list_to_flip_seg, 'extracted_list', swap_parcellations, 'image2reorient')
    swap_parcellations.inputs.axe2flip = "LR"

    # merge the lists
    merge_lists_images = pe.Node(interface=niu.Merge(axis='vstack', numinputs=2),
                                 name='merge_lists_images')
    workflow.connect(split_list_to_not_flip_images, 'extracted_list', merge_lists_images, 'in1')
    workflow.connect(swap_images, 'flipped_image', merge_lists_images, 'in2')

    merge_lists_seg = pe.Node(interface=niu.Merge(axis='vstack', numinputs=2),
                                 name='merge_lists_seg')
    workflow.connect(split_list_to_not_flip_seg, 'extracted_list', merge_lists_seg, 'in1')
    workflow.connect(swap_parcellations, 'flipped_image', merge_lists_seg, 'in2')

    # Create a sub-workflow for groupwise registration
    groupwise = create_atlas(itr_rigid=rigid_iteration,
                             itr_affine=affine_iteration,
                             itr_non_lin=0,
                             verbose=False,
                             name='groupwise')
    workflow.connect(merge_lists_images, 'out', groupwise, 'input_node.in_files')
    workflow.connect(input_node, 'input_ref', groupwise, 'input_node.ref_file')

    # Create the workflow to create the meshes in an average space
    meshes_workflow = create_binary_to_meshes(label=labels, reduction_rate=reduction_rate)
    workflow.connect(merge_lists_images, 'out', meshes_workflow, 'input_node.input_images')
    workflow.connect(merge_lists_seg, 'out', meshes_workflow, 'input_node.input_parcellations')
    workflow.connect(groupwise, 'output_node.trans_files', meshes_workflow, 'input_node.trans_files')
    workflow.connect(groupwise, 'output_node.average_image', meshes_workflow, 'input_node.ref_file')
    workflow.connect(meshes_workflow, 'output_node.output_meshes',
                     output_node, 'extracted_meshes')


    return workflow

