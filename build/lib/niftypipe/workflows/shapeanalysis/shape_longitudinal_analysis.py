# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:


import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function
from nipype.workflows.smri.niftyreg.groupwise import create_groupwise_average as create_atlas

from .centroid_computation import centroid_computation
from .shape_analysis import create_binary_to_meshes
from niftypipe.interfaces.niftk.io import Image2VtkMesh
from ...interfaces.niftk.utils import MergeLabels, extractSubList, SwapDimImage
from ...interfaces.shapeAnalysis import (longitudinal_splitBaselineFollowup, WriteXMLFiles,
                                         SparseGeodesicRegression3, SparseMatching3, sortingTimePoints,
                                         ShootAndFlow3, ParallelTransport, split_list2)


# def create_binary_to_meshes(label,
#                             name='gw_binary_to_meshes',
#                             reduction_rate=0.3):
#     # Create the workflow
#     workflow = pe.Workflow(name=name)
#     workflow.base_output_dir = name
#
#     # Create the input node
#     input_node = pe.Node(niu.IdentityInterface(
#         fields=['input_images',
#                 'input_parcellations',
#                 'input_reference',
#                 'trans_files',
#                 'ref_file']),
#         name='input_node')
#
#     # Create the output node
#     output_node = pe.Node(niu.IdentityInterface(
#         fields=['output_meshes']),
#         name='output_node')
#
#     # Extract the relevant label from the GIF parcellation
#     extract_label = pe.MapNode(interface=MergeLabels(),
#                                iterfield=['in_file'],
#                                name='extract_label')
#     extract_label.inputs.roi_list = label
#     workflow.connect(input_node, 'input_parcellations', extract_label, 'in_file')
#
#     # Removing parasite segmentation: Erosion.
#     erode_binaries = pe.MapNode(interface=niftyseg.BinaryMathsInteger(operation='ero', operand_value=2),
#                                 iterfield=['in_file'], name='erode_binaries')
#     workflow.connect(extract_label, 'out_file', erode_binaries, 'in_file')
#
#     # Removing parasite segmentation: Dilatation.
#     dilate_binaries = pe.MapNode(interface=niftyseg.BinaryMathsInteger(operation='dil', operand_value=2),
#                                  iterfield=['in_file'], name='dilate_binaries')
#     workflow.connect(erode_binaries, 'out_file', dilate_binaries, 'in_file')
#     # Volume extraction
#
#     # Flipping the image
#
#     # Apply the relevant transformations to the roi
#     apply_affine = pe.MapNode(interface=niftyreg.RegResample(inter_val='NN'),
#                               iterfield=['flo_file',
#                                          'trans_file'],
#                               name='apply_affine')
#     workflow.connect(input_node, 'trans_files', apply_affine, 'trans_file')
#     workflow.connect(input_node, 'ref_file', apply_affine, 'ref_file')
#     workflow.connect(dilate_binaries, 'out_file', apply_affine, 'flo_file')
#
#     # compute the large ROI that correspond to the union of all warped label
#     extract_union_roi = pe.Node(interface=niftyreg.RegAverage(),
#                                 name='extract_union_roi')
#     workflow.connect(apply_affine, 'out_file', extract_union_roi, 'avg_files')
#
#     # Binarise the average ROI
#     binarise_roi = pe.Node(interface=niftyseg.UnaryMaths(operation='bin'),
#                            name='binarise_roi')
#     workflow.connect(extract_union_roi, 'out_file', binarise_roi, 'in_file')
#
#     # Dilation of the binarise union ROI
#     dilate_roi = pe.Node(interface=niftyseg.BinaryMathsInteger(operation='dil', operand_value=5),
#                          name='dilate_roi')
#     workflow.connect(binarise_roi, 'out_file', dilate_roi, 'in_file')
#
#     # Apply the transformations
#     apply_rigid_refinement = pe.MapNode(interface=niftyreg.RegAladin(rig_only_flag=True, ln_val=1),
#                                         iterfield=['flo_file', 'in_aff_file'],
#                                         name='apply_rigid_refinement')
#     workflow.connect(input_node, 'input_images', apply_rigid_refinement, 'flo_file')
#     workflow.connect(input_node, 'ref_file', apply_rigid_refinement, 'ref_file')
#     workflow.connect(input_node, 'trans_files', apply_rigid_refinement, 'in_aff_file')
#     workflow.connect(dilate_roi, 'out_file', apply_rigid_refinement, 'rmask_file')
#
#     # Extract the mesh corresponding to the label
#     final_resampling = pe.MapNode(interface=niftyreg.RegResample(inter_val='NN'),
#                                   iterfield=['flo_file',
#                                              'trans_file'],
#                                   name='final_resampling')
#     workflow.connect(apply_rigid_refinement, 'aff_file', final_resampling, 'trans_file')
#     workflow.connect(input_node, 'ref_file', final_resampling, 'ref_file')
#     workflow.connect(dilate_binaries, 'out_file', final_resampling, 'flo_file')
#
#     # Extract the mesh corresponding to the label
#     extract_mesh = pe.MapNode(interface=Image2VtkMesh(in_reductionRate=reduction_rate),
#                               iterfield=['in_file'],
#                               name='extract_mesh')
#     workflow.connect(final_resampling, 'out_file', extract_mesh, 'in_file')
#     # workflow.connect(apply_rigid_refinement, 'aff_file', extract_mesh, 'matrix_file')
#
#     # Create a rename for the average image
#     groupwise_renamer = pe.Node(interface=niu.Rename(format_string='atlas', keep_ext=True),
#                                 name='groupwise_renamer')
#     workflow.connect(input_node, 'ref_file', groupwise_renamer, 'in_file')
#
#     workflow.connect(extract_mesh, 'out_file', output_node, 'output_meshes')
#     return workflow


def create_spatio_temporal_regression_preprocessing(
        label,
        scan_number,
        number_followup,
        param_gammaR=1.e-4,
        param_sigmaV=13,
        param_sigmaW=[11, 8, 4, 2],
        param_maxiters=[100, 200, 200, 100],
        param_T=10,
        param_ntries=1,
        param_MPeps=0,
        name='spatio_temporal_regression_preprocessing'):

    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    # Create the input node
    input_node = pe.Node(niu.IdentityInterface(
        fields=['input_meshes',
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
                'xml_ot',
                'out_xmlDiffeos',
                'out_xmlObject']),
        name='input_node')

    # Create the output node
    output_node = pe.Node(niu.IdentityInterface(
        fields=['b0_vertices_files', 'b0_meshes_vtk_files', 'centroid_b0_vtk_file', 'indiv_meshes_vtk_file',
                'xml_diffeo', 'xml_object', 'out_vertices_centroid_file',
                'b0_ageNorm_file', 'indiv_ageNorm_files',
                'struct_mat', 'out_centroid_mat_file',
                'subjects_ids_unique']),
        name='output_node')

    # separates the baselines from the follow-up, compute the relative times for the individual regression
    # (in case of more than 2 images)
    splitBaselineFollowup = pe.Node(interface=longitudinal_splitBaselineFollowup(),
                                    name='splitBaselineFollowup')
    #workflow.connect(convertVTK2txt, 'out_verticesFiles', splitBaselineFollowup, 'in_all_verticesFile')
    workflow.connect(input_node, 'input_meshes', splitBaselineFollowup, 'in_all_meshes')
    workflow.connect(input_node, 'ages', splitBaselineFollowup, 'in_all_ages')
    workflow.connect(input_node, 'subject_ids', splitBaselineFollowup, 'in_all_subj_ids')
    splitBaselineFollowup.inputs.number_followup = number_followup

    workflow.connect(splitBaselineFollowup, 'b0_meshes', output_node, 'b0_meshes_vtk_files')
    workflow.connect(splitBaselineFollowup, 'subject_ids_unique', output_node, 'subjects_ids_unique')
    workflow.connect(splitBaselineFollowup, 'indiv_meshes', output_node, 'indiv_meshes_vtk_file')
    workflow.connect(splitBaselineFollowup, 'indiv_age2onset_norm_file', output_node, 'indiv_ageNorm_files')
    workflow.connect(splitBaselineFollowup, 'b0_age2onset_norm_file', output_node, 'b0_ageNorm_file')

    k = 5
    extract_youngest_subjects_meshes = pe.Node(interface=extractSubList(),
                                               name='extract_youngest_subjects_meshes')
    extract_youngest_subjects_meshes.inputs.k = k
    workflow.connect(splitBaselineFollowup, 'b0_ages', extract_youngest_subjects_meshes, 'sorting_reference')
    workflow.connect(splitBaselineFollowup, 'b0_meshes', extract_youngest_subjects_meshes, 'in_list')

    extract_youngest_subjID = pe.Node(interface=extractSubList(),
                                      name='extract_youngest_subjID')
    extract_youngest_subjID.inputs.k = k
    workflow.connect(splitBaselineFollowup, 'b0_ages', extract_youngest_subjID, 'sorting_reference')
    workflow.connect(splitBaselineFollowup, 'subject_ids_unique', extract_youngest_subjID, 'in_list')

    compute_initial_shape_regression = centroid_computation(k,
                                                            label,
                                                            False,
                                                            param_gammaR=param_gammaR,
                                                            param_sigmaV=param_sigmaV,
                                                            param_sigmaW=param_sigmaW,
                                                            param_maxiters=param_maxiters,
                                                            param_T=param_T,
                                                            param_ntries=param_ntries,
                                                            param_MPeps=param_MPeps,
                                                            name='compute_initial_shape_regression'
                                                            )
    workflow.connect(extract_youngest_subjects_meshes, 'sorted_list_ref', compute_initial_shape_regression, 'input_node.ages')
    workflow.connect(extract_youngest_subjects_meshes, 'out_sublist',
                     compute_initial_shape_regression, 'input_node.input_vtk_meshes')
    workflow.connect(extract_youngest_subjID, 'out_sublist', compute_initial_shape_regression, 'input_node.subject_ids')
    workflow.connect(compute_initial_shape_regression, 'output_node.out_AgeToOnsetNorm_file',
                     output_node, 'out_AgeToOnsetNorm_file')

    workflow.connect(compute_initial_shape_regression, 'output_node.out_centroid_vtk_file',
                     output_node, 'centroid_b0_vtk_file')
    writeXmlParametersFiles = pe.Node(interface=WriteXMLFiles(),
                                      name='writeXmlParametersFiles')
    workflow.connect(input_node, 'xml_dkw', writeXmlParametersFiles, 'dkw')
    workflow.connect(input_node, 'xml_dkt', writeXmlParametersFiles, 'dkt')
    workflow.connect(input_node, 'xml_dtp', writeXmlParametersFiles, 'dtp')
    workflow.connect(input_node, 'xml_dsk', writeXmlParametersFiles, 'dsk')
    workflow.connect(input_node, 'xml_dcps', writeXmlParametersFiles, 'dcps')
    workflow.connect(input_node, 'xml_dcpp', writeXmlParametersFiles, 'dcpp')
    workflow.connect(input_node, 'xml_dfcp', writeXmlParametersFiles, 'dfcp')
    workflow.connect(input_node, 'xml_dmi', writeXmlParametersFiles, 'dmi')
    workflow.connect(input_node, 'xml_dat', writeXmlParametersFiles, 'dat')
    workflow.connect(input_node, 'xml_dls', writeXmlParametersFiles, 'dls')

    workflow.connect(input_node, 'xml_ods', writeXmlParametersFiles, 'ods')
    workflow.connect(input_node, 'xml_okw', writeXmlParametersFiles, 'okw')
    workflow.connect(input_node, 'xml_ot', writeXmlParametersFiles, 'ot')

    # Connect the data to return
    workflow.connect(writeXmlParametersFiles, 'out_xmlDiffeo', output_node, 'xml_diffeo')
    workflow.connect(writeXmlParametersFiles, 'out_xmlObject', output_node, 'xml_object')

    return workflow


def create_global_regression(nb_subject, name='create_global_regression'):
    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    # Create the input node
    input_node = pe.Node(niu.IdentityInterface(
        fields=['b0_meshes_vtk_files', 'centroid_b0_vtk_file',
                'xml_diffeo', 'xml_object',
                'b0_ageNorm_file',
                ]),
        name='input_node')

    # Create the output node
    output_node = pe.Node(niu.IdentityInterface(
        fields=['CP_file_global', 'MOM_file_global',
                'CP_global_traj_files_txt', 'global_traj_files_vtk',
                'out_t_from', 'files_global_trajectory_source', 'xmlDiffeo_indiv',
                'CP_files_indiv_traj', 'MOM_files_indiv_traj']),
        name='output_node')

    computeSTregression = pe.Node(interface=SparseGeodesicRegression3(),
                                  name='computeSTregression')
    workflow.connect(input_node, 'b0_meshes_vtk_files', computeSTregression, 'in_subjects')
    workflow.connect(input_node, 'b0_ageNorm_file', computeSTregression, 'in_time')
    workflow.connect(input_node, 'xml_diffeo', computeSTregression, 'in_paramDiffeo')
    workflow.connect(input_node, 'xml_object', computeSTregression, 'in_paramObjects')
    workflow.connect(input_node, 'centroid_b0_vtk_file', computeSTregression, 'in_initTemplates')

    workflow.connect(computeSTregression, 'out_file_CP', output_node, 'CP_file_global')
    workflow.connect(computeSTregression, 'out_file_MOM', output_node, 'MOM_file_global')

    shooting_st_regression = pe.Node(interface=ShootAndFlow3(),
                                     name='shooting_st_regression')
    shooting_st_regression.inputs.in_direction = 1
    workflow.connect(computeSTregression, 'out_file_CP', shooting_st_regression, 'in_cp_file')
    workflow.connect(computeSTregression, 'out_file_MOM', shooting_st_regression, 'in_mom_file')
    workflow.connect(input_node, 'xml_diffeo', shooting_st_regression, 'in_paramDiffeo')
    workflow.connect(input_node, 'xml_object', shooting_st_regression, 'in_paramObjects')
    workflow.connect(input_node, 'centroid_b0_vtk_file', shooting_st_regression, 'in_sources')
    workflow.connect(shooting_st_regression, 'out_CP_files_txt', output_node, 'CP_global_traj_files_txt')
    workflow.connect(shooting_st_regression, 'out_files_vtk', output_node, 'global_traj_files_vtk')

    build_target_file_list4residual_comp = pe.Node(interface=sortingTimePoints(),
                                                   name='build_target_file_list4residual_comp')
    workflow.connect(shooting_st_regression, 'out_files_vtk',
                     build_target_file_list4residual_comp, 'in_timePoints_vtkfile')
    workflow.connect(input_node, 'xml_diffeo', build_target_file_list4residual_comp, 'in_xmlDiffeo')
    workflow.connect(input_node, 'b0_ageNorm_file',
                     build_target_file_list4residual_comp, 'in_AgeToOnsetNorm_file')
    workflow.connect(shooting_st_regression, 'out_CP_files_txt', build_target_file_list4residual_comp, 'in_CP_txtfiles')

    workflow.connect(build_target_file_list4residual_comp, 'out_t_from',
                     output_node, 'out_t_from')
    workflow.connect(build_target_file_list4residual_comp, 'files_trajectory_source',
                     output_node, 'files_global_trajectory_source')
    workflow.connect(build_target_file_list4residual_comp, 'files_xmlDiffeo', output_node, 'xmlDiffeo_indiv')

    # Compute the residual deformation from the global trajectory to the baselines shapes
    compute_residual_deformations_from_globalTraj = pe.MapNode(interface=SparseMatching3(),
                                                               iterfield=['in_sources', 'in_targets','in_paramDiffeo'],
                                                               name='compute_residual_deformations_from_globalTraj')
    workflow.connect(build_target_file_list4residual_comp, 'files_trajectory_source',
                     compute_residual_deformations_from_globalTraj, 'in_sources')
    workflow.connect(input_node, 'b0_meshes_vtk_files',
                     compute_residual_deformations_from_globalTraj, 'in_targets')
    workflow.connect(input_node, 'xml_object', compute_residual_deformations_from_globalTraj, 'in_paramObjects')
    workflow.connect(build_target_file_list4residual_comp, 'files_xmlDiffeo',
                     compute_residual_deformations_from_globalTraj, 'in_paramDiffeo')

    workflow.connect(compute_residual_deformations_from_globalTraj, 'out_file_CP',
                     output_node, 'CP_files_indiv_traj')
    workflow.connect(compute_residual_deformations_from_globalTraj, 'out_file_MOM',
                     output_node, 'MOM_files_indiv_traj')

    return workflow


def create_individual_regressions_get_shape_information(name='create_individual_regressions_get_shape_information'):
    # loop on the subject: create the list of targets and sources for the computation of the residual deformations:

    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    # Create the input node
    input_node = pe.Node(niu.IdentityInterface(
        fields=['CP_file_global', 'MOM_file_global', 'global_traj_files_vtk', 'centroid_b0_vtk_file',
                'xmlDiffeo_global', 'xmlObject_global', 'xmlDiffeo_indiv',
                'subject_ids_unique', 'global_t_from', 'files_global_trajectory_source',
                'input_indiv_meshes', 'indiv_AgeToOnsetNorm_file',
                'CP_files_indiv_traj', 'MOM_files_indiv_traj', 'xml_dkw',
                'xml_dkt',
                'xml_dtp',
                'xml_dsk',
                'xml_dcps',
                'xml_dfcp',
                'xml_dmi',
                'xml_dat',
                'xml_dls'
                ]),
        name='input_node')

    # Create the output node
    output_node = pe.Node(niu.IdentityInterface(
        fields=['transported_res_mom', 'transported_res_vect',
                ]),
        name='output_node')

    # Shoot for computing the matching of each TP to the subject baselines.
    shooting_indiv_residual = pe.MapNode(interface=ShootAndFlow3(),
                                         iterfield=['in_cp_file', 'in_mom_file', 'in_sources', 'in_paramDiffeo'],
                                         name='shooting_indiv_residual')
    shooting_indiv_residual.inputs.in_direction = 1
    workflow.connect(input_node, 'CP_files_indiv_traj', shooting_indiv_residual, 'in_cp_file')
    workflow.connect(input_node, 'MOM_files_indiv_traj', shooting_indiv_residual, 'in_mom_file')
    workflow.connect(input_node, 'xmlDiffeo_indiv', shooting_indiv_residual, 'in_paramDiffeo')
    workflow.connect(input_node, 'xmlObject_global', shooting_indiv_residual, 'in_paramObjects')
    workflow.connect(input_node, 'files_global_trajectory_source', shooting_indiv_residual, 'in_sources')

    # write the new xml Diffeo files needed for the individual regression. Update the initial position of the CP
    writeXmlParamDiffeo_indiv = pe.MapNode(interface=WriteXMLFiles(),
                                           iterfield=['dcpp'],
                                           name='writeXmlParamDiffeo_indiv')
    workflow.connect(input_node, 'xml_dkw', writeXmlParamDiffeo_indiv, 'dkw')
    workflow.connect(input_node, 'xml_dkt', writeXmlParamDiffeo_indiv, 'dkt')
    workflow.connect(input_node, 'xml_dtp', writeXmlParamDiffeo_indiv, 'dtp')
    workflow.connect(input_node, 'xml_dsk', writeXmlParamDiffeo_indiv, 'dsk')
    workflow.connect(shooting_indiv_residual, 'out_CP_last_txt', writeXmlParamDiffeo_indiv, 'dcpp')
    workflow.connect(input_node, 'xml_dfcp', writeXmlParamDiffeo_indiv, 'dfcp')
    workflow.connect(input_node, 'xml_dmi', writeXmlParamDiffeo_indiv, 'dmi')
    workflow.connect(input_node, 'xml_dat', writeXmlParamDiffeo_indiv, 'dat')
    workflow.connect(input_node, 'xml_dls', writeXmlParamDiffeo_indiv, 'dls')
    writeXmlParamDiffeo_indiv.inputs.type_xml_file == 'Def'
    writeXmlParamDiffeo_indiv.inputs.xml_diffeo == 'parametersDiffeo_indiv.xml'

    # Compute the individual trajectories of the subjects.
    compute_indiv_st_regression = pe.MapNode(interface=SparseGeodesicRegression3(),
                                             iterfield=['in_subjects', 'in_time', 'in_initTemplates', 'in_paramDiffeo'],
                                  name='compute_indiv_st_regression')
    workflow.connect(input_node, 'input_indiv_meshes', compute_indiv_st_regression, 'in_subjects')
    workflow.connect(input_node, 'indiv_AgeToOnsetNorm_file', compute_indiv_st_regression, 'in_time')
    workflow.connect(writeXmlParamDiffeo_indiv, 'out_xmlDiffeo', compute_indiv_st_regression, 'in_paramDiffeo')
    workflow.connect(input_node, 'xmlObject_global', compute_indiv_st_regression, 'in_paramObjects')
    workflow.connect(shooting_indiv_residual, 'out_last_vtk', compute_indiv_st_regression, 'in_initTemplates')  # the reordered ones ?

    # Parallel transport the individual vectors to their corresponding time point on the global trajectory.
    parallel_transport_to_indiv_traj = pe.MapNode(interface=ParallelTransport(),
                                                  iterfield=['in_vect', 'in_vtk', 'in_mom', 'in_cp',
                                                             'in_paramDiffeo'],
                                                  name='parallel_transport_to_indiv_traj')
    parallel_transport_to_indiv_traj.inputs.in_boolMom = 1
    parallel_transport_to_indiv_traj.inputs.in_t_to = 0
    workflow.connect(input_node, 'xml_dtp', parallel_transport_to_indiv_traj, 'in_t_from')
    workflow.connect(input_node, 'MOM_files_indiv_traj', parallel_transport_to_indiv_traj, 'in_mom')
    workflow.connect(input_node, 'CP_files_indiv_traj', parallel_transport_to_indiv_traj, 'in_cp')
    workflow.connect(input_node, 'xmlDiffeo_indiv', parallel_transport_to_indiv_traj, 'in_paramDiffeo') # need the new xml files
    workflow.connect(input_node, 'xmlObject_global', parallel_transport_to_indiv_traj, 'in_paramObjects')
    workflow.connect(compute_indiv_st_regression, 'out_file_MOM', parallel_transport_to_indiv_traj, 'in_vect')
    workflow.connect(input_node, 'files_global_trajectory_source', parallel_transport_to_indiv_traj, 'in_vtk') # the reordered ones ?

    # Parallel transport the individual vectors to the initial shape of the global trajectory.
    parallel_transport_to_global_traj = pe.MapNode(interface=ParallelTransport(),
                                                   iterfield=['in_vect', 'in_t_from'],
                                                   name='parallel_transport_to_global_traj')
    parallel_transport_to_global_traj.inputs.in_boolMom = 1
    parallel_transport_to_global_traj.inputs.in_t_to = 0
    workflow.connect(input_node, 'MOM_file_global', parallel_transport_to_global_traj, 'in_mom')
    workflow.connect(input_node, 'CP_file_global', parallel_transport_to_global_traj, 'in_cp')
    workflow.connect(input_node, 'xmlDiffeo_global', parallel_transport_to_global_traj, 'in_paramDiffeo')
    workflow.connect(input_node, 'xmlObject_global', parallel_transport_to_global_traj, 'in_paramObjects')
    workflow.connect(parallel_transport_to_indiv_traj, 'out_transported_mom',
                     parallel_transport_to_global_traj, 'in_vect')
    workflow.connect(input_node, 'global_t_from', parallel_transport_to_global_traj, 'in_t_from')
    workflow.connect(input_node, 'centroid_b0_vtk_file',
                     parallel_transport_to_global_traj, 'in_vtk')

    # Create a rename for the transported momentum
    renamer1 = pe.MapNode(interface=niu.Rename(format_string="suj_%(subject_id)s_Mom_final_TRANSPORT_Mom_0",
                                               keep_ext=True),
                          iterfield=['subject_id', 'in_file'],
                          name='renamer1')
    workflow.connect(input_node, 'subject_ids_unique', renamer1, 'subject_id')
    workflow.connect(parallel_transport_to_global_traj, 'out_transported_mom', renamer1, 'in_file')

    # Create a rename for the transported vector
    renamer2 = pe.MapNode(interface=niu.Rename(format_string="suj_%(subject_id)s_Mom_final_TRANSPORT_0", keep_ext=True),
                          iterfield=['subject_id', 'in_file'],
                          name='renamer2')
    workflow.connect(input_node, 'subject_ids_unique', renamer2, 'subject_id')
    workflow.connect(parallel_transport_to_global_traj, 'out_transported_vect', renamer2, 'in_file')

    workflow.connect(renamer1, 'out_file', output_node, 'transported_res_mom')
    workflow.connect(renamer2, 'out_file', output_node, 'transported_res_vect')

    return workflow


def create_spatio_temporal_longitudinal_analysis(labels,
                                                 nb_followup,
                                                 rigid_iteration=1,
                                                 affine_iteration=2,
                                                 scan_number=2,
                                                 name='spatio_temporal_analysis'
                                                 ):
    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    nb_subject = len(nb_followup)

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
        fields=['extracted_meshes', 'xml_diffeo_global', 'xml_object_global', 'xmlDiffeo_indiv',
                'b0_ageNorm_file', 'centroid_b0_vtk_file', 'CP_file_global', 'MOM_file_global',
                'global_traj_files_vtk', 'transported_res_mom',
                'transported_res_vect', 'struct_mat']),
        name='output_node')

    # Create a sub-workflow for groupwise registration
    groupwise = create_atlas(itr_rigid=rigid_iteration,
                             itr_affine=affine_iteration,
                             itr_non_lin=0,
                             name='groupwise')
    workflow.connect(input_node, 'input_images', groupwise, 'input_node.in_files')
    workflow.connect(input_node, 'input_ref', groupwise, 'input_node.ref_file')

    # Create the workflow to create the meshes in an average space
    meshes_workflow = create_binary_to_meshes(label=labels)
    workflow.connect(input_node, 'input_images', meshes_workflow, 'input_node.input_images')
    workflow.connect(input_node, 'input_parcellations', meshes_workflow, 'input_node.input_parcellations')
    workflow.connect(groupwise, 'output_node.trans_files', meshes_workflow, 'input_node.trans_files')
    workflow.connect(groupwise, 'output_node.average_image', meshes_workflow, 'input_node.ref_file')
    workflow.connect(meshes_workflow, 'output_node.output_meshes',
                     output_node, 'extracted_meshes')
    # Create the workflow to generate the required data for the regression
    # Done for only one label. Should be doable for a set a label, we would analyse together.
    preprocessing_analysis = create_spatio_temporal_regression_preprocessing(label=labels,
                                                                             scan_number=scan_number,
                                                                             number_followup=nb_followup
                                                                             )
    workflow.connect(meshes_workflow, 'output_node.output_meshes',
                     preprocessing_analysis, 'input_node.input_meshes')
    workflow.connect(input_node, 'ages',
                     preprocessing_analysis, 'input_node.ages')
    workflow.connect(input_node, 'subject_ids',
                     preprocessing_analysis, 'input_node.subject_ids')
    workflow.connect(input_node, 'xml_dkw',
                     preprocessing_analysis, 'input_node.xml_dkw')
    workflow.connect(input_node, 'xml_dkt',
                     preprocessing_analysis, 'input_node.xml_dkt')
    workflow.connect(input_node, 'xml_dtp',
                     preprocessing_analysis, 'input_node.xml_dtp')
    workflow.connect(input_node, 'xml_dsk',
                     preprocessing_analysis, 'input_node.xml_dsk')
    workflow.connect(input_node, 'xml_dcps',
                     preprocessing_analysis, 'input_node.xml_dcps')
    workflow.connect(input_node, 'xml_dcpp',
                     preprocessing_analysis, 'input_node.xml_dcpp')
    workflow.connect(input_node, 'xml_dfcp',
                     preprocessing_analysis, 'input_node.xml_dfcp')
    workflow.connect(input_node, 'xml_dmi',
                     preprocessing_analysis, 'input_node.xml_dmi')
    workflow.connect(input_node, 'xml_dat',
                     preprocessing_analysis, 'input_node.xml_dat')
    workflow.connect(input_node, 'xml_dls',
                     preprocessing_analysis, 'input_node.xml_dls')
    workflow.connect(input_node, 'xml_ods',
                     preprocessing_analysis, 'input_node.xml_ods')
    workflow.connect(input_node, 'xml_okw',
                     preprocessing_analysis, 'input_node.xml_okw')
    workflow.connect(input_node, 'xml_ot',
                     preprocessing_analysis, 'input_node.xml_ot')
    workflow.connect(preprocessing_analysis, 'output_node.xml_diffeo',
                     output_node, 'xml_diffeo_global')
    workflow.connect(preprocessing_analysis, 'output_node.xml_object',
                     output_node, 'xml_object_global')
    workflow.connect(preprocessing_analysis, 'output_node.b0_ageNorm_file',
                     output_node, 'b0_ageNorm_file')
    workflow.connect(preprocessing_analysis, 'output_node.centroid_b0_vtk_file',
                     output_node, 'centroid_b0_vtk_file')
    workflow.connect(preprocessing_analysis, 'output_node.struct_mat', output_node, 'struct_mat')

    # Create the workflow for the computation of the global regression
    compute_global__regression = create_global_regression(nb_subject=nb_subject)
    workflow.connect(preprocessing_analysis, 'output_node.b0_meshes_vtk_files',
                     compute_global__regression, 'input_node.b0_meshes_vtk_files')
    workflow.connect(preprocessing_analysis, 'output_node.centroid_b0_vtk_file',
                     compute_global__regression, 'input_node.centroid_b0_vtk_file')
    workflow.connect(preprocessing_analysis, 'output_node.xml_diffeo',
                     compute_global__regression, 'input_node.xml_diffeo')
    workflow.connect(preprocessing_analysis, 'output_node.xml_object',
                     compute_global__regression, 'input_node.xml_object')
    workflow.connect(preprocessing_analysis, 'output_node.b0_ageNorm_file',
                     compute_global__regression, 'input_node.b0_ageNorm_file')

    workflow.connect(compute_global__regression, 'output_node.CP_file_global', output_node, 'CP_file_global')
    workflow.connect(compute_global__regression, 'output_node.out_t_from', output_node, 'out_t_from')
    workflow.connect(compute_global__regression, 'output_node.files_global_trajectory_source', output_node, 'files_global_trajectory_source')
    workflow.connect(compute_global__regression, 'output_node.xmlDiffeo_indiv', output_node, 'xmlDiffeo_indiv')
    workflow.connect(compute_global__regression, 'output_node.CP_files_indiv_traj', output_node, 'CP_files_indiv_traj')
    workflow.connect(compute_global__regression, 'output_node.MOM_files_indiv_traj', output_node, 'MOM_files_indiv_traj')
    workflow.connect(compute_global__regression, 'output_node.MOM_file_global',
                     output_node, 'MOM_file_global')
    workflow.connect(compute_global__regression, 'output_node.CP_global_traj_files_txt',
                     output_node, 'CP_global_traj_files_txt')
    workflow.connect(compute_global__regression, 'output_node.global_traj_files_vtk',
                     output_node, 'global_traj_files_vtk')

    # Create the workflow for the computation of the individual regressions and residual deformations transportation
    computation_regression = create_individual_regressions_get_shape_information()
    workflow.connect(compute_global__regression, 'output_node.CP_file_global',
                     computation_regression, 'input_node.CP_file_global')
    workflow.connect(compute_global__regression, 'output_node.MOM_file_global',
                     computation_regression, 'input_node.MOM_file_global')
    workflow.connect(compute_global__regression, 'output_node.global_traj_files_vtk',
                     computation_regression, 'input_node.global_traj_files_vtk')
    workflow.connect(preprocessing_analysis, 'output_node.xml_diffeo', computation_regression, 'input_node.xmlDiffeo_global')
    workflow.connect(compute_global__regression, 'output_node.xmlDiffeo_indiv', computation_regression, 'input_node.xmlDiffeo_indiv')
    workflow.connect(preprocessing_analysis, 'output_node.subjects_ids_unique', computation_regression, 'input_node.subject_ids_unique')
    workflow.connect(compute_global__regression, 'output_node.out_t_from', computation_regression, 'input_node.global_t_from')
    workflow.connect(compute_global__regression, 'output_node.files_global_trajectory_source', computation_regression, 'input_node.files_global_trajectory_source')
    workflow.connect(preprocessing_analysis, 'output_node.indiv_meshes_vtk_file', computation_regression, 'input_node.input_indiv_meshes')
    workflow.connect(preprocessing_analysis, 'output_node.indiv_ageNorm_files', computation_regression, 'input_node.indiv_AgeToOnsetNorm_file')
    workflow.connect(compute_global__regression, 'output_node.CP_files_indiv_traj', computation_regression, 'input_node.CP_files_indiv_traj')
    workflow.connect(compute_global__regression, 'output_node.MOM_files_indiv_traj', computation_regression, 'input_node.MOM_files_indiv_traj')
    workflow.connect(preprocessing_analysis, 'output_node.xml_object', computation_regression, 'input_node.xmlObject_global')
    workflow.connect(preprocessing_analysis, 'output_node.centroid_b0_vtk_file', computation_regression, 'input_node.centroid_b0_vtk_file')
    workflow.connect(input_node, 'xml_dkw',
                     computation_regression, 'input_node.xml_dkw')
    workflow.connect(input_node, 'xml_dkt',
                     computation_regression, 'input_node.xml_dkt')
    workflow.connect(input_node, 'xml_dtp',
                     computation_regression, 'input_node.xml_dtp')
    workflow.connect(input_node, 'xml_dsk',
                     computation_regression, 'input_node.xml_dsk')
    workflow.connect(input_node, 'xml_dcps',
                     computation_regression, 'input_node.xml_dcps')
    workflow.connect(input_node, 'xml_dfcp',
                     computation_regression, 'input_node.xml_dfcp')
    workflow.connect(input_node, 'xml_dmi',
                     computation_regression, 'input_node.xml_dmi')
    workflow.connect(input_node, 'xml_dat',
                     computation_regression, 'input_node.xml_dat')
    workflow.connect(input_node, 'xml_dls',
                     computation_regression, 'input_node.xml_dls')

    workflow.connect(computation_regression, 'output_node.transported_res_mom',
                     output_node, 'transported_res_mom')
    workflow.connect(computation_regression, 'output_node.transported_res_vect',
                     output_node, 'transported_res_vect')

    return workflow



def create_longitudinal_analysis_epilepsy(labels,
                                         nb_followup=2,
                                         reduction_rate=0.3,
                                         rigid_iteration=2,
                                         affine_iteration=3,
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
                'flip_id',
                'no_flip_id',
                'flip_seg',
                'no_flip_seg',
                'gap',
                'time',
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
        fields=['extracted_meshes', 'xml_diffeo_global', 'xml_object_global', 'xmlDiffeo_indiv',
                'b0_ageNorm_file', 'centroid_b0_vtk_file', 'CP_file_global', 'MOM_file_global',
                'global_traj_files_vtk', 'transported_res_mom',
                'transported_res_vect', 'struct_mat']),
        name='output_node')

    # Extract the sublist of parcelation and T1.
    split_list_to_flip_images = pe.Node(interface=Function(function=split_list2,
                                                    input_names=['including_id', 'all_id', 'list_data'],
                                                    output_names='extracted_list'),
                                 name='split_list_to_flip_images')
    workflow.connect(input_node, 'flip_id', split_list_to_flip_images, 'including_id')
    workflow.connect(input_node, 'subject_ids', split_list_to_flip_images, 'all_id')
    workflow.connect(input_node, 'input_images', split_list_to_flip_images, 'list_data')

    split_list_to_not_flip_images = pe.Node(interface=Function(function=split_list2,
                                                    input_names=['including_id', 'all_id', 'list_data'],
                                                    output_names='extracted_list'),
                                 name='split_list_to_not_flip_images')
    workflow.connect(input_node, 'no_flip_id', split_list_to_not_flip_images, 'including_id')
    workflow.connect(input_node, 'subject_ids', split_list_to_not_flip_images, 'all_id')
    workflow.connect(input_node, 'input_images', split_list_to_not_flip_images, 'list_data')

    split_list_to_flip_seg = pe.Node(interface=Function(function=split_list2,
                                                    input_names=['including_id', 'all_id', 'list_data'],
                                                    output_names='extracted_list'),
                                 name='split_list_to_flip_seg')
    workflow.connect(input_node, 'flip_id', split_list_to_flip_seg, 'including_id')
    workflow.connect(input_node, 'subject_ids', split_list_to_flip_seg, 'all_id')
    workflow.connect(input_node, 'flip_seg', split_list_to_flip_seg, 'list_data')

    split_list_to_not_flip_seg = pe.Node(interface=Function(function=split_list2,
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
    # Create the workflow to generate the required data for the regression
    # Done for only one label. Should be doable for a set a label, we would analyse together.
    # TO BE CHANGED !

    splitBaselineFollowup = pe.Node(interface=longitudinal_splitBaselineFollowup(),
                                    name='splitBaselineFollowup')
    #workflow.connect(convertVTK2txt, 'out_verticesFiles', splitBaselineFollowup, 'in_all_verticesFile')
    workflow.connect(meshes_workflow, 'output_node.output_meshes', splitBaselineFollowup, 'in_all_meshes')
    workflow.connect(input_node, 'time', splitBaselineFollowup, 'in_all_ages')
    workflow.connect(input_node, 'subject_ids', splitBaselineFollowup, 'in_all_subj_ids')
    splitBaselineFollowup.inputs.number_followup = nb_followup

    workflow.connect(splitBaselineFollowup, 'b0_meshes', output_node, 'b0_meshes_vtk_files')
    workflow.connect(splitBaselineFollowup, 'subject_ids_unique', output_node, 'subjects_ids_unique')
    workflow.connect(splitBaselineFollowup, 'indiv_meshes', output_node, 'indiv_meshes_vtk_file')
    workflow.connect(splitBaselineFollowup, 'indiv_age2onset_norm_file', output_node, 'indiv_ageNorm_files')
    workflow.connect(splitBaselineFollowup, 'b0_age2onset_norm_file', output_node, 'b0_ageNorm_file')

    compute_common_baseline_shape = centroid_computation(labels,
                                                            False,
                                                            param_sigmaV=8,
                                                            param_sigmaW=[10, 7, 4],
                                                            param_maxiters=[100, 200, 200],
                                                            param_T=10,
                                                            name='compute_common_baseline_shape'
                                                            )
    workflow.connect(splitBaselineFollowup, 'b0_ages', compute_common_baseline_shape, 'input_node.ages')
    workflow.connect(splitBaselineFollowup, 'b0_meshes',
                     compute_common_baseline_shape, 'input_node.input_vtk_meshes')
    workflow.connect(splitBaselineFollowup, 'subject_ids_unique', compute_common_baseline_shape, 'input_node.subject_ids')

    writeXmlParametersFiles = pe.Node(interface=WriteXMLFiles(),
                                      name='writeXmlParametersFiles')
    workflow.connect(input_node, 'xml_dkw', writeXmlParametersFiles, 'dkw')
    workflow.connect(input_node, 'xml_dkt', writeXmlParametersFiles, 'dkt')
    workflow.connect(input_node, 'xml_dtp', writeXmlParametersFiles, 'dtp')
    workflow.connect(input_node, 'xml_dsk', writeXmlParametersFiles, 'dsk')
    workflow.connect(input_node, 'xml_dcps', writeXmlParametersFiles, 'dcps')
    workflow.connect(input_node, 'xml_dcpp', writeXmlParametersFiles, 'dcpp')
    workflow.connect(input_node, 'xml_dfcp', writeXmlParametersFiles, 'dfcp')
    workflow.connect(input_node, 'xml_dmi', writeXmlParametersFiles, 'dmi')
    workflow.connect(input_node, 'xml_dat', writeXmlParametersFiles, 'dat')
    workflow.connect(input_node, 'xml_dls', writeXmlParametersFiles, 'dls')

    workflow.connect(input_node, 'xml_ods', writeXmlParametersFiles, 'ods')
    workflow.connect(input_node, 'xml_okw', writeXmlParametersFiles, 'okw')
    workflow.connect(input_node, 'xml_ot', writeXmlParametersFiles, 'ot')

    compute_deformation_bary2baselines = pe.MapNode(interface=SparseMatching3(),
                                                               iterfield=['in_targets'],
                                                               name='compute_deformation_bary2baselines')
    workflow.connect(compute_common_baseline_shape, 'output_node.out_centroid_vtk_file',
                     compute_deformation_bary2baselines, 'in_sources')
    workflow.connect(splitBaselineFollowup, 'b0_meshes',
                     compute_deformation_bary2baselines, 'in_targets')
    workflow.connect(writeXmlParametersFiles, 'out_xmlObject',
                     compute_deformation_bary2baselines, 'in_paramObjects')
    workflow.connect(writeXmlParametersFiles, 'out_xmlDiffeo',
                     compute_deformation_bary2baselines, 'in_paramDiffeo')

    workflow.connect(compute_deformation_bary2baselines, 'out_file_CP',
                     output_node, 'CP_files_indiv_traj')
    workflow.connect(compute_deformation_bary2baselines, 'out_file_MOM',
                     output_node, 'MOM_files_indiv_traj')

    # Shoot for computing the new meshes subject baselines.
    shooting_indiv_residual = pe.MapNode(interface=ShootAndFlow3(),
                                         iterfield=['in_cp_file', 'in_mom_file', 'in_sources', 'in_paramDiffeo'],
                                         name='shooting_indiv_residual')
    shooting_indiv_residual.inputs.in_direction = 1
    workflow.connect(compute_deformation_bary2baselines, 'out_file_CP', shooting_indiv_residual, 'in_cp_file')
    workflow.connect(compute_deformation_bary2baselines, 'out_file_MOM', shooting_indiv_residual, 'in_mom_file')
    workflow.connect(writeXmlParametersFiles, 'out_xmlDiffeo', shooting_indiv_residual, 'in_paramDiffeo')
    workflow.connect(writeXmlParametersFiles, 'out_xmlObject', shooting_indiv_residual, 'in_paramObjects')
    workflow.connect(compute_common_baseline_shape, 'output_node.out_centroid_vtk_file', shooting_indiv_residual, 'in_sources')


    # Compute the individual trajectories of the subjects.
    compute_indiv_st_regression = pe.MapNode(interface=SparseGeodesicRegression3(),
                                             iterfield=['in_subjects', 'in_time', 'in_initTemplates'],
                                  name='compute_indiv_st_regression')
    workflow.connect(splitBaselineFollowup, 'indiv_meshes', compute_indiv_st_regression, 'in_subjects')
    workflow.connect(splitBaselineFollowup, 'indiv_age2onset_norm_file', compute_indiv_st_regression, 'in_time')
    workflow.connect(writeXmlParametersFiles, 'out_xmlDiffeo', compute_indiv_st_regression, 'in_paramDiffeo')
    workflow.connect(writeXmlParametersFiles, 'out_xmlObject', compute_indiv_st_regression, 'in_paramObjects')
    workflow.connect(shooting_indiv_residual, 'out_last_vtk', compute_indiv_st_regression, 'in_initTemplates')  # the reordered ones ?

    # Parallel transport the individual vectors to their corresponding time point on the global trajectory.
    parallel_transport_to_indiv_traj = pe.MapNode(interface=ParallelTransport(),
                                                  iterfield=['in_vect', 'in_vtk', 'in_mom', 'in_cp',
                                                             'in_paramDiffeo'],
                                                  name='parallel_transport_to_indiv_traj')
    parallel_transport_to_indiv_traj.inputs.in_boolMom = 1
    parallel_transport_to_indiv_traj.inputs.in_t_to = 0
    workflow.connect(input_node, 'xml_dtp', parallel_transport_to_indiv_traj, 'in_t_from')
    workflow.connect(compute_deformation_bary2baselines, 'out_file_MOM', parallel_transport_to_indiv_traj, 'in_mom')
    workflow.connect(compute_deformation_bary2baselines, 'out_file_CP', parallel_transport_to_indiv_traj, 'in_cp')
    workflow.connect(writeXmlParametersFiles, 'out_xmlDiffeo', parallel_transport_to_indiv_traj, 'in_paramDiffeo') # need the new xml files
    workflow.connect(writeXmlParametersFiles, 'out_xmlObject', parallel_transport_to_indiv_traj, 'in_paramObjects')
    workflow.connect(compute_indiv_st_regression, 'out_file_MOM', parallel_transport_to_indiv_traj, 'in_vect')
    workflow.connect(compute_common_baseline_shape, 'output_node.out_centroid_vtk_file',
                     parallel_transport_to_indiv_traj, 'in_vtk') # the reordered ones ?

    # Create a rename for the transported momentum
    renamer1 = pe.MapNode(interface=niu.Rename(format_string="suj_%(subject_id)s_Mom_final_TRANSPORT_Mom_0",
                                               keep_ext=True),
                          iterfield=['subject_id', 'in_file'],
                          name='renamer1')
    workflow.connect(splitBaselineFollowup, 'subject_ids_unique', renamer1, 'subject_id')
    workflow.connect(parallel_transport_to_indiv_traj, 'out_transported_mom', renamer1, 'in_file')

    # Create a rename for the transported vector
    renamer2 = pe.MapNode(interface=niu.Rename(format_string="suj_%(subject_id)s_Mom_final_TRANSPORT_0", keep_ext=True),
                          iterfield=['subject_id', 'in_file'],
                          name='renamer2')
    workflow.connect(splitBaselineFollowup, 'subject_ids_unique', renamer2, 'subject_id')
    workflow.connect(parallel_transport_to_indiv_traj, 'out_transported_vect', renamer2, 'in_file')

    workflow.connect(renamer1, 'out_file', output_node, 'transported_res_mom')
    workflow.connect(renamer2, 'out_file', output_node, 'transported_res_vect')

    return workflow

