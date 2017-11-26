# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:


import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
import nipype.interfaces.niftyreg as niftyreg
import nipype.interfaces.niftyseg as niftyseg
from nipype.workflows.smri.niftyreg.groupwise import create_groupwise_average as create_atlas

from .centroid_computation import centroid_computation

from ...interfaces.niftk.io import Image2VtkMesh
from ...interfaces.shapeAnalysis import (VTKPolyDataReader, decimateVTKfile, CreateStructureOfData, WriteXMLFiles,
                                         SparseGeodesicRegression3, SparseMatching3, sortingTimePoints,
                                         ShootAndFlow3, ParallelTransport)
from ...interfaces.niftk.utils import MergeLabels, extractSubList



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
        fields=['output_meshes']),
        name='output_node')

    # Extract the relevant label from the GIF parcellation
    extract_label = pe.MapNode(interface=MergeLabels(),
                               iterfield=['in_file', 'roi_list'],
                               name='extract_label')
    extract_label.inputs.roi_list = label
    workflow.connect(input_node, 'input_parcellations', extract_label, 'in_file')

    # Removing parasite segmentation: Erosion.
    erode_binaries = pe.MapNode(interface=niftyseg.BinaryMathsInteger(operation='ero', operand_value=2),
                                iterfield=['in_file'], name='erode_binaries')
    workflow.connect(extract_label, 'out_file', erode_binaries, 'in_file')

    # Removing parasite segmentation: Dilatation.
    dilate_binaries = pe.MapNode(interface=niftyseg.BinaryMathsInteger(operation='dil', operand_value=2),
                                 iterfield=['in_file'], name='dilate_binaries')
    workflow.connect(erode_binaries, 'out_file', dilate_binaries, 'in_file')

    # Apply the relevant transformations to the roi
    apply_affine = pe.MapNode(interface=niftyreg.RegResample(inter_val='NN'),
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
    dilate_roi = pe.Node(interface=niftyseg.BinaryMathsInteger(operation='dil', operand_value=6),
                         name='dilate_roi')
    workflow.connect(binarise_roi, 'out_file', dilate_roi, 'in_file')

    # Apply the transformations
    apply_rigid_refinement = pe.MapNode(interface=niftyreg.RegAladin(rig_only_flag=True, ln_val=1),
                                        iterfield=['flo_file', 'in_aff_file'],
                                        name='apply_rigid_refinement')
    workflow.connect(input_node, 'input_images', apply_rigid_refinement, 'flo_file')
    workflow.connect(input_node, 'ref_file', apply_rigid_refinement, 'ref_file')
    workflow.connect(input_node, 'trans_files', apply_rigid_refinement, 'in_aff_file')
    workflow.connect(dilate_roi, 'out_file', apply_rigid_refinement, 'rmask_file')

    # Extract the mesh corresponding to the label
    final_resampling = pe.MapNode(interface=niftyreg.RegResample(inter_val='NN'),
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
        fields=['out_xmlDiffeo', 'out_vertices_centroid_file', 'out_init_shape_vtk_file',
                'out_xmlObject', 'out_AgeToOnsetNorm_file', 'out_centroid_mat_file',
                'structure_data']),
        name='output_node')

    k = 10

    structure = pe.Node(interface=CreateStructureOfData(),
                        name='structure')
    structure.inputs.in_label = label
    workflow.connect(input_node, 'input_meshes', structure, 'input_meshes')
    workflow.connect(input_node, 'ages', structure, 'ages')
    workflow.connect(input_node, 'subject_ids', structure, 'subject_ids')
    workflow.connect(structure, 'out_ageToOnsetNorm_file', output_node, 'out_AgeToOnsetNorm_file')

    extract_youngest_subjects = pe.Node(interface=extractSubList(),
                                        name='extract_youngest_subjects')
    extract_youngest_subjects.inputs.k = k
    workflow.connect(input_node, 'ages', extract_youngest_subjects, 'sorting_reference')
    workflow.connect(input_node, 'input_meshes', extract_youngest_subjects, 'in_list')

    extract_youngest_subjID = pe.Node(interface=extractSubList(),
                                      name='extract_youngest_subjID')
    extract_youngest_subjID.inputs.k = k
    workflow.connect(input_node, 'ages', extract_youngest_subjID, 'sorting_reference')
    workflow.connect(input_node, 'subject_ids', extract_youngest_subjID, 'in_list')

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
    workflow.connect(input_node, 'ages', compute_initial_shape_regression, 'input_node.ages')
    workflow.connect(extract_youngest_subjects, 'out_sublist',
                     compute_initial_shape_regression, 'input_node.input_vtk_meshes')
    workflow.connect(extract_youngest_subjID, 'out_sublist', compute_initial_shape_regression, 'input_node.subject_ids')

    workflow.connect(compute_initial_shape_regression, 'output_node.out_centroid_vtk_file',
                     output_node, 'out_init_shape_vtk_file')

    # create the vtk file containing the Control Point positions, by decimating the baseline shape:
    decimate_init_shape = pe.Node(interface=decimateVTKfile(in_reductionRate=0.85),
                                  name='decimate_init_shape')
    workflow.connect(compute_initial_shape_regression, 'output_node.out_centroid_vtk_file',
                     decimate_init_shape, 'in_file')

    # Convert the vtk file of the CP points to the .txt file:
    convertVTK2txt_init_shape = pe.Node(interface=VTKPolyDataReader(),
                                        name='convertVTK2txt_init_shape')
    workflow.connect(decimate_init_shape, 'out_file', convertVTK2txt_init_shape, 'in_filename')

    writeXmlParametersFiles = pe.Node(interface=WriteXMLFiles(),
                                      name='writeXmlParametersFiles')
    workflow.connect(input_node, 'xml_dkw', writeXmlParametersFiles, 'dkw')
    workflow.connect(input_node, 'xml_dkt', writeXmlParametersFiles, 'dkt')
    workflow.connect(input_node, 'xml_dtp', writeXmlParametersFiles, 'dtp')
    workflow.connect(input_node, 'xml_dsk', writeXmlParametersFiles, 'dsk')
    workflow.connect(input_node, 'xml_dcps', writeXmlParametersFiles, 'dcps')
    workflow.connect(convertVTK2txt_init_shape, 'out_verticesFile', writeXmlParametersFiles, 'dcpp')
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

    return workflow


def create_get_shape_distance_from_regression(scan_number, name='get_shape_deformation_from_regression'):
    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    # Create the input node
    input_node = pe.Node(niu.IdentityInterface(
        fields=['input_meshes', 'in_AgeToOnsetNorm_file',
                'xmlDiffeo', 'xmlObject', 'baseline_vtk_file',
                ]),
        name='input_node')

    # Create the output node
    output_node = pe.Node(niu.IdentityInterface(
        fields=['param_diffeo_file', 'out_file_CP', 'out_file_MOM',
                'transported_res_mom', 'transported_res_vect',
                'param_object_file']),
        name='output_node')

    computeSTregression = pe.Node(interface=SparseGeodesicRegression3(),
                                  name='computeSTregression')
    workflow.connect(input_node, 'input_meshes', computeSTregression, 'in_subjects')
    workflow.connect(input_node, 'in_AgeToOnsetNorm_file', computeSTregression, 'in_time')
    workflow.connect(input_node, 'xmlDiffeo', computeSTregression, 'in_paramDiffeo')
    workflow.connect(input_node, 'xmlObject', computeSTregression, 'in_paramObjects')
    workflow.connect(input_node, 'baseline_vtk_file', computeSTregression, 'in_initTemplates')

    shootingSTregression = pe.Node(interface=ShootAndFlow3(),
                                   name='shootingSTregression')
    shootingSTregression.inputs.in_direction = 1
    workflow.connect(computeSTregression, 'out_file_CP', shootingSTregression, 'in_cp_file')
    workflow.connect(computeSTregression, 'out_file_MOM', shootingSTregression, 'in_mom_file')
    workflow.connect(input_node, 'xmlDiffeo', shootingSTregression, 'in_paramDiffeo')
    workflow.connect(input_node, 'xmlObject', shootingSTregression, 'in_paramObjects')
    workflow.connect(input_node, 'baseline_vtk_file', shootingSTregression, 'in_sources')

    # loop on the subject: create the list of targets and sources for the computation of the residual deformations:
    build_target_file_list4residual_comp = pe.Node(interface=sortingTimePoints(),
                                                   name='build_target_file_list4residual_comp')
    workflow.connect(shootingSTregression, 'out_files_vtk',
                     build_target_file_list4residual_comp,'in_timePoints_vtkfile')
    workflow.connect(input_node, 'xmlDiffeo', build_target_file_list4residual_comp, 'in_xmlDiffeo')
    workflow.connect(input_node, 'in_AgeToOnsetNorm_file',
                     build_target_file_list4residual_comp, 'in_AgeToOnsetNorm_file')
    workflow.connect(shootingSTregression, 'out_CP_files_txt', build_target_file_list4residual_comp, 'in_CP_txtfiles')

    compute_residual_deformations = pe.MapNode(interface=SparseMatching3(),
                                               iterfield=['in_sources', 'in_targets', 'in_paramDiffeo'],
                                               name='compute_residual_deformations')
    workflow.connect(build_target_file_list4residual_comp, 'files_trajectory_source',
                     compute_residual_deformations, 'in_sources')
    workflow.connect(input_node, 'input_meshes',
                     compute_residual_deformations, 'in_targets')
    workflow.connect(input_node, 'xmlObject', compute_residual_deformations, 'in_paramObjects')
    workflow.connect(build_target_file_list4residual_comp, 'files_xmlDiffeo',
                     compute_residual_deformations, 'in_paramDiffeo')

    parallel_transport_res_def = pe.MapNode(interface=ParallelTransport(),
                                            iterfield=['in_vect', 'in_vtk', 'in_t_from'],
                                            name='parallel_transport_res_def')
    parallel_transport_res_def.inputs.in_boolMom = 1
    parallel_transport_res_def.inputs.in_t_to = 0
    workflow.connect(computeSTregression, 'out_file_MOM', parallel_transport_res_def, 'in_mom')
    workflow.connect(computeSTregression, 'out_file_CP', parallel_transport_res_def,'in_cp')
    workflow.connect(input_node, 'xmlDiffeo', parallel_transport_res_def, 'in_paramDiffeo')
    workflow.connect(input_node, 'xmlObject', parallel_transport_res_def, 'in_paramObjects')
    workflow.connect(compute_residual_deformations, 'out_file_MOM', parallel_transport_res_def, 'in_vect')
    workflow.connect(build_target_file_list4residual_comp, 'out_t_from', parallel_transport_res_def, 'in_t_from')
    workflow.connect(build_target_file_list4residual_comp, 'files_trajectory_source',
                     parallel_transport_res_def, 'in_vtk')

    # Create a rename for the transported momentum
    renamer1 = pe.MapNode(interface=niu.Rename(format_string="suj_%(subject_id)s_Mom_final_TRANSPORT_Mom_0", keep_ext=True),
                          iterfield=['subject_id', 'in_file'],
                                name='renamer1')
    workflow.connect(build_target_file_list4residual_comp, 'subject_id', renamer1, 'subject_id')
    workflow.connect(parallel_transport_res_def, 'out_transported_mom', renamer1, 'in_file')

    # Create a rename for the transported vector
    renamer2 = pe.MapNode(interface=niu.Rename(format_string="suj_%(subject_id)s_Mom_final_TRANSPORT_0", keep_ext=True),
                          iterfield=['subject_id', 'in_file'],
                                name='renamer2')
    workflow.connect(build_target_file_list4residual_comp, 'subject_id', renamer2, 'subject_id')
    workflow.connect(parallel_transport_res_def, 'out_transported_vect', renamer2, 'in_file')

    workflow.connect(renamer1, 'out_file', output_node, 'transported_res_mom')
    workflow.connect(renamer2, 'out_file', output_node, 'transported_res_vect')
    workflow.connect(computeSTregression, 'out_file_CP', output_node, 'out_file_CP')
    workflow.connect(computeSTregression, 'out_file_MOM',output_node, 'out_file_MOM')
    return workflow


def create_spatio_temporal_analysis(labels,
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
        fields=['extracted_meshes',
                'param_diffeo_file', 'param_object_file', 'out_AgeToOnsetNorm_file', 'out_centroid_mat_file',
                'out_init_shape_vtk_file', 'out_vertices_centroid_file',
                'transported_res_mom', 'transported_res_vect', 'out_file_CP', 'out_file_MOM']),
        name='output_node')

    # Create a sub-workflow for groupwise registration
    groupwise = create_atlas(itr_rigid=rigid_iteration,
                             itr_affine=affine_iteration,
                             itr_non_lin=0,
                             name='groupwise')
    workflow.connect(input_node, 'input_images', groupwise, 'input_node.in_files')
    workflow.connect(input_node, 'input_ref', groupwise, 'input_node.ref_file')

    # Create the workflow to create the meshes in an average space
    meshes_workflow = create_binary_to_meshes(label=labels, reduction_rate=reduction_rate)
    workflow.connect(input_node, 'input_images', meshes_workflow, 'input_node.input_images')
    workflow.connect(input_node, 'input_parcellations', meshes_workflow, 'input_node.input_parcellations')
    workflow.connect(groupwise, 'output_node.trans_files', meshes_workflow, 'input_node.trans_files')
    workflow.connect(groupwise, 'output_node.average_image', meshes_workflow, 'input_node.ref_file')
    workflow.connect(meshes_workflow, 'output_node.output_meshes',
                     output_node, 'extracted_meshes')
    # Create the workflow to generate the required data for the regression
    # Done for only one label. Should be doable for a set a label, we would analyse together.

    preprocessing_regression = create_spatio_temporal_regression_preprocessing(label=labels,
                                                                               scan_number=scan_number,
                                                                               )
    workflow.connect(meshes_workflow, 'output_node.output_meshes',
                     preprocessing_regression, 'input_node.input_meshes')
    workflow.connect(input_node, 'ages',
                     preprocessing_regression, 'input_node.ages')
    workflow.connect(input_node, 'subject_ids',
                     preprocessing_regression, 'input_node.subject_ids')
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
    workflow.connect(preprocessing_regression, 'output_node.out_xmlDiffeo',
                     output_node, 'param_diffeo_file')
    workflow.connect(preprocessing_regression, 'output_node.out_xmlObject',
                     output_node, 'param_object_file')
    workflow.connect(preprocessing_regression, 'output_node.out_AgeToOnsetNorm_file',
                     output_node, 'out_AgeToOnsetNorm_file')
    workflow.connect(preprocessing_regression, 'output_node.out_centroid_mat_file',
                     output_node, 'out_centroid_mat_file')
    workflow.connect(preprocessing_regression, 'output_node.out_init_shape_vtk_file',
                     output_node, 'out_init_shape_vtk_file')
    workflow.connect(preprocessing_regression, 'output_node.out_vertices_centroid_file',
                     output_node, 'out_vertices_centroid_file')

    # Create the workflow for the computation of the regression and residual deformations transportation
    computation_regression = create_get_shape_distance_from_regression(scan_number=scan_number)
    workflow.connect(meshes_workflow, 'output_node.output_meshes',
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
