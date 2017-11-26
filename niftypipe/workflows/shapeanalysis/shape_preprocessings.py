# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import csv
import os
import scipy.io
import nipype.interfaces.io as nio
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
import nipype.interfaces.niftyreg as niftyreg
import nipype.interfaces.niftyseg as niftyseg
from nipype.workflows.smri.niftyreg.groupwise import create_groupwise_average as create_atlas

from ...interfaces.niftk.utils import MergeLabels
from ...interfaces.niftk.io import Image2VtkMesh
from ...interfaces.shapeAnalysis import (VTKPolyDataReader, ComputeBarycentreBaseLine, VTKPolyDataWriter,
                                         WriteXMLFiles, SparseGeodesicRegression3, SparseMatching3, ShootAndFlow3,
                                         ParallelTransport)


# Create an affine groupwise space
def create_image_to_mesh_workflow(input_images,
                                  input_parcellations,
                                  input_label_id,
                                  result_dir,
                                  rigid_iteration=3,
                                  affine_iteration=3,
                                  reduction_rate=0.1,
                                  name='registrations_init'):
    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_dir = result_dir
    workflow.base_output_dir = name

    # Create a sub-workflow for groupwise registration
    groupwise = create_atlas(itr_rigid=rigid_iteration,
                             itr_affine=affine_iteration,
                             itr_non_lin=0,
                             name='groupwise')
    groupwise.inputs.input_node.in_files = input_images
    groupwise.inputs.input_node.ref_file = input_images[0]

    # Extract the relevant label from the GIF parcellation
    extract_label = pe.MapNode(interface=MergeLabels(),
                               iterfield=['in_file'], name='extract_label')
    extract_label.iterables = ("roi_list", [[l] for l in input_label_id])
    extract_label.inputs.in_file = input_parcellations

    # Removing parasite segmentation: Erosion.
    erode_binaries = pe.MapNode(interface=niftyseg.BinaryMathsInteger(operation='ero', operand_value=1),
                                iterfield=['in_file'], name='erode_binaries')
    workflow.connect(extract_label, 'out_file', erode_binaries, 'in_file')

    # Removing parasite segmentation: Dilatation.
    dilate_binaries = pe.MapNode(interface=niftyseg.BinaryMathsInteger(operation='dil', operand_value=1),
                                 iterfield=['in_file'], name='dilate_binaries')
    workflow.connect(erode_binaries, 'out_file', dilate_binaries, 'in_file')

    # Apply the relevant transformations to the roi
    apply_affine = pe.MapNode(interface=niftyreg.RegResample(inter_val='NN'),
                              iterfield=['flo_file',
                                         'trans_file'],
                              name='apply_affine')
    workflow.connect(groupwise, 'output_node.trans_files', apply_affine, 'trans_file')
    workflow.connect(groupwise, 'output_node.average_image', apply_affine, 'ref_file')
    workflow.connect(dilate_binaries, 'out_file', apply_affine, 'flo_file')

    # compute the large ROI that correspond to the union of all warped label
    extract_union_roi = pe.Node(interface=niftyreg.RegAverage(),
                                name='extract_union_roi')
    workflow.connect(apply_affine, 'res_file', extract_union_roi, 'in_files')

    # Binarise the average ROI
    binarise_roi = pe.Node(interface=niftyseg.UnaryMaths(operation='bin'),
                           name='binarise_roi')
    workflow.connect(extract_union_roi, 'out_file', binarise_roi, 'in_file')

    # Dilation of the binarise union ROI
    dilate_roi = pe.Node(interface=niftyseg.BinaryMathsInteger(operation='dil', operand_value=5),
                         name='dilate_roi')
    workflow.connect(binarise_roi, 'out_file', dilate_roi, 'in_file')

    # Apply the transformations
    apply_rigid_refinement = pe.MapNode(interface=niftyreg.RegAladin(rig_only_flag=True, ln_val=1),
                                        iterfield=['flo_file', 'in_aff_file'],
                                        name='apply_rigid_refinement')
    apply_rigid_refinement.inputs.flo_file = input_images
    workflow.connect(groupwise, 'output_node.average_image', apply_rigid_refinement, 'ref_file')
    workflow.connect(groupwise, 'output_node.trans_files', apply_rigid_refinement, 'in_aff_file')
    workflow.connect(dilate_roi, 'out_file', apply_rigid_refinement, 'rmask_file')

    # Extract the mesh corresponding to the label
    final_resampling = pe.MapNode(interface=niftyreg.RegResample(inter_val='NN'),
                                  iterfield=['flo_file',
                                             'trans_file'],
                                  name='final_resampling')
    workflow.connect(apply_rigid_refinement, 'aff_file', final_resampling, 'trans_file')
    workflow.connect(groupwise, 'output_node.average_image', final_resampling, 'ref_file')
    workflow.connect(dilate_binaries, 'out_file', final_resampling, 'flo_file')

    # Extract the mesh corresponding to the label
    extract_mesh = pe.MapNode(interface=Image2VtkMesh(in_reductionRate=reduction_rate),
                              iterfield=['in_file'],
                              name='extract_mesh')
    workflow.connect(final_resampling, 'res_file', extract_mesh, 'in_file')
    # workflow.connect(apply_rigid_refinement, 'aff_file', extract_mesh, 'matrix_file')

    # Create a rename for the average image
    groupwise_renamer = pe.Node(interface=niu.Rename(format_string='atlas', keep_ext=True),
                                name='groupwise_renamer')
    workflow.connect(groupwise, 'output_node.average_image', groupwise_renamer, 'in_file')

    # Create a datasink
    ds = pe.Node(nio.DataSink(parameterization=False), name='ds')
    ds.inputs.base_directory = result_dir
    workflow.connect(groupwise_renamer, 'out_file', ds, '@avg')
    workflow.connect(apply_rigid_refinement, 'res_file', ds, '@raf_mask')
    workflow.connect(extract_union_roi, 'out_file', ds, '@union_mask')
    workflow.connect(dilate_roi, 'out_file', ds, '@dilate_mask')
    workflow.connect(extract_mesh, 'out_file', ds, 'mesh_vtk')

    return workflow


#
def create_baseline_setup_param_workflow(result_dir,
                                         label,
                                         path2xmlfolder,
                                         csv_file='Empty',
                                         param_gammaR=1.e-4,
                                         param_sigmaV=13,
                                         param_sigmaW=[11, 8, 4, 2],
                                         param_maxiters=[100, 200, 200, 100],
                                         param_T=10,
                                         param_ntries=1,
                                         param_MPeps=0,
                                         name='create_baseline_and_setup'):
    # Create the input and output node
    input_node = pe.Node(niu.IdentityInterface(
        fields=['input_meshes', 'xml_dkw', 'xml_dkt', 'xml_dtp', 'xml_dsk', 'xml_dcps', 'xml_dcpp',
                'xml_dfcp', 'xml_dmi', 'xml_dat', 'xml_dls', 'xml_ods', 'xml_okw',
                'xml_ot', 'nbOfObjects'],
        mandatory_inputs=False),
        name='input_node')
    # output node:
    output_node = pe.Node(niu.IdentityInterface(
        fields=['out_xmlDiffeo', 'out_xmlObject', 'path2xmlfolder',
                'out_verticesFile', 'out_triangleFile', 'out_structFile',
                'out_vertices_centroid_file', 'out_centroid_file', 'out_ageToOnsetNorm_file',
                'out_centroid_vtk_file']),
        name='output_node')

    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_dir = result_dir
    workflow.base_output_dir = name

    # if input_meshes are given in parameters, then we compute the baseline, otherwise, the baseline computation is skipped.
    if input_node.input_meshes is not 'Empty':
        if label == 61:
            side = 'Left'

        if label == 60:
            side = 'Right'

        surf_aligned = [{'Vertices': 'x', 'Faces': 'x', 'SujID': k, 'Gene': 'x', 'FamNo': 'x', 'MutationCarrier': 'x',
                         'TIV': 'x', 'Filename': 'x', 'AgeToOnset': 'x', 'AgeToOnsetNorm': 'x',
                         'Age': 'x', 'Side': 'x', 'OnsetAgeFamily': 'x'}
                        for k in range(len(input_node.input_meshes))]

        # read the csv file to extract the information on each subjects
        f = open(csv_file)
        reader_f = csv.reader(f)
        ind = 0
        for row in reader_f:
            # find the index of the input_meshes containing the subj ID row[2]:
            k = 0
            while row[2] not in input_node.input_meshes[k]:
                k += 1
                if k > len(input_node.input_meshes) - 1:
                    k -= 1
                    break

            if row[2] in input_node.input_meshes[k]:
                print row[2] + " " + input_node.input_meshes[k]
                surf_aligned[ind]["Filename"] = input_node.input_meshes[k]
                surf_aligned[ind]["SujID"] = row[2]
                surf_aligned[ind]["Gene"] = row[3]
                surf_aligned[ind]["FamNo"] = row[5]
                surf_aligned[ind]["MutationCarrier"] = row[7]
                surf_aligned[ind]["TIV"] = row[19]
                surf_aligned[ind]["AgeToOnset"] = row[17]
                surf_aligned[ind]["AgeToOnsetNorm"] = -1  # to compute after, in the ComputeBarycenter node
                surf_aligned[ind]["Age"] = row[9]
                surf_aligned[ind]["OnsetAgeFamily"] = row[16]
                surf_aligned[ind]["Side"] = side

        saved_surf_aligned = 'SurfAligned' + label + '_' + side + '.mat'
        scipy.io.savemat(saved_surf_aligned, mdict={'surf_aligned': surf_aligned})

        # Connect the inputs to the lin_reg node, which is split over in_files

        convertVTK2txt = pe.MapNode(interface=VTKPolyDataReader,
                                    iterfield=['in_ind'],
                                    name='convertVTK2txt')
        convertVTK2txt.inputs.in_struct = saved_surf_aligned
        convertVTK2txt.inputs.in_ind = range(1, len(input_node.input_meshes) + 1)
        workflow.connect(input_node, 'input_meshes', convertVTK2txt, 'in_filename')

        workflow.connect(convertVTK2txt, 'out_verticesFile', output_node, 'out_verticesFile')
        workflow.connect(convertVTK2txt, 'out_triangleFile', output_node, 'out_triangleFile')
        workflow.connect(convertVTK2txt, 'out_structFile', output_node, 'out_structFile')

        computeBaseLine = pe.Node(interface=ComputeBarycentreBaseLine,
                                  name='computeBaseLine')
        computeBaseLine.inputs.in_param_gammaR = param_gammaR
        computeBaseLine.inputs.in_param_sigmaV = param_sigmaV
        computeBaseLine.inputs.in_param_sigmaW = param_sigmaW
        computeBaseLine.inputs.in_param_maxiters = param_maxiters
        computeBaseLine.inputs.in_param_T = param_T
        computeBaseLine.inputs.in_param_ntries = param_ntries
        computeBaseLine.inputs.in_param_MPeps = param_MPeps
        computeBaseLine.inputs.out_filename = saved_surf_aligned[:-4] + 'BaseLine.mat'
        computeBaseLine.inputs.out_ageToOnsetNorm_filename = computeBaseLine.inputs.out_filename[
                                                             :-4] + '_ageToOnsetNorm.txt'
        workflow.connect(convertVTK2txt, 'out_structFile', computeBaseLine, 'in_subjects')
        workflow.connect(computeBaseLine, 'out_vertices_file', output_node, 'out_vertices_centroid_file')
        workflow.connect(computeBaseLine, 'out_file', output_node, 'out_centroid_file')
        workflow.connect(computeBaseLine, 'out_ageToOnsetNorm_file', output_node, 'out_ageToOnsetNorm_file')

        convert_mat2VTK = pe.Node(interface=VTKPolyDataWriter,
                                  name='convert_mat2VTK')
        convert_mat2VTK.inputs.out_filename = os.path.abspath(computeBaseLine.outputs.out_file)[:-4] + '.vtk'
        workflow.connect(computeBaseLine, 'out_file', convert_mat2VTK, 'in_filename')
        workflow.connect(convert_mat2VTK, 'out_filename', output_node, 'out_centroid_vtk_file')

    writeXmlParametersFiles = pe.Node(interface=WriteXMLFiles,
                                      name='writeXmlParametersFiles')
    writeXmlParametersFiles.inputs.path = path2xmlfolder
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
    workflow.connect(writeXmlParametersFiles, 'out_xmlDiffeo', output_node, 'out_xmlDiffeo')
    workflow.connect(writeXmlParametersFiles, 'out_xmlObject', output_node, 'out_xmlObject')

    return workflow


def create_spatiotemporal_regression_residual_def_computation_pt(result_dir,
                                                              name='create_spatiotemporal_reg_resi_def_pt'):
    # Create the input and output node
    input_node = pe.Node(niu.IdentityInterface(
        fields=['input_meshes', 'nbOfObjects', 'in_AgeToOnsetNorm_file', 'in_xmlDiffeo', 'in_xmlObject',
                'out_verticesFile', 'baseline_vtk_file', 'path2xmlfolder', 'resid_def_dkw'],
        mandatory_inputs=False),
        name='input_node')
    # output node:
    output_node = pe.Node(niu.IdentityInterface(
        fields=['out_file_CP', 'out_file_MOM']),
        name='output_node')

    # Create the workflow
    workflow = pe.Workflow(name=name)
    workflow.base_dir = result_dir
    workflow.base_output_dir = name

    computeSTregression = pe.Node(interface=SparseGeodesicRegression3,
                                  name='computeSTregression')
    workflow.connect(input_node, 'input_meshes', computeSTregression, 'in_subjects')
    workflow.connect(input_node, 'nbOfObjects', computeSTregression, 'in_nbOfObjects')
    workflow.connect(input_node, 'in_AgeToOnsetNorm_file', computeSTregression, 'in_time')
    workflow.connect(input_node, 'in_xmlDiffeo', computeSTregression, 'in_paramDiffeo')
    workflow.connect(input_node, 'in_xmlObject', computeSTregression, 'in_paramObjects')
    workflow.connect(input_node, 'out_verticesFile', computeSTregression, 'in_initTemplates')

    shootingSTregression = pe.Node(interface=ShootAndFlow3,
                                   name='shootingSTregression')
    shootingSTregression.inputs.in_direction = 1
    workflow.connect(computeSTregression, 'out_file_CP', shootingSTregression, 'in_cp_file')
    workflow.connect(computeSTregression, 'out_file_MOM', shootingSTregression, 'in_mom_file')
    workflow.connect(input_node, 'in_xmlDiffeo', shootingSTregression, 'in_paramDiffeo')
    workflow.connect(input_node, 'in_xmlObject', shootingSTregression, 'in_paramObjects')
    workflow.connect(input_node, 'baseline_vtk_file', shootingSTregression, 'in_sources')

    # loop on the subject: create the list of targets and sources for the computation of the residual deformations:
    # To DO
    f = open(input_node.in_xmlDiffeo, 'r')
    line = f.readline()
    while '<number-of-timepoints>' not in line:
        line = f.readline()

    nb_tp = line[line.find('<number-of-timepoints>') + len('<number-of-timepoints>') - 1:
    line.find('</number-of-timepoints>')]

    subjects_id = []
    subject_filename_vtk = []
    ages_norm = []
    in_t_from = []
    for line in open(input_node.in_AgeToOnsetNorm_file):
        subjects_id.append(line.split()[0])
        subject_filename_vtk.append(line.split()[1])
        ages_norm.append(line.split()[2])

    time_points = range(0, nb_tp)
    files_trajectory_source = []
    files_subjects_target = []
    for t_ind in time_points:
        t = float(t_ind) / nb_tp
        for i in range(0, len(input_node.input_meshes)):
            aa = [abs(float(v) / nb_tp - ages_norm[i]) for v in time_points]
            closest_tp_ind = aa.index(min(aa))
            closest_tp = time_points[closest_tp_ind]
            if closest_tp == t:
                # Need to be generalised for different objects
                files_trajectory_source.append(shootingSTregression.outputs.out_files_vtk[t_ind])
                files_subjects_target.append(subject_filename_vtk[i])
                in_t_from.append(t_ind)

    write_xml_file4residual_def = pe.Node(interface=WriteXMLFiles,
                                          name='write_xml_file4residual_def')
    write_xml_file4residual_def.inputs.type = 'Def'
    write_xml_file4residual_def.inputs.path = input_node.inputs.path2xmlfolder
    write_xml_file4residual_def.inputs.dkw = input_node.inputs.resid_def_dkw
    write_xml_file4residual_def.inputs.dtp = 10
    write_xml_file4residual_def.inputs.xml_diffeo = 'paramDiffeo_residualDef'

    compute_residual_deformations = pe.MapNode(interface=SparseMatching3,
                                               iterfield=['in_sources', 'in_targets'],
                                               name='compute_residual_deformations')
    compute_residual_deformations.inputs.in_sources = files_trajectory_source
    compute_residual_deformations.inputs.in_targets = files_subjects_target
    workflow.connect(input_node, 'in_xmlObject', compute_residual_deformations, 'in_paramObjects')
    workflow.connect(write_xml_file4residual_def, 'out_xmlDiffeo', compute_residual_deformations, 'in_paramDiffeo')

    parallel_transport_res_def = pe.MapNode(interface=ParallelTransport,
                                            iterfield=['in_vect', 'in_t_from', 'in_vtk', 'out_file'])
    parallel_transport_res_def.inputs.in_vect = input_node.inputs.input_meshes
    parallel_transport_res_def.inputs.in_boolMom = 1
    parallel_transport_res_def.inputs.in_t_to = 0
    parallel_transport_res_def.inputs.in_t_from = in_t_from
    parallel_transport_res_def.inputs.out_file = input_node.inputs.input_meshes
    workflow.connect(compute_residual_deformations, 'out_file_MOM', parallel_transport_res_def, 'in_vect')
    workflow.connect(input_node, 'in_xmlDiffeo', parallel_transport_res_def, 'in_paramDiffeo')
    workflow.connect(input_node, 'in_xmlObject', parallel_transport_res_def, 'in_paramObjects')
    workflow.connect(computeSTregression, 'out_file_CP', parallel_transport_res_def, 'in_cp')
    workflow.connect(computeSTregression, 'out_file_MOM', parallel_transport_res_def, 'in_mom')
    return workflow
