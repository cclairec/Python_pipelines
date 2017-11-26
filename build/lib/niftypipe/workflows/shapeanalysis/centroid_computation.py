import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu

from ...interfaces.shapeAnalysis import (VTKPolyDataReader, CreateStructureOfData,
                                        ComputeBarycentreBaseLine, VTKPolyDataWriter)
from ...interfaces.niftk.utils import RemoveFile


def centroid_computation(label,
                         map_node_use=False, # True when input_vtk_meshes is a list of list
                         param_gammaR=1.e-4,
                         param_sigmaV=13,
                         param_sigmaW=[11, 4],
                         param_maxiters=[100, 200],
                         param_T=10,
                         param_ntries=1,
                         param_MPeps=0,
                         suffix='BaseLine',
                         path_matlab='~/Code/Codes',
                         name='centroid_computation'
                         ):

    # Create the input node
    input_node = pe.Node(niu.IdentityInterface(
        fields=['input_vtk_meshes',
                'ages',
                'subject_ids',
                'subject_ids_2'  # When subject_ids is a list of list, subject_ids_2 is a list of strings
                ]),
        name='input_node')

    # Create the output node
    output_node = pe.Node(niu.IdentityInterface(
        fields=['out_centroid_vtk_file', 'out_verticesFiles', 'out_triangleFiles', 'out_verticesFile', 'out_triangleFile',
                'out_vertices_centroid_file', 'out_centroid_mat_file','out_AgeToOnsetNorm_file',
                'out_structFile']),
        name='output_node')

    w = pe.Workflow(name=name)

    if map_node_use:
        print "computing multiple centroids, for multiple population"
        # create a structure gathering all the filename with the ages vertices and subject IDs.
        structure_mapNode = pe.MapNode(interface=CreateStructureOfData(),
                                       iterfield=['input_meshes', 'ages', 'subject_ids'],
                                       name='1structure_mapNode')
        structure_mapNode.inputs.in_label = label
        w.connect(input_node, 'ages', structure_mapNode, 'ages')
        w.connect(input_node, 'input_vtk_meshes', structure_mapNode, 'input_meshes')
        w.connect(input_node, 'subject_ids', structure_mapNode, 'subject_ids')
        w.connect(structure_mapNode, 'out_ageToOnsetNorm_file', output_node, 'out_AgeToOnsetNorm_file')

        convertVTK2txt_mapNode = pe.MapNode(interface=VTKPolyDataReader(),
                                            iterfield=['in_struct', 'in_filenames'],
                                            name='2convertVTK2txt_mapNode')
        w.connect(structure_mapNode, 'out_file_mat', convertVTK2txt_mapNode, 'in_struct')
        w.connect(input_node, 'input_vtk_meshes', convertVTK2txt_mapNode, 'in_filenames')

        w.connect(convertVTK2txt_mapNode, 'out_verticesFiles', output_node, 'out_verticesFiles')
        w.connect(convertVTK2txt_mapNode, 'out_triangleFiles', output_node, 'out_triangleFiles')
        w.connect(convertVTK2txt_mapNode, 'out_structFile', output_node, 'out_structFile')

        computeMeanShape = pe.MapNode(interface=ComputeBarycentreBaseLine(),
                                      iterfield=['in_subjects'],
                                      name='3computeMeanShape')
        computeMeanShape.inputs.in_param_gammaR = param_gammaR
        computeMeanShape.inputs.in_param_sigmaV = param_sigmaV
        computeMeanShape.inputs.in_param_sigmaW = param_sigmaW
        computeMeanShape.inputs.in_param_maxiters = param_maxiters
        computeMeanShape.inputs.in_param_T = param_T
        computeMeanShape.inputs.in_param_ntries = param_ntries
        computeMeanShape.inputs.in_param_MPeps = param_MPeps
        computeMeanShape.inputs.out_suffix = suffix
        computeMeanShape.inputs.path_matlab = path_matlab
        w.connect(convertVTK2txt_mapNode, 'out_structFile', computeMeanShape, 'in_subjects')
        w.connect(computeMeanShape, 'out_vertices_file', output_node, 'out_vertices_centroid_file')
        w.connect(computeMeanShape, 'out_file', output_node, 'out_centroid_mat_file')

        free_space = pe.MapNode(interface=RemoveFile(),
                                iterfield=['in_file'],
                                name='4free_space')
        w.connect(structure_mapNode, 'out_file_mat', free_space, 'in_file')
        w.connect(computeMeanShape, 'out_file', free_space, 'wait_node')

        renamer_meanShapes = pe.MapNode(interface=niu.Rename(format_string="suj_%(subject_id)s_LR_mean_shape", keep_ext=True),
                                        iterfield=['subject_id', 'in_file'],
                                        name='5renamer_meanShapes')
        w.connect(input_node, 'subject_ids_2', renamer_meanShapes, 'subject_id')
        w.connect(computeMeanShape, 'out_file', renamer_meanShapes, 'in_file')

        convert_mat2VTK_mapNode = pe.MapNode(interface=VTKPolyDataWriter(),
                                             iterfield=['in_filename'],
                                             name='6convert_mat2VTK_mapNode')
        convert_mat2VTK_mapNode.inputs.nb_meshes = 1  # one centroid
        w.connect(renamer_meanShapes, 'out_file', convert_mat2VTK_mapNode, 'in_filename')
        w.connect(convert_mat2VTK_mapNode, 'out_filename', output_node, 'out_centroid_vtk_file')

    else:
        print "Computing one centroid for one population"
        # create a structure_mapNode gathering all the filename with the ages and subject IDs.
        structure = pe.Node(interface=CreateStructureOfData(),
                            name='1structure')
        structure.inputs.in_label = label
        w.connect(input_node, 'input_vtk_meshes', structure, 'input_meshes')
        w.connect(input_node, 'ages', structure, 'ages')
        w.connect(input_node, 'subject_ids', structure, 'subject_ids')
        w.connect(structure, 'out_ageToOnsetNorm_file', output_node, 'out_AgeToOnsetNorm_file')

        convertVTK2txt = pe.Node(interface=VTKPolyDataReader(),
                                 name='2convertVTK2txt')
        w.connect(structure, 'out_file_mat', convertVTK2txt, 'in_struct')
        w.connect(input_node, 'input_vtk_meshes', convertVTK2txt, 'in_filenames')

        w.connect(convertVTK2txt, 'out_verticesFiles', output_node, 'out_verticesFiles')
        w.connect(convertVTK2txt, 'out_triangleFiles', output_node, 'out_triangleFiles')
        w.connect(convertVTK2txt, 'out_structFile', output_node, 'out_structFile')

        computeBaseLine = pe.Node(interface=ComputeBarycentreBaseLine(),
                                  name='3computeBaseLine')
        computeBaseLine.inputs.in_param_gammaR = param_gammaR
        computeBaseLine.inputs.in_param_sigmaV = param_sigmaV
        computeBaseLine.inputs.in_param_sigmaW = param_sigmaW
        computeBaseLine.inputs.in_param_maxiters = param_maxiters
        computeBaseLine.inputs.in_param_T = param_T
        computeBaseLine.inputs.in_param_ntries = param_ntries
        computeBaseLine.inputs.in_param_MPeps = param_MPeps
        computeBaseLine.inputs.out_suffix = suffix
        w.connect(convertVTK2txt, 'out_structFile', computeBaseLine, 'in_subjects')
        w.connect(computeBaseLine, 'out_vertices_file', output_node, 'out_vertices_centroid_file')
        w.connect(computeBaseLine, 'out_file', output_node, 'out_centroid_mat_file')

        convert_mat2VTK = pe.Node(interface=VTKPolyDataWriter(),
                                  name='4convert_mat2VTK')
        convert_mat2VTK.inputs.nb_meshes = 1  # one centroid
        w.connect(computeBaseLine, 'out_file', convert_mat2VTK, 'in_filename')
        w.connect(convert_mat2VTK, 'out_filename', output_node, 'out_centroid_vtk_file')

    return w

