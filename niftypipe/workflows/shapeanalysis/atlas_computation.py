import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu

from ...interfaces.shapeAnalysis import (WriteXMLFiles, CreateStructureOfData,
                                         SparseAtlas3, VTKPolyDataWriter)
from nipype.interfaces.utility import Function


def atlas_computation(map_node_use=False,  # True when input_vtk_meshes is a list of list
                      dkw=10,
                      dkt='Exact',
                      okw=[8],
                      dtp=30,
                      dsk=0.5,
                      dcps=5,
                      dcpp='x',
                      dfcp='Off',
                      dmi=200,
                      dat=0.00005,
                      dls=20,
                      ods=[0.5],
                      ot=["NonOrientedSurfaceMesh"],
                      type_xml_file='All',
                      name='atlas_computation'
                      ):
    # Create the input node
    input_node = pe.Node(niu.IdentityInterface(
        fields=['input_vtk_meshes',
                'subject_ids',
                'subject_ids_2'  # needed when computing several atlas (like Left/right mean shape), so when subject_ids is a list of list, subject_ids_2 is a list of strings
                ]),
        name='input_node')

    # Create the output node
    output_node = pe.Node(niu.IdentityInterface(
        fields=['out_template_vtk_file']),
        name='output_node')

    w = pe.Workflow(name=name)

    # writing the corresponding xml files:
    xml_files = pe.Node(interface=WriteXMLFiles(), name='xml_obj')
    xml_files.inputs.dkw = dkw
    xml_files.inputs.dkt = dkt
    xml_files.inputs.okw = okw
    xml_files.inputs.dkw = dkw
    xml_files.inputs.dtp = dtp
    xml_files.inputs.dsk = dsk
    xml_files.inputs.dcps = dcps
    xml_files.inputs.dcpp = dcpp
    xml_files.inputs.dfcp = dfcp
    xml_files.inputs.dmi = dmi
    xml_files.inputs.dat = dat
    xml_files.inputs.dls = dls
    xml_files.inputs.ods = ods
    xml_files.inputs.ot = ot
    xml_files.inputs.type_xml_file = type_xml_file

    if map_node_use:
        # extract the first mesh of the list as to be the initial template of the atlas:

        init_template_extract = pe.MapNode(interface=Function(function=extract_1_from_list,
                                                              input_names=['in_list', 'ind'],
                                                              output_names='res'),
                                           iterfield='in_list',
                                           name='init_template_extract')
        init_template_extract.inputs.ind = 0
        w.connect(input_node, 'input_vtk_meshes', init_template_extract, 'in_list')
        # Compute the atlas of the subjects (in this case subjects are list of list, we then expect a list of atlas
        atlas_mapnode = pe.MapNode(interface=SparseAtlas3(),
                                   iterfield=['in_initTemplates', 'in_subjects'],
                                   name='atlas_mapnode')
        w.connect(input_node, 'input_vtk_meshes', atlas_mapnode, 'in_subjects')
        w.connect(init_template_extract, 'res', atlas_mapnode, 'in_initTemplates')
        w.connect(xml_files, 'out_xmlDiffeo', atlas_mapnode, 'in_paramDiffeo')
        w.connect(xml_files, 'out_xmlObject', atlas_mapnode, 'in_paramObjects')

        renamer_meanShapes = pe.MapNode(interface=niu.Rename(format_string="suj_%(subject_id)s_LR_mean_shape", keep_ext=True),
                                        iterfield=['subject_id', 'in_file'],
                                        name='renamer_meanShapes')
        w.connect(input_node, 'subject_ids_2', renamer_meanShapes, 'subject_id')
        w.connect(atlas_mapnode, 'out_file_vtk', renamer_meanShapes, 'in_file')
        w.connect(renamer_meanShapes, 'out_file', output_node, 'out_template_vtk_file')
    else:
        print "Computing one atlas for one population"
        # create a structure_mapNode gathering all the filename with the ages and subject IDs.

        init_template_extract = pe.Node(interface=Function(function=extract_1_from_list,
                                                           input_names=['in_list', 'ind'],
                                                           output_names='res'),
                                        name='init_template_extract')
        init_template_extract.inputs.ind = 0
        w.connect(input_node, 'input_vtk_meshes', init_template_extract, 'in_list')

        # Compute the atlas of the subjects (in this case subjects are list of list, we then expect a list of atlas
        atlas_node = pe.Node(interface=SparseAtlas3(),
                                   name='atlas_node')
        w.connect(input_node, 'input_vtk_meshes', atlas_node, 'in_subjects')
        w.connect(init_template_extract, 'res', atlas_node, 'in_initTemplates')
        w.connect(xml_files, 'out_xmlDiffeo', atlas_node, 'in_paramDiffeo')
        w.connect(xml_files, 'out_xmlObject', atlas_node, 'in_paramObjects')

        w.connect(atlas_node, 'out_file_vtk', output_node, 'out_template_vtk_file')
        w.connect(atlas_node, 'out_file_CP', output_node, 'out_template_CP_file')
        w.connect(atlas_node, 'out_file_MOM', output_node, 'out_template_MOM_file')
        w.connect(atlas_node, 'out_files_vtk', output_node, 'out_template_vtk_files')

    return w


def extract_1_from_list(in_list, ind):
    res = in_list[ind]
    print res
    return res
