#! /usr/bin/env python
import argparse
import sys
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from niftypipe.interfaces.niftk.base import run_workflow
from niftypipe.workflows.shapeanalysis import atlas_computation

workflow = pe.Workflow(name='test_wf_matlab_node')
workflow.base_dir = '/Users/clairec/DataAndResults/Nipype_tests/'
input_node = pe.Node(niu.IdentityInterface(fields=['input_vtk_meshes', 'subject_ids_2', 'subject_ids']),
                                           name='input_node')
input_node.inputs.subject_ids = [['S1', 'S1'], ['S1', 'S1'], ['S2', 'S2'], ['S2', 'S2'], ['S3', 'S3'], ['S3', 'S3'], ['S4', 'S4'], ['S4', 'S4']]
input_node.inputs.subject_ids_2 = ['S1', 'S2', 'S3', 'S4']
input_node.inputs.input_vtk_meshes = [['/Users/clairec/DataAndResults/Nipype_tests/test_symmetric_11Avril/spatio_temporal_analysis/gw_binary_to_meshes/extract_mesh/mapflow/_extract_mesh0/parcellationS1_1_merged_59_maths_maths_res_mesh.vtk', '/Users/clairec/DataAndResults/Nipype_tests/test_symmetric_11Avril/spatio_temporal_analysis/gw_binary_to_meshes/extract_mesh/mapflow/_extract_mesh8/parcellationS1_1_swapDim_axe_z_merged_60_maths_maths_res_mesh.vtk'], ['/Users/clairec/DataAndResults/Nipype_tests/test_symmetric_11Avril/spatio_temporal_analysis/gw_binary_to_meshes/extract_mesh/mapflow/_extract_mesh1/parcellationS1_2_merged_59_maths_maths_res_mesh.vtk', '/Users/clairec/DataAndResults/Nipype_tests/test_symmetric_11Avril/spatio_temporal_analysis/gw_binary_to_meshes/extract_mesh/mapflow/_extract_mesh9/parcellationS1_2_swapDim_axe_z_merged_60_maths_maths_res_mesh.vtk'], ['/Users/clairec/DataAndResults/Nipype_tests/test_symmetric_11Avril/spatio_temporal_analysis/gw_binary_to_meshes/extract_mesh/mapflow/_extract_mesh2/parcellationS2_1_merged_59_maths_maths_res_mesh.vtk', '/Users/clairec/DataAndResults/Nipype_tests/test_symmetric_11Avril/spatio_temporal_analysis/gw_binary_to_meshes/extract_mesh/mapflow/_extract_mesh10/parcellationS2_1_swapDim_axe_z_merged_60_maths_maths_res_mesh.vtk'], ['/Users/clairec/DataAndResults/Nipype_tests/test_symmetric_11Avril/spatio_temporal_analysis/gw_binary_to_meshes/extract_mesh/mapflow/_extract_mesh3/parcellationS2_2_merged_59_maths_maths_res_mesh.vtk', '/Users/clairec/DataAndResults/Nipype_tests/test_symmetric_11Avril/spatio_temporal_analysis/gw_binary_to_meshes/extract_mesh/mapflow/_extract_mesh11/parcellationS2_2_swapDim_axe_z_merged_60_maths_maths_res_mesh.vtk'], ['/Users/clairec/DataAndResults/Nipype_tests/test_symmetric_11Avril/spatio_temporal_analysis/gw_binary_to_meshes/extract_mesh/mapflow/_extract_mesh4/parcellationS3_1_merged_59_maths_maths_res_mesh.vtk', '/Users/clairec/DataAndResults/Nipype_tests/test_symmetric_11Avril/spatio_temporal_analysis/gw_binary_to_meshes/extract_mesh/mapflow/_extract_mesh12/parcellationS3_1_swapDim_axe_z_merged_60_maths_maths_res_mesh.vtk'], ['/Users/clairec/DataAndResults/Nipype_tests/test_symmetric_11Avril/spatio_temporal_analysis/gw_binary_to_meshes/extract_mesh/mapflow/_extract_mesh5/parcellationS3_2_merged_59_maths_maths_res_mesh.vtk', '/Users/clairec/DataAndResults/Nipype_tests/test_symmetric_11Avril/spatio_temporal_analysis/gw_binary_to_meshes/extract_mesh/mapflow/_extract_mesh13/parcellationS3_2_swapDim_axe_z_merged_60_maths_maths_res_mesh.vtk'], ['/Users/clairec/DataAndResults/Nipype_tests/test_symmetric_11Avril/spatio_temporal_analysis/gw_binary_to_meshes/extract_mesh/mapflow/_extract_mesh6/parcellationS4_1_merged_59_maths_maths_res_mesh.vtk', '/Users/clairec/DataAndResults/Nipype_tests/test_symmetric_11Avril/spatio_temporal_analysis/gw_binary_to_meshes/extract_mesh/mapflow/_extract_mesh14/parcellationS4_1_swapDim_axe_z_merged_60_maths_maths_res_mesh.vtk'], ['/Users/clairec/DataAndResults/Nipype_tests/test_symmetric_11Avril/spatio_temporal_analysis/gw_binary_to_meshes/extract_mesh/mapflow/_extract_mesh7/parcellationS4_2_merged_59_maths_maths_res_mesh.vtk', '/Users/clairec/DataAndResults/Nipype_tests/test_symmetric_11Avril/spatio_temporal_analysis/gw_binary_to_meshes/extract_mesh/mapflow/_extract_mesh15/parcellationS4_2_swapDim_axe_z_merged_60_maths_maths_res_mesh.vtk']]

node_wf = atlas_computation(map_node_use=True, name='wf_as_node')
workflow.connect(input_node, 'input_vtk_meshes', node_wf, 'input_node.input_vtk_meshes')
workflow.connect(input_node, 'subject_ids', node_wf, 'input_node.subject_ids')
workflow.connect(input_node, 'subject_ids_2', node_wf, 'input_node.subject_ids_2')

# convertVTK2txt = pe.Node(interface=VTKPolyDataReader(),
#                          name='convertVTK2txt')
# workflow.connect(input_node, 'in_struct', convertVTK2txt, 'in_struct')
# workflow.connect(input_node, 'in_filenames', convertVTK2txt, 'in_filenames')

import logging
logger = logging.getLogger('workflow')
logger.setLevel(logging.DEBUG)
run_workflow(workflow=workflow, qsubargs=None, parser=None)

print "the workflow finished to run"
