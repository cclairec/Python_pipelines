#! /usr/bin/env python

import dtilikelihood.base as dmri
import dtilikelihood.graphics as graphics
import argparse
import math
import os
import niftk
import nipype.pipeline.engine as pe



parser = argparse.ArgumentParser(description='Diffusion usage example')
parser.add_argument('-i', '--tensors',
                    dest='tensors',
                    metavar='tensors',
                    help='Tensor map used for likelihood simulations',
                    required=True)
parser.add_argument('-l', '--bvals',
                    dest='bvals',
                    metavar='bvals',
                    help='bval file to be associated associated with the tensor map',
                    required=True)
parser.add_argument('-c', '--bvecs',
                    dest='bvecs',
                    metavar='bvecs',
                    help='bvec file to be associated associated with the tensor map',
                    required=True)
parser.add_argument('-b', '--b0',
                    dest='b0',
                    metavar='b0',
                    help='b0 file to be associated associated with the tensor map',
                    required=True)
parser.add_argument('-t', '--t1',
                    dest='t1',
                    metavar='t1',
                    help='T1 file to be associated associated with the tensor map',
                    required=True)
parser.add_argument('-m', '--mask',
                    dest='mask',
                    metavar='mask',
                    help='mask file to be associated associated with the tensor map',
                    required=True)
parser.add_argument('-p', '--parcellation',
                    dest='parcellation',
                    metavar='parcellation',
                    help='parcellation file to be associated associated with the tensor map',
                    required=True)

args = parser.parse_args()

result_dir = os.getcwd()+'/results/'
# os.mkdir(result_dir)

inter_types = ['LIN', 'CUB']
log_data_values = [True, False]
number_of_repeats = 2
base_res = 'dti_likelihood_study_'
for i in range(number_of_repeats):
    for log in log_data_values:
        for inter in inter_types:
            pipeline_name =
            if log is True:
                pipeline_name += 'log_'
            pipeline_name = pipeline_name + inter + '_' + str(i)
            r = dmri.create_dti_likelihood_study_workflow(name=pipeline_name,
                                                          log_data=log,
                                                          dwi_interp_type=inter,
                                                          result_dir=result_dir)
            r.base_dir = os.getcwd()
            tofsl = pe.Node(interface=niftk.io.Dtitk2Fsl(),
                            name='tofsl')
            tofsl.inputs.in_file = os.path.abspath(args.tensors)
            r.connect(tofsl, 'out_file', r.get_node('input_node'), 'in_tensors_file')
            r.inputs.input_node.in_bvec_file = os.path.abspath(args.bvecs)
            r.inputs.input_node.in_bval_file = os.path.abspath(args.bvals)
            r.inputs.input_node.in_b0_file = os.path.abspath(args.b0)
            r.inputs.input_node.in_t1_file = os.path.abspath(args.t1)
            r.inputs.input_node.in_mask_file = os.path.abspath(args.mask)
            r.inputs.input_node.in_labels_file = os.path.abspath(args.parcellation)
            r.inputs.input_node.in_stddev_translation = 0.75
            r.inputs.input_node.in_stddev_rotation = 0.5*math.pi/180
            r.inputs.input_node.in_stddev_shear = 0.04
            # SNR of 15, based on the mean b0 in the JHU parcellation region
            r.inputs.input_node.in_noise_sigma = 30

            r.write_graph(graph2use='colored')
            r.run('MultiProc')

graphics.plot_results(result_base_directory=result_dir,
                      number_of_repeats=number_of_repeats,
                      result_directory_prefix=base_res)
