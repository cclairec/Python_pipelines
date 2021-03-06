#! /usr/bin/env python

import argparse
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import os
import sys

from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)
from niftypipe.workflows.shapeanalysis.shape_symmetrical_analysis import create_symmetric_spatio_temporal_analysis

print 'perform_spatiotemporal_shape_symmetric.py  -i /Users/clairec/DataAndResults/examples/longitudinal/images/imageS1_1.nii.gz /Users/clairec/DataAndResults/examples/longitudinal/images/imageS1_2.nii.gz /Users/clairec/DataAndResults/examples/longitudinal/images/imageS2_1.nii.gz /Users/clairec/DataAndResults/examples/longitudinal/images/imageS2_2.nii.gz /Users/clairec/DataAndResults/examples/longitudinal/images/imageS3_1.nii.gz /Users/clairec/DataAndResults/examples/longitudinal/images/imageS3_2.nii.gz /Users/clairec/DataAndResults/examples/longitudinal/images/imageS4_1.nii.gz /Users/clairec/DataAndResults/examples/longitudinal/images/imageS4_2.nii.gz -p /Users/clairec/DataAndResults/examples/longitudinal/parcellations/parcellationS1_1.nii.gz /Users/clairec/DataAndResults/examples/longitudinal/parcellations/parcellationS1_2.nii.gz /Users/clairec/DataAndResults/examples/longitudinal/parcellations/parcellationS2_1.nii.gz /Users/clairec/DataAndResults/examples/longitudinal/parcellations/parcellationS2_2.nii.gz /Users/clairec/DataAndResults/examples/longitudinal/parcellations/parcellationS3_1.nii.gz /Users/clairec/DataAndResults/examples/longitudinal/parcellations/parcellationS3_2.nii.gz /Users/clairec/DataAndResults/examples/longitudinal/parcellations/parcellationS4_1.nii.gz /Users/clairec/DataAndResults/examples/longitudinal/parcellations/parcellationS4_2.nii.gz -l 59 60 -s S1 S1 S2 S2 S3 S3 S4 S4 -a 0 1 1 2 2 3 3 4 -o test_symmetric_04Mars'
def main():
    help_message = "Help !"
    parser = argparse.ArgumentParser(description=help_message)
    # Input images
    parser.add_argument('-i', '--input_img',
                        dest='input_img',
                        metavar='image',
                        nargs='+',
                        help='Input structural image(s)')
    parser.add_argument('-p', '--input_par',
                        dest='input_par',
                        metavar='par',
                        help='Input parcellation image(s) from GIF',
                        nargs='+')
    parser.add_argument('-s', '--subject_ids',
                        dest='subject_ids',
                        metavar='list',
                        help='list of the subject Ids',
                        nargs='+')
    parser.add_argument('-a', '--ages',
                        type=float,
                        dest='ages',
                        metavar='list',
                        help='list of the subject ages',
                        nargs='+')

    # Other inputs
    parser.add_argument('-l', '--label_val',
                        dest='input_lab',
                        metavar='val',
                        type=int,
                        nargs='+',
                        help='Label index (indices) of the structure on interest. 2 values, one for Left then right')
    parser.add_argument('--rigid_it',
                        type=int,
                        dest='rigid_iteration',
                        metavar='number',
                        help='Number of iteration to perform for the rigid step (default is 3)',
                        default=3)
    parser.add_argument('--affine_it',
                        type=int,
                        dest='affine_iteration',
                        metavar='number',
                        help='Number of iteration to perform for the affine step (default is 3)',
                        default=3)
    parser.add_argument('-r', '--reduc_rate',
                        type=float,
                        dest='reduct_rate',
                        metavar='number',
                        help='decimation rate for the mesh extraction method. Reduction of r*100%.',
                        default=0.6)
    parser.add_argument('-xml_dkw',
                        type=int,
                        dest='xml_dkw',
                        metavar='number',
                        help='Diffeo Kernel width',
                        default=12)
    parser.add_argument('-xml_dkt',
                        dest='xml_dkt',
                        metavar='string',
                        help=' Diffeo Kernel type',
                        default="Exact")
    parser.add_argument('-xml_dtp',
                        type=int,
                        dest='xml_dtp',
                        metavar='number',
                        help='Diffeo: number of time points',
                        default=30)
    parser.add_argument('-xml_dsk',
                        type=float,
                        dest='xml_dsk',
                        metavar='number',
                        help='Diffeo: smoothing kernel width',
                        default=0.5)
    parser.add_argument('-xml_dcps',
                        type=int,
                        dest='xml_dcps',
                        metavar='number',
                        help='Diffeos: Initial spacing for Control Points',
                        default=12)
    parser.add_argument('-xml_dcpp',
                        dest='xml_dcpp',
                        metavar='number',
                        help="Diffeos: name of a file containing positions of control points. " +
                             "In case of conflict with initial-cp-spacing, if a file name is given in " +
                             "initial-cp-position and initial-cp-spacing is set, the latter is ignored and " +
                             "control point positions in the file name are used.",
                        default='x')
    parser.add_argument('-xml_dfcp',
                        dest='xml_dfcp',
                        metavar='On/Off',
                        help='Diffeos: Freeze the Control Points',
                        default="Off")
    parser.add_argument('-xml_dmi',
                        type=int,
                        dest='xml_dmi',
                        metavar='number',
                        help='Diffeos: Maximum of descent iterations',
                        default=100)
    parser.add_argument('-xml_dat',
                        type=float,
                        dest='xml_dat',
                        metavar='number',
                        help='Diffeos: adaptative tolerence for the gradient descent',
                        default=0.00005)
    parser.add_argument('-xml_dls',
                        type=int,
                        dest='xml_dls',
                        metavar='number',
                        help='Diffeos: Maximum line search iterations',
                        default=20)
    parser.add_argument('-xml_ods',
                        type=float,
                        nargs='+',
                        dest='xml_ods',
                        metavar='number',
                        help='Object: weight of the object in the fidelity-to-data term',
                        default=[0.5])
    parser.add_argument('-xml_okw',
                        type=int,
                        nargs='+',
                        dest='xml_okw',
                        metavar='number',
                        help='Object: Kernel width',
                        default=[7])
    parser.add_argument('-xml_ot',
                        nargs='+',
                        dest='xml_ot',
                        metavar='number',
                        help='Object type',
                        default=["NonOrientedSurfaceMesh"])

    # Output directory
    parser.add_argument('-o', '--output_dir',
                        dest='output_dir',
                        metavar='output_dir',
                        help='Output directory where to save the results',
                        required=True,
                        default='test_workflow')

    # Add default arguments in the parser
    default_parser_argument(parser)
    args = parser.parse_args()
    result_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    input_img = [os.path.abspath(f) for f in args.input_img]
    input_par = [os.path.abspath(f) for f in args.input_par]
    # creating a list of labels:
    nb_labels_to_merge = len(args.input_lab)/2
    labels = [args.input_lab[:nb_labels_to_merge]] * len(input_img)
    print labels
    labels_r = [args.input_lab[nb_labels_to_merge:]] * len(input_img)
    labels = labels+labels_r

    # Create the workflow
    workflow = create_symmetric_spatio_temporal_analysis(labels=labels,
                                                         reduction_rate=args.reduct_rate,
                                                         scan_number=len(input_img))
    workflow.base_dir = result_dir
    workflow.inputs.input_node.input_images = input_img
    workflow.inputs.input_node.input_parcellations = input_par
    workflow.inputs.input_node.input_ref = input_img[0]
    workflow.inputs.input_node.subject_ids = args.subject_ids
    workflow.inputs.input_node.ages = args.ages
    workflow.inputs.input_node.xml_dkw = args.xml_dkw
    workflow.inputs.input_node.xml_dkt = args.xml_dkt
    workflow.inputs.input_node.xml_dtp = args.xml_dtp
    workflow.inputs.input_node.xml_dsk = args.xml_dsk
    workflow.inputs.input_node.xml_dcps = args.xml_dcps
    workflow.inputs.input_node.xml_dcpp = args.xml_dcpp
    workflow.inputs.input_node.xml_dfcp = args.xml_dfcp
    workflow.inputs.input_node.xml_dmi = args.xml_dmi
    workflow.inputs.input_node.xml_dat = args.xml_dat
    workflow.inputs.input_node.xml_dls = args.xml_dls
    workflow.inputs.input_node.xml_ods = args.xml_ods
    workflow.inputs.input_node.xml_okw = args.xml_okw
    workflow.inputs.input_node.xml_ot = args.xml_ot

    # Edit the qsub arguments based on the input arguments
    qsubargs_time = '750:00:00'
    qsubargs_mem = '11G'
    if args.use_qsub is True and args.openmp_core > 1:
        qsubargs_mem = str(max(0.95, 1.9/args.openmp_core)) + 'G'

    qsubargs = '-l s_stack=10240 -j y -b y -S /bin/csh -V'
    qsubargs = qsubargs + ' -l h_rt=' + qsubargs_time
    qsubargs = qsubargs + ' -l tmem=' + qsubargs_mem + ' -l h_vmem=' + qsubargs_mem + ' -l vf=' + qsubargs_mem

    # Create a data sink
    ds = pe.Node(nio.DataSink(parameterization=False),
                 name='data_sink')

    workflow.connect([
        (workflow.get_node('output_node'), ds, [('extracted_meshes', '@extracted_meshes')]),
        (workflow.get_node('output_node'), ds, [('param_diffeo_file', '@param_diffeo_file')]),
        (workflow.get_node('output_node'), ds, [('param_object_file', '@param_object_file')]),
        (workflow.get_node('output_node'), ds, [('out_AgeToOnsetNorm_file', '@out_AgeToOnsetNorm_file')]),
        (workflow.get_node('output_node'), ds, [('out_init_shape_vtk_file', '@out_centroid_vtk_file')]),
        (workflow.get_node('output_node'), ds, [('out_sym_vtk_files', '@out_sym_vtk_files')]),
        (workflow.get_node('output_node'), ds, [('out_init_shape_vtk_file', '@out_init_shape_vtk_file')]),
        (workflow.get_node('output_node'), ds, [('transported_res_mom', '@transported_res_mom')]),
        (workflow.get_node('output_node'), ds, [('transported_res_vect', '@transported_res_vect')]),
        (workflow.get_node('output_node'), ds, [('out_file_CP', '@out_file_CP')]),
        (workflow.get_node('output_node'), ds, [('out_file_MOM', '@out_file_MOM')]),
    ])

    if args.graph is True:
        generate_graph(workflow=workflow)
        sys.exit(0)

    run_workflow(workflow=workflow,
                 qsubargs=qsubargs,
                 parser=args)


if __name__ == "__main__":
    main()
