#!/home/claicury/Code/Install/niftypipe/bin/python

import argparse
import os
import sys
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
from niftypipe.workflows.shapeanalysis.shape_analysis import create_get_deformation_shape_analysis
from niftypipe.interfaces.niftk.base import (generate_graph,
                                             run_workflow,
                                             default_parser_argument)


def main():


    help_message = "perform used for shape analysis of hippocampi in epilepsia"
    parser = argparse.ArgumentParser(description=help_message)
    # Input images
    parser.add_argument('-i', '--input_img',
                        dest='input_img',
                        metavar='image',
                        nargs='+',
                        help='Input structural image(s)')
    parser.add_argument('-p', '--input_seg',
                        dest='seg',
                        metavar='par',
                        help='Input segmention image(s)',
                        nargs='+')
    parser.add_argument('-s', '--subject_ids',
                        dest='subject_ids',
                        metavar='list',
                        help='list of the subject Ids',
                        nargs='+')

    # Other inputs
    parser.add_argument('-l', '--label_val',
                        dest='input_lab',
                        metavar='val',
                        type=int,
                        nargs='+',
                        help='Label index (indices) to consider for the refinement')
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
                        help='decimation rate for the mesh extraction method',
                        default=0.05)
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
                        default=10)
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
                        default=[0.9])
    parser.add_argument('-xml_okw',
                        type=int,
                        nargs='+',
                        dest='xml_okw',
                        metavar='number',
                        help='Object: Kernel width',
                        default=[4])
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
    input_seg = [os.path.abspath(f) for f in args.seg]
    # Not relevent when flipping the controls:
    # if len(args.flip_id)+len(args.no_flip_id)!=len(args.subject_ids):
    #     raise IOError("to be flipped ID and other ID should be the same size as the whole ID list. " \
    #                   "len(args.flip_id) = "+str(len(args.flip_id))+ " " \
    #                   "len(args.no_flip_id) = "+str(len(args.no_flip_id)) +" " \
    #                   "len(args.subject_ids) = "+str(len(args.subject_ids)))
    labels=args.input_lab
    print labels
    # Create the workflow
    workflow = create_get_deformation_shape_analysis(labels=labels,
                                               reduction_rate=args.reduct_rate,
                                               dkw=args.xml_dkw,
                                              dkt=args.xml_dkt,
                                              okw=args.xml_okw,
                                              dtp=args.xml_dtp,
                                              dsk=args.xml_dsk,
                                              dcps=args.xml_dcps,
                                              dcpp=args.xml_dcpp,
                                              dfcp=args.xml_dfcp,
                                              dmi=args.xml_dmi,
                                              dat=args.xml_dat,
                                              dls=args.xml_dls,
                                              ods=args.xml_ods,
                                              ot=args.xml_ot)
    workflow.base_dir = result_dir
    workflow.inputs.input_node.input_images = input_img
    workflow.inputs.input_node.input_seg = input_seg
    workflow.inputs.input_node.input_ref = input_img[0]
    workflow.inputs.input_node.subject_ids = args.subject_ids

    # Edit the qsub arguments based on the input arguments
    qsubargs_time = '02:00:00'
    qsubargs_mem = '1.9G'
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
        (workflow.get_node('output_node'), ds, [('out_template_vtk_file', '@out_template_vtk_file')]),
        (workflow.get_node('output_node'), ds, [('out_template_CP_file', '@out_template_CP_file')]),
        (workflow.get_node('output_node'), ds, [('out_template_MOM_file', '@out_template_MOM_file')]),
        (workflow.get_node('output_node'), ds, [('out_template_vtk_files', '@out_template_vtk_files')]),
    ])

    if args.graph is True:
        generate_graph(workflow=workflow)
        sys.exit(0)

    run_workflow(workflow=workflow,
                 qsubargs=qsubargs,
                 parser=args)

if __name__ == "__main__":
    main()
