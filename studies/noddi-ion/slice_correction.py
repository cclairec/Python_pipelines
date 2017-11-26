# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function
import os
import nipype.interfaces.fsl as fsl
import inspect
from niftypipe.workflows.dmri.niftyfit_tensor_preprocessing import (get_B0s_from_bvals_bvecs, get_DWIs_from_bvals_bvecs)
from .steps_spinal_cord_segmentation import create_spinal_cord_segmentation_workflow


def gen_substitutions_slice(op_basename):
    subs = [('average_output_thresh', op_basename + '_average_b0'),
            (r'vol.*maths_res_merged_thresh', op_basename + '_corrected_dwi')]

    return subs


def find_slice_files(input_directory):
    import os
    import glob

    dwis = glob.glob(os.path.join(input_directory, '*_corrected_dwi.nii.gz'))
    average_b0 = glob.glob(os.path.join(input_directory, '*_average_b0.nii.gz'))
    return dwis[0], average_b0[0]


def sort_slices(b0_files, dwi_files):
    out_b0files = map(list, zip(*b0_files))
    out_dwifiles = map(list, zip(*dwi_files))
    return out_b0files, out_dwifiles


def create_slice_wise_dwi_motion_correction(name='slice_wise_motion_correction',
                                            output_dir='slice_wise_motion_correction',
                                            dwi_interp_type='CUB',
                                            computeMask=True,
                                            precomputed_mask=''):
    """
     Creates a workflow for correcting in a slice wise mode the DWI images.
     It is useful for spinal cord dwi imaging

    """

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = output_dir

    # We need to create an input node for the workflow
    input_node = pe.Node(
        niu.IdentityInterface(
            fields=['in_dwi_4d_file',
                    'in_bvec_file',
                    'in_bval_file'], mandatory_inputs=False),
        name='input_node')

    # For spinal cord we do our own eddy current/motion correction
    # Node using fslsplit() to split the 4D file, separate the B0 and DWIs
    split_dwis = pe.Node(interface=fsl.Split(dimension="t"), name='split_dwis')
    workflow.connect(input_node, 'in_dwi_4d_file', split_dwis, 'in_file')

    # Node using niu.Select() to select only the B0 files
    function_find_B0s = niu.Function(input_names=['bvals', 'bvecs', 'b0_threshold'], output_names=['out'])
    function_find_B0s.inputs.function_str = str(inspect.getsource(get_B0s_from_bvals_bvecs))
    find_B0s = pe.Node(interface=function_find_B0s, name='find_B0s')
    find_B0s.inputs.b0_threshold = 20.0
    select_B0s = pe.Node(interface=niu.Select(), name='select_B0s')

    workflow.connect(input_node, 'in_bval_file', find_B0s, 'bvals')
    workflow.connect(input_node, 'in_bvec_file', find_B0s, 'bvecs')
    workflow.connect(find_B0s, 'out', select_B0s, 'index')
    workflow.connect(split_dwis, 'out_files', select_B0s, 'inlist')

    # Node using niu.Select() to select only the DWIs files
    function_find_DWIs = niu.Function(input_names=['bvals', 'bvecs', 'b0_threshold'], output_names=['out'])
    function_find_DWIs.inputs.function_str = str(inspect.getsource(get_DWIs_from_bvals_bvecs))
    find_DWIs = pe.Node(interface=function_find_DWIs, name='find_DWIs')
    find_DWIs.inputs.b0_threshold = 20.0
    select_DWIs = pe.Node(interface=niu.Select(), name='select_DWIs')

    workflow.connect(input_node, 'in_bval_file', find_DWIs, 'bvals')
    workflow.connect(input_node, 'in_bvec_file', find_DWIs, 'bvecs')
    workflow.connect(find_DWIs, 'out', select_DWIs, 'index')
    workflow.connect(split_dwis, 'out_files', select_DWIs, 'inlist')

    # For spinal cord we do our own eddy current/motion correction
    # Node using fslsplit() to split the 4D file, separate the B0 and DWIs
    split_slices_b0 = pe.MapNode(interface=fsl.Split(dimension="z"), name='split_slices_b0', iterfield=['in_file'])
    split_slices_dwi = pe.MapNode(interface=fsl.Split(dimension="z"), name='split_slices_dwi', iterfield=['in_file'])
    workflow.connect(select_B0s, 'out', split_slices_b0, 'in_file')
    workflow.connect(select_DWIs, 'out', split_slices_dwi, 'in_file')

    # We resort the files to fit in a slice way
    sorting_slices = pe.Node(name='sort_slices',
                             interface=Function(input_names=['b0_files', 'dwi_files'],
                                                output_names=['out_b0files', 'out_dwifiles'],
                                                function=sort_slices))

    workflow.connect(split_slices_b0, 'out_files', sorting_slices, 'b0_files')
    workflow.connect(split_slices_dwi, 'out_files', sorting_slices, 'dwi_files')

    # We apply the per slice correction
    slice_correction = pe.Node(name='slice_correction',
                               interface=Function(
                                   input_names=['b0_files_list', 'dwi_files_list', 'result_dir', 'bvals', 'bvecs',
                                                'dwi_interp_type', 'index_b0'],
                                   output_names=['output_directories'],
                                   function=correction_slice_wise))
    slice_correction.inputs.result_dir = workflow.base_output_dir
    slice_correction.inputs.dwi_interp_type = dwi_interp_type
    workflow.connect(sorting_slices, 'out_b0files', slice_correction, 'b0_files_list')
    workflow.connect(sorting_slices, 'out_dwifiles', slice_correction, 'dwi_files_list')
    workflow.connect(input_node, 'in_bval_file', slice_correction, 'bvals')
    workflow.connect(input_node, 'in_bvec_file', slice_correction, 'bvecs')
    workflow.connect(find_B0s, 'out', slice_correction, 'index_b0')

    # We search the different results and we make the new list
    slice_files_finder = pe.MapNode(interface=niu.Function(input_names=['input_directory'],
                                                           output_names=['dwis', 'average_b0'],
                                                           function=find_slice_files),
                                    name='slice_files_finder',
                                    iterfield=['input_directory'])
    workflow.connect(slice_correction, 'output_directories', slice_files_finder, 'input_directory')

    # Remerge all the slices of the DWIs and B0s
    merge_dwis = pe.Node(interface=fsl.Merge(dimension='z'), name='merge_slices_dwis')
    merge_b0s = pe.Node(interface=fsl.Merge(dimension='z'), name='merge_slices_b0s')

    workflow.connect(slice_files_finder, 'dwis', merge_dwis, 'in_files')
    workflow.connect(slice_files_finder, 'average_b0', merge_b0s, 'in_files')
    sc_template = os.path.abspath(
        '/usr2/mrtools/pipelines/database/spinal_cord_segmentation/b0/spinal_cord_template_b0.nii.gz')
    sc_template_mask = os.path.abspath(
        '/usr2/mrtools/pipelines/database/spinal_cord_segmentation/b0/spinal_cord_template_b0_mask.nii.gz')
    if computeMask:
        sc_segmentation = create_spinal_cord_segmentation_workflow(name='spinal_cord_segmentation',
                                                                   spinal_cord_template=sc_template,
                                                                   spinal_cord_template_mask=sc_template_mask)
        sc_segmentation.connect(merge_b0s, 'merged_file', sc_segmentation.get_node('input_node'), 'in_file')

    # Output node
    output_node = pe.Node(interface=niu.IdentityInterface(
        fields=['dwis',
                'average_b0',
                'dwi_mask']),
        name="output_node")
    workflow.connect(merge_b0s, 'merged_file', output_node, 'average_b0')
    workflow.connect(merge_dwis, 'merged_file', output_node, 'dwis')
    if computeMask:
        workflow.connect(sc_segmentation, 'output_node.out_mask', output_node, 'dwi_mask')
        # workflow.connect(sc_segmentation,'output_node.out_wm_mask',output_node,'dwi_mask')
    else:
        output_node.inputs.dwi_mask = precomputed_mask

    return workflow


def correction_slice_wise(b0_files_list, dwi_files_list, result_dir, bvals, bvecs, dwi_interp_type, index_b0):
    from distutils import spawn
    import nipype.interfaces.niftyseg as niftyseg
    import nipype.interfaces.niftyreg as niftyreg
    import nipype.interfaces.io as nio
    import os
    from niftypipe.interfaces.niftk.base import (generate_graph, run_workflow)
    from niftypipe.workflows.groupwise.niftyreg_coregistration import create_atlas
    from niftypipe.workflows.dmri.niftyfit_tensor_preprocessing \
        import (reorder_list_from_bval_bvecs_slice_wise, gen_substitutions_slice)

    i = 0
    slice_correction_sinks = []
    for b0_files in b0_files_list:
        # We make the output directory
        slice_outputdir = os.path.join(result_dir, 'slice_' + str(i + 1))
        iter_wf = pe.Workflow(name='slice_' + str(i + 1))
        iter_wf.base_dir = result_dir
        iter_wf.base_output_dir = slice_outputdir

        # Perform rigid groupwise registration
        b0_correction = create_atlas(in_files=b0_files_list,
                                     output_dir=result_dir,
                                     ref_file=b0_files[0],
                                     name='groupwise_B0_coregistration',
                                     itr_rigid=1,
                                     itr_affine=0,
                                     itr_non_lin=0,
                                     linear_options_hash={'ln_val': 1, 'lp_val': 1, 'maxit_val': 5,
                                                          'verbosity_off_flag': True})

        # Threshold the B0 to 0 for avoid negative numbers with the logarithm
        # negative DWi values at sharp edges, which is not physically possible.
        tresh_val_b0 = pe.Node(interface=fsl.maths.Threshold(thresh=0.0, direction='below'),
                               name='tresh_b0')

        change_datatype = pe.MapNode(interface=niftyseg.BinaryMaths(operation='add',
                                                                    operand_value=0.0,
                                                                    output_datatype='float'),
                                     name='change_datatype',
                                     iterfield=['in_file'])

        # Resample the DWI and B0s
        resampling = pe.MapNode(niftyreg.RegResample(verbosity_off_flag=True), name='resampling',
                                iterfield=['trans_file', 'flo_file'])

        # Remerge all the DWIs
        merge_dwis = pe.Node(interface=fsl.Merge(dimension='t'), name='merge_dwis')

        # Threshold the DWIs to 0: if we use cubic or sync interpolation we may end up with
        # negative DWi values at sharp edges, which is not physically possible.
        threshold_dwis = pe.Node(interface=fsl.maths.Threshold(thresh=0.0, direction='below'),
                                 name='threshold_dwis')

        reorder_transformations = pe.Node(interface=niu.Function(
            input_names=['B0s', 'bvals', 'bvecs'],
            output_names=['out'],
            function=reorder_list_from_bval_bvecs_slice_wise),
            name='reorder_transformations')

        # We start calculating the mean b0 corrected image
        b0_correction.inputs.input_node.in_files = b0_files
        b0_correction.inputs.input_node.ref_file = b0_files[0]
        iter_wf.connect(b0_correction, 'output_node.average_image', tresh_val_b0, 'in_file')

        #############################################################
        # Reorder the B0 and DWIs transformations to match the bval #
        # WARNING: The aff_file from dwi_to_registragion is not     #
        # doing anything we resample each dwi image to the anterior #
        # b0                                                        #
        #############################################################
        iter_wf.connect(b0_correction, 'output_node.trans_files', reorder_transformations, 'B0s')
        reorder_transformations.inputs.bvals = bvals
        reorder_transformations.inputs.bvecs = bvecs

        file_list = []
        b0 = 0
        dwi = 0
        for each in range(len(b0_files + dwi_files_list[i])):
            if each in index_b0:
                file_list.append(b0_files[b0])
                b0 += 1
            else:
                file_list.append(dwi_files_list[i][dwi])
                dwi += 1

        #############################################################
        #   Resample the DWIs with affine                           #
        #   transformations and merge back into a 4D image          #
        #############################################################
        change_datatype.inputs.in_file = file_list
        resampling.inputs.inter_val = dwi_interp_type
        iter_wf.connect(b0_correction, 'output_node.average_image', resampling, 'ref_file')
        iter_wf.connect(change_datatype, 'out_file', resampling, 'flo_file')
        iter_wf.connect(reorder_transformations, 'out', resampling, 'trans_file')
        iter_wf.connect(resampling, 'res_file', merge_dwis, 'in_files')
        iter_wf.connect(merge_dwis, 'merged_file', threshold_dwis, 'in_file')

        ds = pe.Node(nio.DataSink(), name='ds')
        ds.inputs.base_directory = slice_outputdir
        ds.inputs.parameterization = False
        subsgen = pe.Node(interface=niu.Function(input_names=['op_basename'],
                                                 output_names=['substitutions'],
                                                 function=gen_substitutions_slice),
                          name='subsgen')
        subsgen.inputs.op_basename = 'slice_' + str(i + 1)
        iter_wf.connect(subsgen, 'substitutions', ds, 'regexp_substitutions')
        iter_wf.connect(tresh_val_b0, 'out_file', ds, '@b0')
        iter_wf.connect(threshold_dwis, 'out_file', ds, '@dwis')

        dot_exec = spawn.find_executable('dot')
        if dot_exec is not None:
            generate_graph(workflow=iter_wf)

        # Run the workflow
        qsubargs = '-l h_rt=02:00:00 -l tmem=2.9G -l h_vmem=2.9G -l vf=2.9G -l s_stack=10240 -j y -b y -S /bin/csh -V'
        run_workflow(workflow=iter_wf,
                     qsubargs=qsubargs)

        slice_correction_sinks.append(slice_outputdir)
        i += 1

    return slice_correction_sinks
