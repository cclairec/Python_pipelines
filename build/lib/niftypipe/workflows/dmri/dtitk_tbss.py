# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
import nipype.interfaces.fsl as fsl
import nipype.interfaces.niftyseg as niftyseg
import nipype.interfaces.niftyreg as niftyreg
import os
from ...interfaces.niftk import dtitk
from .dtitk_tensor_groupwise import create_dtitk_groupwise_workflow


def create_cross_sectional_tbss_pipeline(in_files,
                                         output_dir,
                                         name='cross_sectional_tbss',
                                         skeleton_threshold=0.2,
                                         design_mat=None,
                                         design_con=None):
    workflow = pe.Workflow(name=name)
    workflow.base_dir = output_dir
    workflow.base_output_dir = name

    # Create the dtitk groupwise registration workflow
    groupwise_dtitk = create_dtitk_groupwise_workflow(in_files=in_files,
                                                      name="dtitk_groupwise",
                                                      rig_iteration=3,
                                                      aff_iteration=3,
                                                      nrr_iteration=6)

    # Create the average FA map
    mean_fa = pe.Node(interface=dtitk.TVtool(),
                      name="mean_fa")
    workflow.connect(groupwise_dtitk, 'output_node.out_template', mean_fa, 'in_file')
    mean_fa.inputs.operation = 'fa'

    # Register the FMRIB58_FA_1mm.nii.gz atlas to the mean FA map
    reg_atlas = pe.Node(interface=niftyreg.RegAladin(),
                        name='reg_atlas')
    workflow.connect(mean_fa, 'out_file', reg_atlas, 'ref_file')
    reg_atlas.inputs.flo_file = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'FMRIB58_FA_1mm.nii.gz')

    # Apply the transformation to the lower cingulum image
    war_atlas = pe.Node(interface=niftyreg.RegResample(),
                        name='war_atlas')
    workflow.connect(mean_fa, 'out_file', war_atlas, 'ref_file')
    war_atlas.inputs.flo_file = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'LowerCingulum_1mm.nii.gz')
    workflow.connect(reg_atlas, 'aff_file', war_atlas, 'trans_file')
    war_atlas.inputs.inter_val = 'LIN'

    # Threshold the propagated lower cingulum
    thr_atlas = pe.Node(interface=niftyseg.BinaryMaths(),
                        name='thr_atlas')
    workflow.connect(war_atlas, 'out_file', thr_atlas, 'in_file')
    thr_atlas.inputs.operation = 'thr'
    thr_atlas.inputs.operand_value = 0.5

    # Binarise the propagated lower cingulum
    bin_atlas = pe.Node(interface=niftyseg.UnaryMaths(),
                        name='bin_atlas')
    workflow.connect(thr_atlas, 'out_file', bin_atlas, 'in_file')
    bin_atlas.inputs.operation = 'bin'

    # Create all the individual FA maps
    individual_fa = pe.MapNode(interface=dtitk.TVtool(),
                               name="individual_fa",
                               iterfield=['in_file'])
    workflow.connect(groupwise_dtitk, 'output_node.out_res', individual_fa, 'in_file')
    individual_fa.inputs.operation = 'fa'

    # Create all the individual MD maps
    individual_md = pe.MapNode(interface=dtitk.TVtool(),
                               name="individual_md",
                               iterfield=['in_file'])
    workflow.connect(groupwise_dtitk, 'output_node.out_res', individual_md, 'in_file')
    individual_md.inputs.operation = 'tr'

    # Create all the individual RD maps
    individual_rd = pe.MapNode(interface=dtitk.TVtool(),
                               name="individual_rd",
                               iterfield=['in_file'])
    workflow.connect(groupwise_dtitk, 'output_node.out_res', individual_rd, 'in_file')
    individual_rd.inputs.operation = 'rd'

    # Create all the individual RD maps
    individual_ad = pe.MapNode(interface=dtitk.TVtool(),
                               name="individual_ad",
                               iterfield=['in_file'])
    workflow.connect(groupwise_dtitk, 'output_node.out_res', individual_ad, 'in_file')
    individual_ad.inputs.operation = 'ad'

    # Combine all the warped FA images into a 4D image
    merged_4d_fa = pe.Node(interface=fsl.Merge(),
                           name='merged_4d_fa')
    merged_4d_fa.inputs.dimension = 't'
    workflow.connect(individual_fa, 'out_file', merged_4d_fa, 'in_files')

    # Combine all the warped MD images into a 4D image
    merged_4d_md = pe.Node(interface=fsl.Merge(),
                           name='merged_4d_md')
    merged_4d_md.inputs.dimension = 't'
    workflow.connect(individual_md, 'out_file', merged_4d_md, 'in_files')

    # Combine all the warped RD images into a 4D image
    merged_4d_rd = pe.Node(interface=fsl.Merge(),
                           name='merged_4d_rd')
    merged_4d_rd.inputs.dimension = 't'
    workflow.connect(individual_rd, 'out_file', merged_4d_rd, 'in_files')

    # Combine all the warped AD images into a 4D image
    merged_4d_ad = pe.Node(interface=fsl.Merge(),
                           name='merged_4d_ad')
    merged_4d_ad.inputs.dimension = 't'
    workflow.connect(individual_ad, 'out_file', merged_4d_ad, 'in_files')

    # Threshold the 4D FA image to 0
    merged_4d_fa_thresholded = pe.Node(interface=niftyseg.BinaryMaths(),
                                       name='merged_4d_fa_thresholded')
    merged_4d_fa_thresholded.inputs.operation = 'thr'
    merged_4d_fa_thresholded.inputs.operand_value = 0
    workflow.connect(merged_4d_fa, 'merged_file', merged_4d_fa_thresholded, 'in_file')

    # Extract the min value from the 4D FA image
    minimal_value_across_all_fa = pe.Node(interface=niftyseg.UnaryMaths(),
                                          name='minimal_value_across_all_fa')
    minimal_value_across_all_fa.inputs.operation = 'tmin'
    workflow.connect(merged_4d_fa_thresholded, 'out_file', minimal_value_across_all_fa, 'in_file')

    # Create the mask image
    fa_mask = pe.Node(interface=niftyseg.UnaryMaths(),
                      name='fa_mask')
    fa_mask.inputs.operation = 'bin'
    fa_mask.inputs.output_datatype = 'char'
    workflow.connect(minimal_value_across_all_fa, 'out_file', fa_mask, 'in_file')

    # Mask the mean FA image
    masked_mean_fa = pe.Node(interface=fsl.ApplyMask(),
                             name='masked_mean_fa')
    workflow.connect(mean_fa, 'out_file', masked_mean_fa, 'in_file')
    workflow.connect(fa_mask, 'out_file', masked_mean_fa, 'mask_file')

    # Create the skeleton image
    skeleton = pe.Node(interface=fsl.TractSkeleton(),
                       name='skeleton')
    skeleton.inputs.skeleton_file = True
    workflow.connect(masked_mean_fa, 'out_file', skeleton, 'in_file')

    # Threshold the skeleton image
    thresholded_skeleton = pe.Node(interface=niftyseg.BinaryMaths(),
                                   name='thresholded_skeleton')
    thresholded_skeleton.inputs.operation = 'thr'
    thresholded_skeleton.inputs.operand_value = skeleton_threshold
    workflow.connect(skeleton, 'skeleton_file', thresholded_skeleton, 'in_file')

    # Binarise the skeleton image
    binarised_skeleton = pe.Node(interface=niftyseg.UnaryMaths(),
                                 name='binarised_skeleton')
    binarised_skeleton.inputs.operation = 'bin'
    workflow.connect(thresholded_skeleton, 'out_file', binarised_skeleton, 'in_file')

    # Create skeleton distance map
    invert_mask1 = pe.Node(interface=niftyseg.BinaryMaths(),
                           name='invert_mask1')
    invert_mask1.inputs.operation = 'mul'
    invert_mask1.inputs.operand_value = -1
    workflow.connect(fa_mask, 'out_file', invert_mask1, 'in_file')
    invert_mask2 = pe.Node(interface=niftyseg.BinaryMaths(),
                           name='invert_mask2')
    invert_mask2.inputs.operation = 'add'
    invert_mask2.inputs.operand_value = 1
    workflow.connect(invert_mask1, 'out_file', invert_mask2, 'in_file')
    invert_mask3 = pe.Node(interface=niftyseg.BinaryMaths(),
                           name='invert_mask3')
    invert_mask3.inputs.operation = 'add'
    workflow.connect(invert_mask2, 'out_file', invert_mask3, 'in_file')
    workflow.connect(binarised_skeleton, 'out_file', invert_mask3, 'operand_file')
    distance_map = pe.Node(interface=fsl.DistanceMap(),
                           name='distance_map')
    workflow.connect(invert_mask3, 'out_file', distance_map, 'in_file')

    # Project the FA values onto the skeleton
    all_fa_projected = pe.Node(interface=fsl.TractSkeleton(),
                               name='all_fa_projected')
    all_fa_projected.inputs.threshold = skeleton_threshold
    all_fa_projected.inputs.project_data = True
    workflow.connect(masked_mean_fa, 'out_file', all_fa_projected, 'in_file')
    workflow.connect(distance_map, 'distance_map', all_fa_projected, 'distance_map')
    workflow.connect(merged_4d_fa, 'merged_file', all_fa_projected, 'data_file')
    workflow.connect(bin_atlas, 'out_file', all_fa_projected, 'search_mask_file')

    # Project the MD values onto the skeleton
    all_md_projected = pe.Node(interface=fsl.TractSkeleton(),
                               name='all_md_projected')
    all_md_projected.inputs.threshold = skeleton_threshold
    all_md_projected.inputs.project_data = True
    workflow.connect(masked_mean_fa, 'out_file', all_md_projected, 'in_file')
    workflow.connect(distance_map, 'distance_map', all_md_projected, 'distance_map')
    workflow.connect(merged_4d_fa, 'merged_file', all_md_projected, 'data_file')
    workflow.connect(merged_4d_md, 'merged_file', all_md_projected, 'alt_data_file')
    workflow.connect(bin_atlas, 'out_file', all_md_projected, 'search_mask_file')

    # Project the RD values onto the skeleton
    all_rd_projected = pe.Node(interface=fsl.TractSkeleton(),
                               name='all_rd_projected')
    all_rd_projected.inputs.threshold = skeleton_threshold
    all_rd_projected.inputs.project_data = True
    workflow.connect(masked_mean_fa, 'out_file', all_rd_projected, 'in_file')
    workflow.connect(distance_map, 'distance_map', all_rd_projected, 'distance_map')
    workflow.connect(merged_4d_fa, 'merged_file', all_rd_projected, 'data_file')
    workflow.connect(merged_4d_rd, 'merged_file', all_rd_projected, 'alt_data_file')
    workflow.connect(bin_atlas, 'out_file', all_rd_projected, 'search_mask_file')

    # Project the RD values onto the skeleton
    all_ad_projected = pe.Node(interface=fsl.TractSkeleton(),
                               name='all_ad_projected')
    all_ad_projected.inputs.threshold = skeleton_threshold
    all_ad_projected.inputs.project_data = True
    workflow.connect(masked_mean_fa, 'out_file', all_ad_projected, 'in_file')
    workflow.connect(distance_map, 'distance_map', all_ad_projected, 'distance_map')
    workflow.connect(merged_4d_fa, 'merged_file', all_ad_projected, 'data_file')
    workflow.connect(merged_4d_ad, 'merged_file', all_ad_projected, 'alt_data_file')
    workflow.connect(bin_atlas, 'out_file', all_ad_projected, 'search_mask_file')

    # Create an output node
    output_node = pe.Node(
        interface=niu.IdentityInterface(
            fields=['mean_fa',
                    'all_fa_skeletonised',
                    'all_md_skeletonised',
                    'all_rd_skeletonised',
                    'all_ad_skeletonised',
                    'skeleton',
                    'skeleton_bin',
                    't_contrast_raw_stat',
                    't_contrast_uncorrected_pvalue',
                    't_contrast_corrected_pvalue']),
        name='output_node')

    # Connect the workflow to the output node
    workflow.connect(masked_mean_fa, 'out_file', output_node, 'mean_fa')
    workflow.connect(all_fa_projected, 'projected_data', output_node, 'all_fa_skeletonised')
    workflow.connect(all_md_projected, 'projected_data', output_node, 'all_md_skeletonised')
    workflow.connect(all_rd_projected, 'projected_data', output_node, 'all_rd_skeletonised')
    workflow.connect(all_ad_projected, 'projected_data', output_node, 'all_ad_skeletonised')
    workflow.connect(skeleton, 'skeleton_file', output_node, 'skeleton')
    workflow.connect(binarised_skeleton, 'out_file', output_node, 'skeleton_bin')

    # Run randomise if required and connect its output to the output node
    if design_mat is not None and design_con is not None:
        randomise = pe.Node(interface=fsl.Randomise(),
                            name='randomise')
        randomise.inputs.base_name = 'stats_tbss'
        randomise.inputs.tfce2D = True
        randomise.inputs.num_perm = 5000
        workflow.connect(all_fa_projected, 'projected_data', randomise, 'in_file')
        randomise.inputs.design_mat = design_mat
        randomise.inputs.design_con = design_con
        workflow.connect(binarised_skeleton, 'out_file', randomise, 'mask')

        workflow.connect(randomise, 'tstat_files', output_node, 't_contrast_raw_stat')
        workflow.connect(randomise, 't_p_files', output_node, 't_contrast_uncorrected_pvalue')
        workflow.connect(randomise, 't_corrected_p_files', output_node, 't_contrast_corrected_pvalue')

    # Create nodes to rename the outputs
    mean_fa_renamer = pe.Node(interface=niu.Rename(format_string='tbss_mean_fa',
                                                   keep_ext=True),
                              name='mean_fa_renamer')
    workflow.connect(output_node, 'mean_fa', mean_fa_renamer, 'in_file')

    mean_sk_renamer = pe.Node(interface=niu.Rename(format_string='tbss_mean_fa_skeleton',
                                                   keep_ext=True),
                              name='mean_sk_renamer')
    workflow.connect(output_node, 'skeleton', mean_sk_renamer, 'in_file')

    bin_ske_renamer = pe.Node(interface=niu.Rename(format_string='tbss_mean_fa_skeleton_mask',
                                                   keep_ext=True),
                              name='bin_ske_renamer')
    workflow.connect(output_node, 'skeleton_bin', bin_ske_renamer, 'in_file')

    fa_skel_renamer = pe.Node(interface=niu.Rename(format_string='tbss_all_fa_skeletonised',
                                                   keep_ext=True),
                              name='fa_skel_renamer')
    workflow.connect(output_node, 'all_fa_skeletonised', fa_skel_renamer, 'in_file')
    md_skel_renamer = pe.Node(interface=niu.Rename(format_string='tbss_all_md_skeletonised',
                                                   keep_ext=True),
                              name='md_skel_renamer')
    workflow.connect(output_node, 'all_md_skeletonised', md_skel_renamer, 'in_file')
    rd_skel_renamer = pe.Node(interface=niu.Rename(format_string='tbss_all_rd_skeletonised',
                                                   keep_ext=True),
                              name='rd_skel_renamer')
    workflow.connect(output_node, 'all_rd_skeletonised', rd_skel_renamer, 'in_file')
    ad_skel_renamer = pe.Node(interface=niu.Rename(format_string='tbss_all_ad_skeletonised',
                                                   keep_ext=True),
                              name='ad_skel_renamer')
    workflow.connect(output_node, 'all_ad_skeletonised', ad_skel_renamer, 'in_file')

    # Create a data sink
    ds = pe.Node(nio.DataSink(parameterization=False), name='data_sink')
    ds.inputs.base_directory = os.path.abspath(output_dir)

    # Connect the data sink
    workflow.connect(mean_fa_renamer, 'out_file', ds, '@mean_fa')
    workflow.connect(mean_sk_renamer, 'out_file', ds, '@skel_fa')
    workflow.connect(bin_ske_renamer, 'out_file', ds, '@bkel_fa')
    workflow.connect(fa_skel_renamer, 'out_file', ds, '@all_fa')
    workflow.connect(md_skel_renamer, 'out_file', ds, '@all_md')
    workflow.connect(rd_skel_renamer, 'out_file', ds, '@all_rd')
    workflow.connect(ad_skel_renamer, 'out_file', ds, '@all_ad')

    if design_mat is not None and design_con is not None:
        workflow.connect(output_node, 't_contrast_raw_stat',
                         ds, '@t_contrast_raw_stat')
        workflow.connect(output_node, 't_contrast_uncorrected_pvalue',
                         ds, '@t_contrast_uncorrected_pvalue')
        workflow.connect(output_node, 't_contrast_corrected_pvalue',
                         ds, '@t_contrast_corrected_pvalue')

    return workflow
