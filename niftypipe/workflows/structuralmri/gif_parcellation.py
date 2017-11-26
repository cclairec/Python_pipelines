# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
import nipype.interfaces.utility as niu
import nipype.pipeline.engine as pe
import nipype.interfaces.niftyreg as niftyreg
from nipype.interfaces.dcm2nii import Dcm2nii
from nipype.utils.filemanip import split_filename
import nipype.interfaces.fsl as fsl
from ...interfaces.niftk.filters import N4BiasCorrection
from ...interfaces.niftk.gif import Gif
from ...interfaces.niftk.io import Pct2Dcm


def extract_db_info_function(in_db_file):
    import os
    from glob import glob

    def find_database_info(in_file):
        import xml.etree.ElementTree as Xml
        tree = Xml.parse(in_file)
        root = tree.getroot()
        data = root.findall('data')[0]
        return data.find('path').text, data.find('sform').text, data.find('gm').text

    out_templates, sform, group_mask = find_database_info(in_db_file)
    if not sform == '1':
        raise Exception('Images from the database are expected to be sform aligned')
    out_templates = glob(os.path.join(os.path.dirname(in_db_file), out_templates, '*.nii*'))
    return out_templates, os.path.join(os.path.dirname(in_db_file), group_mask)


def registration_sink_function(templates, aff_files, cpp_files, in_dir):
    import os
    import sys
    import shutil
    from nipype.utils.filemanip import split_filename

    if len(templates) != len(aff_files) or len(templates) != len(cpp_files):
        print ('ERROR: (registration_sink_function): number of inputs differ')
        print ('templates (%s) / affs (%s) / cpps (%s)'
               % (str(len(templates)), str(len(aff_files)), str(len(cpp_files))))
        sys.exit(1)
    template_names = [split_filename(os.path.basename(f))[1] for f in templates]
    out_dir = os.path.join(in_dir, 'cpps')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for i in range(len(templates)):
        shutil.copy(aff_files[i], os.path.join(out_dir, template_names[i] + '.txt'))
        shutil.copy(cpp_files[i], os.path.join(out_dir, template_names[i] + '.nii.gz'))
    return out_dir


def create_gif_propagation_workflow(in_file,
                                    in_db_file,
                                    output_dir,
                                    in_mask_file=None,
                                    name='gif_propagation',
                                    use_lncc=False):

    """create_niftyseg_gif_propagation_pipeline.
    @param in_file            input target file
    @param in_db_file         input database xml file for the GIF algorithm
    @param output_dir         output directory
    @param in_mask_file       optional input mask for the target T1 file
    @param name               optional name of the pipeline
    """

    # Extract the basename of the input file
    subject_id = split_filename(os.path.basename(in_file))[1]
    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    input_node = pe.Node(interface=niu.IdentityInterface(fields=['input_file',
                                                                 'mask_file',
                                                                 'database_file']),
                         name='input_node')
    input_node.inputs.input_file = in_file
    input_node.inputs.database_file = in_db_file
    input_node.inputs.mask_file = in_mask_file

    # Extract the database information
    extract_db_info = pe.Node(interface=niu.Function(input_names=['in_db_file'],
                                                     output_names=['out_templates',
                                                                   'group_mask'],
                                                     function=extract_db_info_function),
                              name='extract_db_info')
    workflow.connect(input_node, 'database_file', extract_db_info, 'in_db_file')

    # Affine registration - All images in the database are registered to the input image
    affine_registration = pe.MapNode(interface=niftyreg.RegAladin(),
                                     iterfield='flo_file',
                                     name='affine_registration')
    workflow.connect(input_node, 'input_file', affine_registration, 'ref_file')
    workflow.connect(extract_db_info, 'out_templates', affine_registration, 'flo_file')

    # Extract a robust affine registration if applicable
    robust_affine = pe.Node(interface=niftyreg.RegAverage(),
                            name='robust_affine')
    workflow.connect(affine_registration, 'aff_file', robust_affine, 'avg_lts_files')

    # A mask is created
    propagate_mask = None
    if in_mask_file is None:
        propagate_mask = pe.Node(interface=niftyreg.RegResample(inter_val='NN', pad_val=0),
                                 name='propagate_mask')
        workflow.connect(input_node, 'input_file', propagate_mask, 'ref_file')
        workflow.connect(extract_db_info, 'group_mask', propagate_mask, 'flo_file')
        workflow.connect(robust_affine, 'out_file', propagate_mask, 'trans_file')

    # Initial Bias correction of the input image
    bias_correction = pe.Node(interface=N4BiasCorrection(in_downsampling=2),
                              name='bias_correction')
    workflow.connect(input_node, 'input_file', bias_correction, 'in_file')
    if in_mask_file is None:
        workflow.connect(propagate_mask, 'out_file', bias_correction, 'mask_file')
    else:
        workflow.connect(input_node, 'mask_file', bias_correction, 'mask_file')

    # Non linear registration
    non_linear_registration = pe.MapNode(interface=niftyreg.RegF3D(ln_val=4),
                                         iterfield='flo_file',
                                         name='non_linear_registration')
    workflow.connect(bias_correction, 'out_file', non_linear_registration, 'ref_file')
    workflow.connect(extract_db_info, 'out_templates', non_linear_registration, 'flo_file')
    workflow.connect(robust_affine, 'out_file', non_linear_registration, 'aff_file')
    if in_mask_file is None:
        workflow.connect(propagate_mask, 'out_file', non_linear_registration, 'rmask_file')
    else:
        workflow.connect(input_node, 'mask_file', non_linear_registration, 'rmask_file')
    if use_lncc:
        non_linear_registration.inputs.lncc_val = -5

    # Save all the images where required
    registration_sink = pe.Node(interface=niu.Function(input_names=['templates', 'aff_files', 'cpp_files', 'in_dir'],
                                                       output_names=['out_dir'],
                                                       function=registration_sink_function),
                                name='registration_sink')
    registration_sink.inputs.in_dir = output_dir
    workflow.connect(extract_db_info, 'out_templates', registration_sink, 'templates')
    workflow.connect(affine_registration, 'aff_file', registration_sink, 'aff_files')
    workflow.connect(non_linear_registration, 'cpp_file', registration_sink, 'cpp_files')

    # Run GIF
    gif = pe.Node(interface=Gif(database_file=in_db_file),
                  name='gif')
    gif.inputs.omp_core_val = 8
    workflow.connect(registration_sink, 'out_dir', gif, 'cpp_dir')
    workflow.connect(bias_correction, 'out_file', gif, 'in_file')

    if in_mask_file is None:
        workflow.connect(propagate_mask, 'out_file', gif, 'mask_file')
    else:
        workflow.connect(input_node, 'mask_file', gif, 'mask_file')

    # Rename and redirect the output
    output_merger = pe.Node(interface=niu.Merge(numinputs=7),
                            name='output_merger')
    workflow.connect(gif, 'parc_file', output_merger, 'in1')
    workflow.connect(gif, 'prior_file', output_merger, 'in2')
    workflow.connect(gif, 'tiv_file', output_merger, 'in3')
    workflow.connect(gif, 'seg_file', output_merger, 'in4')
    workflow.connect(gif, 'brain_file', output_merger, 'in5')
    workflow.connect(gif, 'bias_file', output_merger, 'in6')
    workflow.connect(gif, 'volume_file', output_merger, 'in7')
    renamer = pe.MapNode(interface=niu.Rename(format_string=subject_id + "_%(type)s", keep_ext=True),
                         iterfield=['in_file', 'type'],
                         name='renamer')
    renamer.inputs.type = ['labels', 'prior', 'tiv', 'seg', 'brain', 'bias_corrected', 'volumes']
    workflow.connect(output_merger, 'out', renamer, 'in_file')

    return workflow


def extract_nac_pet(dicom_folder):
    """
    Extract the Non attenuation Corrected PET from a DICOM exam.

    The function extract the last frame of the DICOM exam.

    The DICOM exam is expected to be of the dynamic PET scan itself. It also expects the image dimension
    in the transversal acquisition direction to be of 127

    It uses dcm2nii to convert the DICOM subset into a nifti file

    :param dicom_folder: The input DICOM folder of the dynamic PET scan
    :return: the nifti converted image file (the last frame of the dynamic PET scan)
    """
    from glob import glob
    import os
    import shutil
    import re
    from nipype.interfaces.dcm2nii import Dcm2nii

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    files = glob(os.path.join(os.path.abspath(dicom_folder), '*'))
    sorted_files = sorted(files, key=natural_keys)
    nac_pet_files = sorted_files[-127:]
    for f in nac_pet_files:
        shutil.copy(f, os.getcwd())
    dcm2nii = Dcm2nii()
    dcm2nii.inputs.source_dir = os.getcwd()
    nii_outputs = dcm2nii.run().outputs.converted_files
    print (nii_outputs)
    return nii_outputs[0]


def convert_pct_hu_to_umap(pct_file,
                           structural_mri_file,
                           ute_echo2_file):
    """
    Takes a pCT image in HU units into 10'000.cm-1, and transported into the space of the ute_echo2

    The pCT image is in the space of the structural image. The structural image is therefore registered to the
    ute_echo2 image, and the resulting transformation is used to resample the pCT image.

    :param pct_file: The input pseudoCT image file
    :param structural_mri_file: The structural image used to synthesise the pseudoCT
    :param ute_echo2_file: The ute echo2 image file
    :return: The converted and transported pCT image
    """
    import os
    # Convert pseudoCT in HU to pseudoCT_mmrumap in 10000*cm-1
    upet_w = 0.096
    b = 0.7445
    cmd1 = 'seg_maths %s -thr 0 -sub 1024 -thr 0 -div 1000 -mul %s mmrumap.nii.gz' % (pct_file, str(b))
    os.system(cmd1)
    cmd2 = 'seg_maths %s -thr 0 -sub 1024 -mul -1 -thr 0 -mul -1 -div 1000 -add mmrumap.nii.gz -add 1 ' % pct_file +\
           '-mul %s -thr 0 -uthr 0.4095 -mul 10000 -scl -range -odt ushort mmrumap.nii.gz' % str(upet_w)
    os.system(cmd2)
    cmd3 = 'reg_aladin -voff -rigOnly -ref %s -flo %s -aff affine2UTE.txt -res unused.nii.gz' %\
           (ute_echo2_file, structural_mri_file)
    os.system(cmd3)
    cmd4 = 'reg_resample -ref %s -flo mmrumap.nii.gz -trans affine2UTE.txt -pad 0 -res mmrumap.nii.gz' % ute_echo2_file
    os.system(cmd4)

    return os.path.abspath('mmrumap.nii.gz')


def create_full_mask(in_file):
    import nibabel as nib
    import os
    import numpy as np
    from nipype.utils.filemanip import split_filename
    in_image = nib.load(in_file)
    data = in_image.get_data()
    out_data = np.ones(data.shape)
    out_img = nib.Nifti1Image(np.uint8(out_data), in_image.get_affine())
    out_img.set_data_dtype('uint8')
    out_img.set_qform(in_image.get_qform())
    out_img.set_sform(in_image.get_sform())
    _, basename, extension = split_filename(in_file)
    out_file_name = basename + '_full_mask' + extension
    out_img.to_filename(out_file_name)
    return os.path.abspath(out_file_name)


def create_gif_pseudoct_workflow(in_ute_echo2_file,
                                 in_ute_umap_dir,
                                 in_db_file,
                                 cpp_dir,
                                 in_t1_file=None,
                                 in_t2_file=None,
                                 in_mask_file=None,
                                 in_nac_pet_dir=None,
                                 name='gif_pseudoct'):

    """create_niftyseg_gif_propagation_pipeline.
    @param in_ute_echo2_file  input UTE echo file
    @param in_ute_umap_dir    input UTE umap file
    @param in_db_file         input database xml file for the GIF algorithm
    @param cpp_dir            cpp directory
    @param in_t1_file         input T1 target file
    @param in_t2_file         input T2 target file
    @param in_mask_file       optional input mask for the target T1 file
    @param name               optional name of the pipeline
    """

    in_file = in_t1_file if in_t1_file else in_t2_file
    subject_id = split_filename(os.path.basename(in_file))[1]

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    gif = pe.Node(interface=Gif(database_file=in_db_file, cpp_dir=cpp_dir,
                                lncc_ker=3, regNMI=True, regBE=0.01),
                  name='gif')
    if in_mask_file:
        gif.inputs.mask_file = in_mask_file

    # Create empty masks for the bias correction to cover the full image
    t1_full_mask = pe.Node(interface=niu.Function(input_names=['in_file'], output_names=['out_file'],
                                                  function=create_full_mask),
                           name='t1_full_mask')
    t1_full_mask.inputs.in_file = in_t1_file
    t2_full_mask = pe.Node(interface=niu.Function(input_names=['in_file'], output_names=['out_file'],
                                                  function=create_full_mask),
                           name='t2_full_mask')
    t2_full_mask.inputs.in_file = in_t2_file

    # Create bias correction nodes that are adapted to our needs. i.e. Boost the T2 bias correction
    bias_correction_t1 = pe.Node(interface=N4BiasCorrection(),
                                 name='bias_correction_t1')
    if in_t1_file:
        bias_correction_t1.inputs.in_file = in_t1_file

    # Create bias correction nodes that are adapted to our needs. i.e. Boost the T2 bias correction
    bias_correction_t2 = pe.Node(interface=N4BiasCorrection(in_maxiter=300, in_convergence=0.0001),
                                 name='bias_correction_t2')
    if in_t2_file:
        bias_correction_t2.inputs.in_file = in_t2_file

    # Only connect the nodes if the input image exist respectively
    if in_t1_file:
        workflow.connect(t1_full_mask, 'out_file', bias_correction_t1, 'mask_file')
    if in_t2_file:
        workflow.connect(t2_full_mask, 'out_file', bias_correction_t2, 'mask_file')

    if in_t1_file and in_t2_file:
        affine_mr_target = pe.Node(interface=niftyreg.RegAladin(maxit_val=10), name='affine_mr_target')
        workflow.connect(bias_correction_t1, 'out_file', affine_mr_target, 'ref_file')
        workflow.connect(bias_correction_t2, 'out_file', affine_mr_target, 'flo_file')
        resample_mr_target = pe.Node(interface=niftyreg.RegResample(pad_val=float('nan')),
                                     name='resample_MR_target')
        workflow.connect(bias_correction_t1, 'out_file', resample_mr_target, 'ref_file')
        workflow.connect(bias_correction_t2, 'out_file', resample_mr_target, 'flo_file')
        lister = pe.Node(interface=niu.Merge(2),
                         name='lister')
        merger = pe.Node(interface=fsl.Merge(dimension='t', output_type='NIFTI_GZ'),
                         name='fsl_merge')
        workflow.connect(affine_mr_target, 'aff_file', resample_mr_target, 'trans_file')
        workflow.connect(bias_correction_t1, 'out_file', lister, 'in1')
        workflow.connect(resample_mr_target, 'out_file', lister, 'in2')
        workflow.connect(lister, 'out', merger, 'in_files')
        workflow.connect(merger, 'merged_file', gif, 'in_file')
    else:
        if in_t1_file:
            workflow.connect(bias_correction_t1, 'out_file', gif, 'in_file')
        if in_t2_file:
            workflow.connect(bias_correction_t2, 'out_file', gif, 'in_file')

    pct_hu_to_umap = pe.Node(interface=niu.Function(input_names=['pCT_file', 'structural_mri_file', 'ute_echo2_file'],
                                                    output_names=['pct_umap_file'],
                                                    function=convert_pct_hu_to_umap),
                             name='pCT_HU_to_umap')
    pct_hu_to_umap.inputs.structural_mri_file = in_file
    pct_hu_to_umap.inputs.ute_echo2_file = in_ute_echo2_file
    workflow.connect(gif, 'synth_file', pct_hu_to_umap, 'pCT_file')

    pct2dcm_pct_umap = pe.Node(interface=Pct2Dcm(in_umap_name='pCT_umap'), name='pct2dcm_pct_umap')
    workflow.connect(pct_hu_to_umap, 'pct_umap_file', pct2dcm_pct_umap, 'in_umap_file')
    pct2dcm_pct_umap.inputs.in_ute_umap_dir = os.path.abspath(in_ute_umap_dir)

    merger_output_number = 2

    pct2dcm_ute_umap_end = None
    pct2dcm_pct_umap_end = None
    if in_nac_pet_dir:

        ute_umap_dcm2nii = pe.Node(interface=Dcm2nii(source_dir=in_ute_umap_dir),
                                   name='ute_umap_dcm2nii')
        first_item_selector = pe.Node(interface=niu.Select(index=0),
                                      name='first_item_selector')
        workflow.connect(ute_umap_dcm2nii, 'converted_files', first_item_selector, 'inlist')

        nac_extractor = pe.Node(interface=niu.Function(input_names=['dicom_folder'],
                                                       output_names=['nifti_file'],
                                                       function=extract_nac_pet),
                                name='nac_extractor')
        nac_extractor.inputs.dicom_folder = in_nac_pet_dir

        ute_to_nac_registration = pe.Node(interface=niftyreg.RegAladin(rig_only_flag=True),
                                          name='ute_to_nac_registration')
        workflow.connect(nac_extractor, 'nifti_file', ute_to_nac_registration, 'ref_file')
        ute_to_nac_registration.inputs.flo_file = in_ute_echo2_file

        ute_resample = pe.Node(interface=niftyreg.RegResample(), name='ute_resample')
        workflow.connect(first_item_selector, 'out', ute_resample, 'ref_file')
        workflow.connect(first_item_selector, 'out', ute_resample, 'flo_file')
        workflow.connect(ute_to_nac_registration, 'aff_file', ute_resample, 'aff_file')

        pct2dcm_ute_umap_end = pe.Node(interface=Pct2Dcm(in_umap_name='UTE_umap_end'), name='pct2dcm_ute_umap_end')
        workflow.connect(ute_resample, 'res_file', pct2dcm_ute_umap_end, 'in_umap_file')
        pct2dcm_ute_umap_end.inputs.in_ute_umap_dir = os.path.abspath(in_ute_umap_dir)

        pct_resample = pe.Node(interface=niftyreg.RegResample(), name='pct_resample')
        workflow.connect(pct_hu_to_umap, 'pct_umap_file', pct_resample, 'ref_file')
        workflow.connect(pct_hu_to_umap, 'pct_umap_file', pct_resample, 'flo_file')
        workflow.connect(ute_to_nac_registration, 'aff_file', pct_resample, 'aff_file')

        pct2dcm_pct_umap_end = pe.Node(interface=Pct2Dcm(in_umap_name='pCT_umap_end'), name='pct2dcm_pct_umap_end')
        workflow.connect(pct_resample, 'res_file', pct2dcm_pct_umap_end, 'in_umap_file')
        pct2dcm_pct_umap_end.inputs.in_ute_umap_dir = os.path.abspath(in_ute_umap_dir)

        merger_output_number = 4

    # merge output
    output_merger = pe.Node(interface=niu.Merge(numinputs=merger_output_number),
                            name='output_merger')
    workflow.connect(gif, 'synth_file', output_merger, 'in1')
    workflow.connect(pct2dcm_pct_umap, 'output_file', output_merger, 'in2')

    renamer = pe.Node(interface=niu.Rename(format_string=subject_id + "_%(type)s",
                                           keep_ext=True), name='renamer')
    if in_nac_pet_dir:
        workflow.connect(pct2dcm_ute_umap_end, 'output_file', output_merger, 'in3')
        workflow.connect(pct2dcm_pct_umap_end, 'output_file', output_merger, 'in4')
        renamer.inputs.type = ['synth', 'pct', 'ute_end', 'pct_end']
    else:
        renamer.inputs.type = ['synth', 'pct']
    workflow.connect(output_merger, 'out', renamer, 'in_file')

    return workflow
