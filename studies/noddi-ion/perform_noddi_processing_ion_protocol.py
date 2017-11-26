#! /usr/bin/env python

import os, re
import nipype.pipeline.engine           as pe
import nipype.interfaces.io             as nio
import nipype.interfaces.fsl            as fsl
import nipype.interfaces.utility        as niu  
from nipype.interfaces                  import Function
from nipype                             import config, logging
import argparse
from nipype import config

# modified custom interfaces
import niftk

import nipype.interfaces.niftyreg as niftyreg
import nipype.interfaces.niftyseg as niftyseg

mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
mni_template_mask = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain_mask_dil.nii.gz')
spinal_cord_template = os.path.abspath('/usr2/mrtools/pipelines/database/spinal_cord_segmentation/b0/spinal_cord_template_b0.nii.gz')
spinal_cord_template_mask = os.path.abspath('/usr2/mrtools/pipelines/database/spinal_cord_segmentation/b0/spinal_cord_template_b0_mask.nii.gz')

def gen_substitutions(op_basename):    
    subs = []    

    subs.append(('eddy_corrected', op_basename+'_corrected_dwi'))
    subs.append(('merge.bval', op_basename + '_corrected_dwi.bval'))
    subs.append(('merge.bvec', op_basename + '_corrected_dwi.bvec'))
    subs.append(('noddi_', op_basename+ '_noddi_'))
    subs.append((r'slice.*corrected_dwi_merged', op_basename+ '_corrected_dwi'))
    subs.append((r'spinal_cord_template.*_mask.*_merged', op_basename+ '_mask'))
    subs.append((r'MNI152_T1_2.*_mask.*_res', op_basename+ '_mask'))

    return subs

# select DWI files from list of converted files coming from DCM2Nii
def find_DTIfiles(in_files,bval_files,bvec_files,filename_pattern):
   from nipype.utils.filemanip import split_filename
   out_files=[]
   out_bvecfiles=[]
   out_bvalfiles=[]
   for f in in_files:
        base_nii, filename_nii, nii_ext = split_filename(f)
        # change this line to accommodate different file names
        if filename_pattern in filename_nii and not 'Reg' in filename_nii:
            out_files.append(f);
            for bvecf in bvec_files:
                base_bvec, filename_bvec, bvec_ext = split_filename(bvecf)
                if filename_nii in filename_bvec:
                    out_bvecfiles.append(bvecf);
            for bvalf in bval_files:            
                base_bval, filename_bval, bval_ext = split_filename(bvalf)
                if filename_nii in filename_bval:
                    out_bvalfiles.append(bvalf);
   return out_files,out_bvalfiles,out_bvecfiles


# prepends b=0 to bval/bvec files that have been modified by dcm2nii
# (that occurs with DWI acquired with interleaved b=0 on patched IoN scanner)
def prepend_b0(bval_files,bvec_files,numB0s=1,delim=' '):
    from os.path import abspath
    from nipype.utils.filemanip import split_filename
    out_bvecfiles=[]
    out_bvalfiles=[]
    for bvf in bval_files + bvec_files:
        base_bfile, filename_bfile, bfile_ext = split_filename(bvf)
        bfile_out_filename='%s.b0prepend%s'%(filename_bfile, bfile_ext)
        out_fp = open(bfile_out_filename, 'w')
        in_fp = open(bvf, 'r')
        for line in in_fp:
            out_line=line
            for i in range(0,numB0s):
                out_line='0%s%s'%(delim,out_line)
            out_fp.write(out_line)
        if (bfile_ext=='.bval'):
            out_bvalfiles.append(abspath(bfile_out_filename))
        elif (bfile_ext=='.bvec'):
            out_bvecfiles.append(abspath(bfile_out_filename))
    return out_bvalfiles,out_bvecfiles



def merge_bvalsbvecs(bvec_files,bval_files, normalise):
   from numpy import loadtxt,savetxt, concatenate, reshape, sqrt
   from os.path import abspath

   out_bvalfile="merge.bval"
   out_bvecfile="merge.bvec"
   
   print(normalise)
   
   merged_bvals=None
   merged_bvecs=None
   
   for f in bval_files:
       bval=loadtxt(f)
       if merged_bvals is None:
            merged_bvals=bval
       else:
            merged_bvals=concatenate((merged_bvals,bval));
       
   for f in bvec_files:
       bvec=loadtxt(f)
       if merged_bvecs is None:
            merged_bvecs=bvec
       else:
            merged_bvecs=concatenate((merged_bvecs,bvec),axis=1);
            
    
   if normalise==True:
       b_norm=sqrt(merged_bvecs[0,]**2+merged_bvecs[1,]**2+merged_bvecs[2,]**2)
       merged_bvals=merged_bvals*b_norm
    
    
   savetxt(out_bvalfile, reshape(merged_bvals,(1,merged_bvals.shape[0])), fmt='%.3f', delimiter=' ')
   savetxt(out_bvecfile, merged_bvecs, fmt='%.3f', delimiter=' ')
   return abspath(out_bvalfile),abspath(out_bvecfile)

def _gen_dcm2nii_config():
    from os.path import abspath
    dcm2nii_config_file=os.path.join(os.getcwd(),'noddi_mri_processing','converter.ini');
    f = open(dcm2nii_config_file, "w")
    # disable interactive mode
    f.write("[BOOL]\nManualNIfTIConv=0\n")
    f.write("PhilipsPrecise=1\n")
    f.close()
    return dcm2nii_config_file

def _gen_index(in_file):
    import numpy as np
    import nibabel as nb
    import os
    out_file=os.path.join(os.getcwd(),'noddi_mri_processing','index.txt')
    vols = nb.load(in_file).get_data().shape[-1]
    np.savetxt(out_file, np.ones((vols,)).T,fmt='%1.0f')
    return out_file

def _gen_acqpfile(acqtime,phase_enc='+y'):
    import os
    
    out_string=''
    
    if (str.lower(phase_enc)=='+x'):
        out_string='1 0 0'
    elif (str.lower(phase_enc)=='-x'):
        out_string='-1 0 0'
    elif (str.lower(phase_enc)=='+y'):
        out_string='0 1 0'
    elif (str.lower(phase_enc)=='-y'):
        out_string='0 -1 0'
    elif (str.lower(phase_enc)=='+z'):
        out_string='0 0 1'
    elif (str.lower(phase_enc)=='-z'):
        out_string='0 0 -1'
    else:
        error('No valid phase encoding')
    
    # make output string
    out_string='%s %1.6f'%(out_string,acqtime)
    
    # write to file
    acqp_file=os.path.join(os.getcwd(),'noddi_mri_processing','acqp.txt')
    f = open(acqp_file, "w")
    f.write(out_string)
    f.close()
        
    return acqp_file

'''
This file provides the creation of the whole workflow necessary for 
processing NODDI MRI images, starting from the DICOM images

input parameters are the workflow name and dti_filename pattern (to select the NODDI sequences)

data and corresponding bvalues/bvectors are selected from the output of dcm2nii, corrected for
dcm2nii conversion (removal of leading b=0 in bval/bvec), merged and fed into eddy before NODDI processing

'''

def create_noddi_mri_processing_workflow(name='noddi_mri_processing', 
                                         dti_filename_pattern = 'DTI',
                                         useBET=False,
                                         workingOnSpinalCord=False,
                                         working_using_philips_dwi_param=False,
                                         precomputed_mask='',
                                         computeMask=False):

    workflow = pe.Workflow(name=name)
    
    # input specification
    input_node = pe.Node(
        niu.IdentityInterface(
            fields=['in_dicom_folder',
                    'in_epi_readouttime',
                    'in_noise_scaling_factor',
                    'in_result_dir',
		    'in_subject_name'],
             mandatory_inputs=True),
        name='input_node')
    
    workflow.add_nodes([input_node])   
    
    # converts from DCM to NIFTI using PhilipsPrecise=1 for correct value conversion
    converter = pe.Node(interface=niftk.Dcm2niiPhilips(), 
                        name='convert_dicom')
    converter.inputs.config_file=_gen_dcm2nii_config()
    workflow.add_nodes([converter])
    workflow.connect(input_node, 'in_dicom_folder', converter, 'source_dir')
    
    # collect only DTI files    
    select_dti_files = pe.Node(name='select_dwi', 
                               interface=Function(input_names=['in_files','bval_files','bvec_files','filename_pattern'], 
                               output_names=['out_files','bval_files','bvec_files'], 
                               function=find_DTIfiles))
    select_dti_files.inputs.filename_pattern=dti_filename_pattern
    workflow.add_nodes([select_dti_files])
    workflow.connect(converter, 'converted_files', select_dti_files, 'in_files')
    workflow.connect(converter, 'bvecs', select_dti_files, 'bvec_files')
    workflow.connect(converter, 'bvals', select_dti_files, 'bval_files')
    
    if not working_using_philips_dwi_param:
        # prepend b=0 (if patch is used)
        mod_bvalbvec = pe.Node(name='prepend_b0_bfiles', 
                               interface=Function(input_names=['bval_files','bvec_files'], 
                               output_names=['bval_files','bvec_files'], 
                               function=prepend_b0))
        workflow.add_nodes([mod_bvalbvec])
        workflow.connect(select_dti_files, 'bval_files', mod_bvalbvec, 'bval_files')
        workflow.connect(select_dti_files, 'bvec_files', mod_bvalbvec, 'bvec_files')

    # merge bval and bvec files
    merge_bvalbvec = pe.Node(name='merge_bfiles', 
                             interface=Function(input_names=['bval_files','bvec_files','normalise'], 
                             output_names=['merged_bval_file','merged_bvec_file'], 
                             function=merge_bvalsbvecs))
    merge_bvalbvec.inputs.normalise=True
    workflow.add_nodes([merge_bvalbvec])

    if not working_using_philips_dwi_param:
        workflow.connect(mod_bvalbvec, 'bval_files', merge_bvalbvec, 'bval_files')
        workflow.connect(mod_bvalbvec, 'bvec_files', merge_bvalbvec, 'bvec_files')
    else:
        workflow.connect(select_dti_files, 'bval_files', merge_bvalbvec, 'bval_files')
        workflow.connect(select_dti_files, 'bvec_files', merge_bvalbvec, 'bvec_files')

    # merge DTI datafiles
    nii_merge = pe.Node(interface=fsl.Merge(),
                        name='merge_dwi_data')
    nii_merge.inputs.dimension = 't'
    workflow.add_nodes([nii_merge])
    workflow.connect(select_dti_files,'out_files',nii_merge,'in_files')


    if not workingOnSpinalCord:
        if computeMask:
            if useBET:
                # run FSL BET to get brainmask
                bet = pe.Node(interface=fsl.BET(), 
                              name="bet")
                bet.inputs.mask = True
                bet.inputs.functional = True
                workflow.add_nodes([bet])
                workflow.connect(nii_merge,'merged_file',bet,'in_file')
            else:
                # We run a skull stripping using registration 
                # We extracta B0 image
                image_extraction = pe.Node(interface = niftyseg.BinaryMathsInteger(), 
                                           name = 'extract_b0')
                image_extraction.inputs.operation = 'tp'
                image_extraction.inputs.operand_value = 0
                workflow.connect(nii_merge,'merged_file',image_extraction,'in_file')

                # We do an affine registration
                mni_to_input_rigid = pe.Node(interface=niftyreg.RegAladin(verbosity_off_flag=True), 
                                             name='mni_to_input_rigid')
	        mni_to_input_rigid.inputs.flo_file = mni_template
                workflow.connect(image_extraction, 'out_file', mni_to_input_rigid, 'ref_file')

    	    mni_to_input_nonrigid = pe.Node(interface=niftyreg.RegF3D(verbosity_off_flag=True), 
                                            name='mni_to_input_non_rigid')
            mni_to_input_nonrigid.inputs.jl_val=0.001
            mni_to_input_nonrigid.inputs.be_val=0.01
            mni_to_input_nonrigid.inputs.maxit_val=1000
            mni_to_input_nonrigid.inputs.flo_file = mni_template
            workflow.connect(mni_to_input_rigid, 'aff_file', mni_to_input_nonrigid, 'aff_file')
            workflow.connect(image_extraction, 'out_file', mni_to_input_nonrigid, 'ref_file')

            # We apply the non-rigid transformation to the mask with reg_resample
            mask_resample  = pe.Node(interface = niftyreg.RegResample(), 
                                     name = 'mask_resample')
            mask_resample.inputs.inter_val = 'NN'
            mask_resample.inputs.flo_file = mni_template_mask
            workflow.connect(image_extraction, 'out_file', mask_resample, 'ref_file')
            workflow.connect(mni_to_input_nonrigid, 'cpp_file', mask_resample, 'trans_file')

    if workingOnSpinalCord:
	sc_motion_correction = niftk.diffusion.create_slice_wise_dwi_motion_correction(name='slice_wise_motion_correction',
					                                                output_dir=os.path.join(os.getcwd(),'noddi_mri_processing','sw_motion_correction'),
                                                                                        dwi_interp_type = 'CUB',
                                                                                        computeMask=computeMask,
                                                                                        precomputed_mask=precomputed_mask)

        sc_motion_correction.base_dir=os.path.join(os.getcwd(),'noddi_mri_processing')
    	sc_motion_correction.connect(nii_merge,'merged_file', sc_motion_correction.get_node('input_node'), 'in_dwi_4d_file')
        sc_motion_correction.connect(merge_bvalbvec,'merged_bval_file',sc_motion_correction.get_node('input_node'), 'in_bval_file')
        sc_motion_correction.connect(merge_bvalbvec,'merged_bvec_file',sc_motion_correction.get_node('input_node'), 'in_bvec_file')
    else:   
        # run FSL EDDY for eddy current correction
        eddy = pe.Node(interface=fsl.Eddy(), 
                       name="eddy")
        eddy.inputs.args='-v'
        workflow.add_nodes([eddy])
        workflow.connect(nii_merge,'merged_file',eddy,'in_file')
        workflow.connect(merge_bvalbvec,'merged_bval_file',eddy,'in_bval')
        workflow.connect(merge_bvalbvec,'merged_bvec_file',eddy,'in_bvec')
        workflow.connect([(nii_merge,eddy, [(('merged_file', _gen_index), 'in_index')])])
        workflow.connect([(input_node,eddy,[(('in_epi_readouttime', _gen_acqpfile), 'in_acqp')])])
        if computeMask:
            if useBET:
                workflow.connect(bet,'mask_file',eddy,'in_mask')
            else:
                workflow.connect(mask_resample,'res_file',eddy,'in_mask')
        else:
            eddy.inputs.in_mask=precomputed_mask
    
    # do NODDI
    noddi_fitting = pe.Node(interface = niftk.Noddi(), 
                            name = 'noddi_fitting')
    workflow.connect(input_node, 'in_noise_scaling_factor', noddi_fitting, 'noise_scaling_factor')
    if workingOnSpinalCord:
        workflow.connect(sc_motion_correction, 'output_node.dwis', noddi_fitting, 'in_dwis')
    else:
        workflow.connect(eddy, 'out_corrected', noddi_fitting, 'in_dwis')

    if workingOnSpinalCord:
        workflow.connect(sc_motion_correction, 'output_node.dwi_mask',noddi_fitting, 'in_mask')
    else:
        if computeMask:
            if useBET:
                workflow.connect(bet, 'mask_file', noddi_fitting, 'in_mask')
            else:
                workflow.connect(mask_resample,'res_file',noddi_fitting,'in_mask')
        else:
            noddi_fitting.inputs.in_mask=precomputed_mask

    workflow.connect(merge_bvalbvec, 'merged_bval_file', noddi_fitting, 'in_bvals')
    workflow.connect(merge_bvalbvec, 'merged_bvec_file', noddi_fitting, 'in_bvecs')
    
    # Create datasink for output
    data_sink = pe.Node(nio.DataSink(), 
                        name='data_output')
    data_sink.inputs.parameterization = False

    subsgen = pe.Node(interface = niu.Function(input_names = ['op_basename'], 
                                               output_names = ['substitutions'], 
                                               function = gen_substitutions), 
                                               name = 'replace_filenames')
    workflow.connect(input_node, 'in_subject_name', subsgen, 'op_basename')
    
    workflow.connect(subsgen, 'substitutions', data_sink, 'regexp_substitutions')
    workflow.connect(noddi_fitting, 'out_neural_density', data_sink, '@out_neural_density')
    workflow.connect(noddi_fitting, 'out_orientation_dispersion_index', data_sink, '@out_orientation_dispersion_index')
    workflow.connect(noddi_fitting, 'out_csf_volume_fraction', data_sink, '@out_csf_volume_fraction')
    workflow.connect(noddi_fitting, 'out_objective_function', data_sink, '@out_objective_function')
    workflow.connect(noddi_fitting, 'out_kappa_concentration', data_sink, '@out_kappa_concentration')
    workflow.connect(noddi_fitting, 'out_error', data_sink, '@out_error')
    workflow.connect(noddi_fitting, 'out_fibre_orientations_x', data_sink, '@out_fibre_orientations_x')
    workflow.connect(noddi_fitting, 'out_fibre_orientations_y', data_sink, '@out_fibre_orientations_y')
    workflow.connect(noddi_fitting, 'out_fibre_orientations_z', data_sink, '@out_fibre_orientations_z')    
    workflow.connect(input_node, 'in_result_dir', data_sink, 'base_directory')
    if workingOnSpinalCord:
        workflow.connect(sc_motion_correction, 'output_node.dwi_mask', data_sink, '@dwi_mask')
    else:
        if computeMask:
            if useBET: 
                workflow.connect(bet, 'mask_file', data_sink, '@dwi_mask')
            else:
                workflow.connect(mask_resample,'res_file', data_sink, '@dwi_mask')

    if workingOnSpinalCord:
        workflow.connect(sc_motion_correction, 'output_node.dwis', data_sink, '@corrected_dwis')
    else:
        workflow.connect(eddy, 'out_corrected', data_sink, '@corrected_dwis')
    workflow.connect(merge_bvalbvec, 'merged_bval_file', data_sink, '@bvals')
    workflow.connect(merge_bvalbvec, 'merged_bvec_file', data_sink, '@bvecs')
       
    return workflow

help_message = \
'------------------------------------------------------------------------------' + \
'-- Perform NODDI Model Fitting with pre-processing steps - NMR Research Unit -' + \
'------------------------------------------------------------------------------' + \
'Warning: Use script arguments -e, --epi-rot or EPI_READ_OUT_TIME environment ' + \
'variable to change the default epi read out time value in seconds. By default' + \
'is 0.028647. Use script arguments -n, --nsf or NOISE_SCALING_FACTOR environment '+ \
'variable to change the default noise scaling factor value. By default is 100. ' + \
'Set the noise scaling factor to 1 for spinal cord data because it usually has lower SNR.' + \
'------------------------------------------------------------------------------' + \
'Mandatory input data is the DICOM folder, remember to download the data using: ' + \
'getphilips -d B1234567-XYWZ-0ABCD '+ \
'------------------------------------------------------------------------------'
parser = argparse.ArgumentParser(description=help_message)
parser.add_argument('-i', '--input',
                    dest='dicom',
                    metavar='dicom',
                    type=str,
                    help='Dicom folder with all the files. Remember to download the data using: getphilips -d B1234567-XYWZ-0ABCD',
                    required=True)
parser.add_argument('-o', '--output_dir', 
                    dest='output_dir', 
                    type=str,
                    metavar='output_dir', 
                    help='Output directory containing the results.\n' + \
                    'Default is a directory called results',
                    default=os.path.abspath('results'), 
                    required=False)
parser.add_argument('-m', '--mask', 
                    dest='precomputed_mask', 
                    type=str,
                    metavar='precomputed_mask', 
                    help='Provided a precomputed mask',
                    required=False)
parser.add_argument('-e', '--epi_rot', 
                    dest='epi_rot', 
                    type=float,
                    metavar='epi_rot', 
                    help='EPI read out time value in seconds.\n' + \
                    'Default is 0.028647 or the value EPI_READ_OUT_TIME environtment variable.',
                    required=False)
parser.add_argument('-n', '--nsf', 
                    dest='noise_scaling_factor', 
                    type=float,
                    metavar='noise_scaling_factor', 
                    help='Noise scaling factor value.\n' + \
                    'Default is 100 or the value NOISE_SCALING_FACTOR environtment variable. Set the noise scaling factor to 1 for spinal cord data because it usually has lower SNR.', 
                    required=False)
parser.add_argument('-c', '--sc', 
                    dest='spinal_cord', 
                    metavar='spinal_cord', 
                    action='store_const',
                    help='Spinal cord DWI images.\n' + \
                    'Use it to calculate the NODDI fitting for spinal cord files that follow ZDTI filename pattern.',
                    default=False, 
                    const=True, 
                    required=False)
parser.add_argument('-p', '--pattern', 
                    dest='dti_file_pattern', 
                    type=str,
                    metavar='dti_file_pattern', 
                    help='File name pattern.\n' + \
                    'Use it to allow different names patterns to find the DWI images needed to calculate the NODDI fitting, by default PDTI (brain), for spinal cord normally is ZDTI.',
                    default='PDTI',  
                    required=False)
parser.add_argument('-s', '--subject_name', 
                    dest='subject_name', 
                    type=str,
                    metavar='subject_name', 
                    help='Subject name pattern.\n' + \
                    'Use it to change the prefix of the output files. By default: BXXXXX_noddi_odi.nii',
                    default=None,  
                    required=False)
parser.add_argument('-b', '--bet', 
                    dest='bet_mask', 
                    metavar='bet_mask', 
                    action='store_const',
                    help='Computing brain mask using BET.\n' + \
                    'Be aware that computing the brain mask with BET could be less precise. This flag is not used in the spinal cord NODDI fitting.',
                    default='False',
                    const=True,  
                    required=False)
parser.add_argument('-x', '--philips', 
                    dest='philips_patch', 
                    metavar='philips_patch', 
                    action='store_const',
                    help='DWI images are acquired using Philips normal gradients/protocol.\n' + \
                    'Be aware that this flag is so important. If you use it we are expecting that the DWI images include the B0 image and that they have a mean DWI intensity volume at the end. Fail to select or not this flag, when it is needed, will made a completely wrong NODDI fitting.',
                    default='False',
                    const=True,  
                    required=False)
parser.add_argument('-g', '--graph',
                    dest='graph',
                    help='Print a graph describing the node connections',
                    action='store_true',
                    default=False)

args = parser.parse_args()

# If we want to process different DWI files than the default
filename_pattern=args.dti_file_pattern
working_on_spinal_cord=False
working_using_philips_dwi_param=False
if args.spinal_cord:
    filename_pattern='ZDTI'
    working_on_spinal_cord=True

if args.philips_patch:
    working_using_philips_dwi_param=True

# If we want to use BET for skullstripping
use_BET_for_skullstripping=True
if args.bet_mask is None:
    use_BET_for_skullstripping=False
else:
    use_BET_for_skullstripping=args.bet_mask is True

result_dir = os.path.abspath(args.output_dir)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)     

input_subject_name = re.sub(r'-\d*-\d*', '',args.dicom)
if args.subject_name is None:
	input_subject_name = re.sub(r'-\d*-\d*', '',args.dicom)
else:
	input_subject_name = args.subject_name

if not os.path.exists(os.path.join(os.getcwd(),'noddi_mri_processing')):
    os.mkdir(os.path.join(os.getcwd(),'noddi_mri_processing'))   

# Specify how and where to save the log files
config.update_config({
                       'logging': {
                                  'log_directory': os.path.join(os.getcwd(),'noddi_mri_processing'),
                                  'log_to_file': True
                                  }
                     })
logging.update_logging(config)
config.enable_debug_mode()
iflogger = logging.getLogger('interface')

# We set up a default value for EPI
epi_readouttime=0.028647
if args.epi_rot is None:
    iflogger.info('Getting the EPI read out time from EPI_READ_OUT_TIME environtment variable')
    try:    
        epi_readouttime=float(os.environ['EPI_READ_OUT_TIME'])
    except KeyError:   
        iflogger.info('The environtment variable EPI_READ_OUT_TIME is not set up. Using the default value -> '+str(epi_readouttime))
else:
    epi_readouttime=float(args.epi_rot)

noise_scaling_factor=100
if args.noise_scaling_factor is None:
    iflogger.info('Getting the noise scaling factor from NOISE_SCALING_FACTOR environtment variable')
    try:    
        noise_scaling_factor=int(os.environ['NOISE_SCALING_FACTOR'])
    except KeyError:   
        iflogger.info('The environtment variable NOISE_SCALING_FACTOR is not set up. Using the default value -> '+str(noise_scaling_factor))
else:
    noise_scaling_factor=int(args.noise_scaling_factor)

calculate_mask=False
precomputed_mask=''
if args.precomputed_mask == None:
    calculate_mask=True
else:
    precomputed_mask = os.path.abspath(args.precomputed_mask)
    if not os.path.exists(precomputed_mask):
        iflogger.info('Precomputed mask not found at '+precomputed_mask)
    else:
        iflogger.info('Precomputed mask = '+precomputed_mask)

iflogger.info('DICOM directory = '+args.dicom)
iflogger.info('Subject name = '+input_subject_name)
iflogger.info('Output directory = '+result_dir)
iflogger.info('EPI read out time = '+str(epi_readouttime))
iflogger.info('Noise scaling factor = '+str(noise_scaling_factor))
iflogger.info('DTI filename pattern = '+filename_pattern)

if working_using_philips_dwi_param:
    iflogger.info('Data acquired using Philips DWI configuration - Taking into account that for processing DWI images')
else:
    iflogger.info('Data acquired using own house ION patch scanner configuration - Taking into account that for processing DWI images')

if working_on_spinal_cord:
    iflogger.info('Computing in the Spinal Cord')
else:
    if use_BET_for_skullstripping:
        iflogger.info('Using BET for skull-stripping')

if not os.path.isdir(os.path.abspath(args.dicom)):
    iflogger.info('ERROR: Input dicom directory doesn\'t exist: '+os.path.abspath(args.dicom))
    iflogger.info('The NODDI fitting hasn\'t been done.')
else:
    iflogger.info('Input dicom directory exist: '+os.path.abspath(args.dicom))
 
    noddi_fitting_ion = create_noddi_mri_processing_workflow(name='noddi_mri_processing', 
                                                         dti_filename_pattern = filename_pattern,
                                                         useBET=use_BET_for_skullstripping,
                                                         workingOnSpinalCord=working_on_spinal_cord,
                                                         working_using_philips_dwi_param=working_using_philips_dwi_param,
                                                         precomputed_mask=precomputed_mask,
                                                         computeMask=calculate_mask)
    noddi_fitting_ion.base_dir=os.getcwd()
    noddi_fitting_ion.base_output_dir='noddi_mri_processing'
    noddi_fitting_ion.inputs.input_node.in_dicom_folder=os.path.abspath(args.dicom)
    noddi_fitting_ion.inputs.input_node.in_epi_readouttime=epi_readouttime
    noddi_fitting_ion.inputs.input_node.in_noise_scaling_factor=noise_scaling_factor
    noddi_fitting_ion.inputs.input_node.in_result_dir=result_dir
    noddi_fitting_ion.inputs.input_node.in_subject_name=input_subject_name

    # output the graph if required
    if args.graph is True:
        niftk.base.generate_graph(workflow=noddi_fitting_ion)

    # Run the workflow
    qsubargs = '-l h_rt=02:00:00 -l tmem=2.9G -l h_vmem=2.9G -l vf=2.9G -l s_stack=10240 -j y -b y -S /bin/csh -V'
    niftk.base.run_workflow(workflow=noddi_fitting_ion,
                            qsubargs=qsubargs)

    iflogger.info('The NODDI fitting has been done. Have a nice day!')
