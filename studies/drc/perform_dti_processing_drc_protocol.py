#! /usr/bin/env python
import argparse
import textwrap
import os
import sys
import nipype.interfaces.utility as niu
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
import nipype.interfaces.dcm2nii as mricron
from niftypipe.workflows.dmri.niftyfit_tensor_preprocessing import create_diffusion_mri_processing_workflow
from niftypipe.interfaces.niftk.base import (generate_graph, run_workflow)
import glob
import subprocess
from nipype.interfaces.fsl.base import FSLCommand
from nipype.interfaces.fsl.base import FSLCommandInputSpec
from nipype.interfaces.base import (TraitedSpec, File, Directory,
                                    traits, InputMultiPath, BaseInterface,
                                    BaseInterfaceInputSpec)


midas_path = '/var/lib/midas/data/'

database_list = ['fidelity',
                 'ppadti',
                 'adni',
                 'adni2',
                 'adnigo',
                 'adni-main',
                 'avid2',
                 'elan',
                 'elan3',
                 'genada',
                 'genfi',
                 'genfi2',
                 'genman',
                 'hdni-prep',
                 'inddex',
                 'jai351',
                 'janssen',
                 'lha1946',
                 'lundbeck',
                 'lundbeck-qa',
                 'mci-study',
                 'miriad',
                 'neurogrid',
                 'new-training',
                 'opelan',
                 'owyeth',
                 'pelan',
                 'pelan251',
                 'pfizer',
                 'radar',
                 'rebecca',
                 'sam',
                 'sanofi',
                 'statin',
                 'wyeth',
                 'yoad']


class Midas2NiiInputSpec(FSLCommandInputSpec):
    in_file = File(argstr="%s", exists=True, mandatory=True, position=2,
                   desc="Input image filename ** WITH .IMG EXTENSION **")
    out_file = File(argstr="%s", position=-2, name_source=['in_file'], name_template='%s_nii',
                    desc="Output file")


class Midas2NiiOutputSpec(TraitedSpec):
    out_file = File(desc="Output nii image file")


class Midas2Nii(FSLCommand):
    """
    Converts MIDAS Analyse formatted images into normal NIFTI ones.
    The input needs to be the .img file and the header .hdr needs to be present in the same directory

    Example
    --------
    from midas2nii import Midas2Nii
    converter = Midas2Nii()
    converter.inputs.in_file = "030583-T1.img"
    converter.run()
    """
    _cmd = "/var/drc/software/32bit/nifti-midas/midas2nii.sh"
    _suffix = "_nii"
    input_spec = Midas2NiiInputSpec
    output_spec = Midas2NiiOutputSpec
    _output_type = 'NIFTI'


def convert_midas2_dicom(midas_code, midas_dirs):
    # Check if 4 or 5 digits long
    if len(midas_code) == 4:
        midas_code = "0" + midas_code

    midas_ims = list()
    # look through various database paths to find a corresponding midas image
    for test_dir in midas_dirs:
        files = glob.glob(os.path.normpath(test_dir) + '/images/ims-study/' + midas_code + "-00*-1.hdr")
        if len(files) > 0:
            midas_ims.append(files[0])
            break
    if len(midas_ims) is 0:
        print "NO FILE FOUND"
        return None
    print 'file from search is: ', str(midas_ims)

    # call getdicompath.sh to find the first dicom file
    command = 'sh /var/lib/midas/pkg/x86_64-unknown-linux-gnu/dicom-to-midas/getdicompath.sh ' + midas_ims[0]

    dicom_file = subprocess.check_output(command.split())
    dicom_dir = os.path.dirname(dicom_file)
    print "Dicom directory is: " + dicom_dir
    # return the directory
    return dicom_dir


class Midas2DicomInputSpec(BaseInterfaceInputSpec):
    midas_code = traits.String(mandatory=True, desc="4/5 digit midas code")
    midas_dirs = InputMultiPath(mandatory=True, desc="Midas database directories to search")


class Midas2DicomOutputSpec(TraitedSpec):
    dicom_dir = Directory(exists=True, desc="Dicom directory path of inputted midas image")


class Midas2Dicom(BaseInterface):
    """

    Examples
    --------

    """
    input_spec = Midas2DicomInputSpec
    output_spec = Midas2DicomOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['dicom_dir'] = self.dicom_dir
        return outputs

    def _run_interface(self, runtime):
        self.dicom_dir = convert_midas2_dicom(self.inputs.midas_code, self.inputs.midas_dirs)
        return runtime


def find_and_convert_midas_T1(midas_code, midas_databases):
    # Check if 4 or 5 digits long
    if len(midas_code) == 4:
        midas_code = "0" + midas_code

    midas_t1 = ''

    # look through various database paths to find a corresponding midas image
    for db in midas_databases:
        path_to_search = os.path.join(os.path.abspath(db), 'images/ims-study')
        files = glob.glob(path_to_search + os.path.sep + midas_code + '-012-1.img*')
        if len(files) > 0:
            midas_t1 = files[0]
            break
    if len(midas_t1) is 0:
        print "NO FILE FOUND"
        return None
    print 'file from search is: ', str(midas_t1)

    # call midas2nii.sh to find the first dicom file
    nifti_file = midas_code + '_t1.nii'
    command = 'sh /var/drc/software/32bit/nifti-midas/midas2nii.sh ' \
              + midas_t1 + ' ' + nifti_file + '; gzip ' + nifti_file
    os.system(command)
    ret = os.path.join(os.getcwd(), nifti_file + '.gz')
    print "converted file is : " + ret
    return midas_t1, os.path.abspath(ret)


def find_and_convert_midas_brain(midas_code, midas_databases, midas_t1):
    # Check if 4 or 5 digits long
    if len(midas_code) == 4:
        midas_code = "0" + midas_code

    midas_brain = ''

    # look through various database paths to find a corresponding midas image
    for db in midas_databases:
        path_to_search = os.path.join(os.path.abspath(db), 'regions/brain')
        files = glob.glob(path_to_search + os.path.sep + '*' + midas_code + '*')
        if len(files) > 0:
            midas_brain = files[0]
            break
    if len(midas_brain) is 0:
        print "ERROR: NO BRAIN FILE FOUND in databases"
        return None, None
    print 'file from search is: ', midas_brain

    command = '/var/drc/software/64bit/midas/bin/makemask ' + midas_t1 + ' ' + midas_brain + ' brain'
    os.system(command)
    if not os.path.exists('brain.img'):
        print 'ERROR: makemask (/var/drc/software/64bit/midas/bin/makemask) failed'
        return None, None

    nifti_file = midas_code + '_brain.nii'
    command = 'sh /var/drc/software/32bit/nifti-midas/midas2nii.sh brain.img ' + nifti_file + '; gzip ' + nifti_file
    os.system(command)
    ret = os.path.join(os.getcwd(), nifti_file + '.gz')
    print "converted file is : " + ret
    return midas_brain, os.path.abspath(ret)


class RetrieveAndConvertMidasBrainMaskInputSpec(BaseInterfaceInputSpec):
    midas_code = traits.String(mandatory=True, desc="4/5 digit midas code")
    midas_databases = InputMultiPath(mandatory=True, desc="Midas database directories to search")
    t1_image = File(mandatory=True, desc="Corresponding T1 from DICOMs")


class RetrieveAndConvertMidasBrainMaskOutputSpec(TraitedSpec):
    out_file = File(desc="Output Brain Mask in the geometry of the input T1")


class RetrieveAndConvertMidasBrainMask(BaseInterface):
    """

    Examples
    --------

    """
    input_spec = RetrieveAndConvertMidasBrainMaskInputSpec
    output_spec = RetrieveAndConvertMidasBrainMaskOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.out_file
        return outputs

    def _run_interface(self, runtime):
        midas_t1, unused = find_and_convert_midas_T1(self.inputs.midas_code,
                                                     self.inputs.midas_databases)
        unused, brainmask = find_and_convert_midas_brain(self.inputs.midas_code,
                                                         self.inputs.midas_databases,
                                                         midas_t1)
        command = 'fslswapdim ' + brainmask + ' -z -y -x ' + brainmask
        os.system(command)        
        command = 'fslcpgeom ' + self.inputs.t1_image + ' ' + brainmask
        os.system(command)
        command = 'fslmaths ' + brainmask + ' -bin ' + brainmask
        os.system(command)
        self.out_file = os.path.abspath(brainmask)
        return runtime


def find_and_merge_dwi_data(input_bvals, input_bvecs, input_files):
    import os
    import glob
    import sys
    import errno
    from niftypipe.workflows.dmri.niftyfit_tensor_preprocessing import merge_dwi_function

    input_path = os.path.dirname(input_files[0])
    dwis_files = []

    for bvals_file in input_bvals:
        if ('iso' in bvals_file) and \
                (('001.bval' in bvals_file) or ('001A.bval' in bvals_file) or ('001B.bval' in bvals_file)):
            dwi_base, _ = os.path.splitext(bvals_file)
            dwi_file = dwi_base + '.nii.gz'
            if not os.path.exists(dwi_file):
                dwi_file = dwi_base + '.nii'
            dwis_files.append(dwi_file)
        else:
            dwi_base, _ = os.path.splitext(bvals_file)
            input_bvals.remove(dwi_base + '.bval')
            input_bvecs.remove(dwi_base + '.bvec')

    dwis, bvals, bvecs = merge_dwi_function(dwis_files, input_bvals, input_bvecs)

    fms = glob.glob(input_path + os.sep + '*fieldmap*.nii*')
    fms.sort()
    if len(fms) == 0:
        print 'I/O Could not find any field map image in ', input_path, ', exiting.'
        sys.exit(errno.EIO)
    if len(fms) < 2:
        print 'I/O Field Map error: either the magnitude or phase is missing in ', input_path, ', exiting.'
        sys.exit(errno.EIO)
    if len(fms) > 2:
        print 'I/O Field Map warning: there are more field map images than expected in ', \
            input_path, \
            ', \nAssuming the first two are relevant...'
    fmmag = fms[0]
    fmph = fms[1]

    t1s = glob.glob(input_path + os.sep + 'o*MPRAGE*.nii*')
    if len(t1s) == 0:
        t1s = glob.glob(input_path + os.sep + '*MPRAGE*.nii*')
    if len(t1s) == 0:
        print 'I/O Could not find any MPRAGE image in ', input_path, ', exiting.'
        sys.exit(errno.EIO)
    if len(t1s) > 1:
        print 'I/O warning: there is more than 1 MPRAGE image in ', input_path, \
            ', \nAssuming the first one is relevant...'
    t1 = t1s[0]

    return dwis, bvals, bvecs, fmmag, fmph, t1


# Create a drc diffusion pipeline
def create_drc_diffusion_processing_workflow(midas_code,
                                             output_dir,
                                             dwi_interp_type='CUB',
                                             susceptibility_params=[34.56, 2.46, '-y'],
                                             log_data=False,
                                             resample_t1=False,
                                             rigid_only=False,
                                             retrieve_mask=True):

    database_paths = [midas_path + db for db in database_list]

    r = create_diffusion_mri_processing_workflow(name='dmri_workflow',
                                                 t1_mask_provided=retrieve_mask,
                                                 resample_in_t1=resample_t1,
                                                 log_data=log_data,
                                                 susceptibility_correction=True,
                                                 in_susceptibility_params=susceptibility_params,
                                                 dwi_interp_type=dwi_interp_type,
                                                 wls_tensor_fit=False,
                                                 rigid_only=rigid_only)

    infosource = pe.Node(niu.IdentityInterface(fields=['subject_id']), name='infosource')
    infosource.iterables = ('subject_id', midas_code)

    midas2dicom = pe.Node(Midas2Dicom(midas_dirs=database_paths), name='m2d')
    r.connect(infosource, 'subject_id', midas2dicom, 'midas_code')

    dg = pe.Node(nio.DataGrabber(template='*', sort_filelist=False, outfields=['dicom_files']), name='dg')
    r.connect(midas2dicom, 'dicom_dir', dg, 'base_directory')

    dcm2nii = pe.Node(interface=mricron.Dcm2nii(args='-d n', gzip_output=True, anonymize=True,
                                                reorient=True, reorient_and_crop=True), name='dcm2nii')
    r.connect(dg, 'dicom_files', dcm2nii, 'source_names')

    find_and_merge_dwis = pe.Node(
        interface=niu.Function(input_names=['input_bvals', 'input_bvecs', 'input_files'],
                               output_names=['dwis', 'bvals', 'bvecs', 'fieldmapmag', 'fieldmapphase', 't1'],
                               function=find_and_merge_dwi_data),
        name='find_and_merge_dwis')
    r.connect(dcm2nii, 'converted_files', find_and_merge_dwis, 'input_files')
    r.connect(dcm2nii, 'bvals', find_and_merge_dwis, 'input_bvals')
    r.connect(dcm2nii, 'bvecs', find_and_merge_dwis, 'input_bvecs')

    dwi_renamer = pe.Node(interface=niu.Rename(format_string='%(subject_id)s', keep_ext=True), name='dwi_renamer')
    r.connect(find_and_merge_dwis, 'dwis', dwi_renamer, 'in_file')
    r.connect(infosource, 'subject_id', dwi_renamer, 'subject_id')

    r.connect(dwi_renamer, 'out_file', r.get_node('input_node'), 'in_dwi_4d_file')
    r.connect(find_and_merge_dwis, 'bvals', r.get_node('input_node'), 'in_bval_file')
    r.connect(find_and_merge_dwis, 'bvecs', r.get_node('input_node'), 'in_bvec_file')
    r.connect(find_and_merge_dwis, 'fieldmapmag', r.get_node('input_node'), 'in_fm_magnitude_file')
    r.connect(find_and_merge_dwis, 'fieldmapphase', r.get_node('input_node'), 'in_fm_phase_file')
    r.connect(find_and_merge_dwis, 't1', r.get_node('input_node'), 'in_t1_file')
    if retrieve_mask is True:
        mask_retriever = pe.Node(interface=RetrieveAndConvertMidasBrainMask(midas_databases=database_paths),
                                 name='mask_retriever')
        r.connect(find_and_merge_dwis, 't1', mask_retriever, 't1_image')
        r.connect(infosource, 'subject_id', mask_retriever, 'midas_code')
        r.connect(mask_retriever, 'out_file', r.get_node('input_node'), 'in_t1_mask_file')

    ds = pe.Node(nio.DataSink(parameterization=False, base_directory=output_dir), name='ds')
    r.connect(r.get_node('renamer'), 'out_file', ds, '@outputs')
    r.connect(r.get_node('reorder_transformations'), 'out', ds, 'transformations')

    return r


help_message = textwrap.dedent('''
Perform Diffusion Model Fitting with pre-processing steps. 
Mandatory Input is the 4/5 MIDAS code from which the DWIs, bval bvecs, 
as well as a T1 image are extracted for reference space. 
The Field maps are provided so susceptibility correction is applied. 

Values to use for the susceptibility parameters:

## DRC ## (--ped=-y --etd=2.46 --rot=34.56) and 
## 1946 ## (--ped=-y --etd=2.46 --rot=25.92). 

Note that these values are indicative.
''')

parser = argparse.ArgumentParser(description=help_message)

parser.add_argument('-m', '--midas_code',
                    dest='midas_code',
                    nargs='+',
                    required=True,
                    help='MIDAS code of the subject image')
parser.add_argument('-o', '--output',
                    dest='output',
                    metavar='output',
                    help='Result directory where the output data is to be stored',
                    required=False,
                    default='results')
parser.add_argument('-r', '--resample-t1',
                    dest='resample_t1',
                    help='Resample the outputs in the T1 space',
                    required=False,
                    action='store_true')
parser.add_argument('--interpolation',
                    dest='interpolation',
                    help='Interpolation options CUB (default) or LIN',
                    required=False,
                    default='CUB')
parser.add_argument('-g', '--graph',
                    dest='graph',
                    help='Print a graph describing the node connections',
                    action='store_true',
                    default=False)
parser.add_argument('--rot',
                    dest='rot',
                    type=float,
                    metavar='rot',
                    help='Diffusion Read-Out time used for susceptibility correction\n' +
                         'Default is 34.56',
                    default=34.56,
                    required=False)
parser.add_argument('--etd',
                    dest='etd',
                    type=float,
                    metavar='etd',
                    help='Echo Time difference used for susceptibility correction\n' +
                         'Default is 2.46',
                    default=2.46,
                    required=False)
parser.add_argument('--ped',
                    nargs='?',
                    const=None,
                    choices=[Q for x in ['x', 'y', 'z'] for Q in (x, '-' + x)],
                    dest='ped',
                    type=str,
                    metavar='ped',
                    help='Phase encoding direction used for susceptibility correction (x, y or z)\n' +
                         '--ped=val form must be used for -ve indices' +
                         'Default is the -y direction (-y)',
                    default='-y',
                    required=False)
parser.add_argument('--rigid',
                    dest='rigid_only',
                    action='store_true',
                    help='Only use rigid registration for DWI (no eddy current correction)',
                    required=False)
parser.add_argument('--create_mask',
                    dest='create_mask',
                    action='store_true',
                    help='Create a T1 mask instead of retrieving it from the DRC database',
                    required=False)
parser.add_argument('-x', '-y', '-z',
                    dest='pedwarn',
                    help=argparse.SUPPRESS,
                    required=False,
                    action='store_true')

args = parser.parse_args()

if args.ped is None:
    print 'argument --ped: expected one argument, make sure to use --ped='
    sys.exit(1)
if args.pedwarn:
    print 'One of -x, -y or -z found, did you mean --ped=-x, --ped=-y --ped=-z?'
    sys.exit(1)

result_dir = os.path.abspath(args.output)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

# Check if some images have already been processed, and if so, remove them from the analysis
codes = []
for code in args.midas_code:
    result_file = os.path.join(result_dir, code + '_tensors.nii.gz')
    if not os.path.exists(result_file):
        codes.append(code)

workflow = create_drc_diffusion_processing_workflow(codes, result_dir,
                                                    dwi_interp_type=args.interpolation,
                                                    susceptibility_params=[args.rot, args.etd, args.ped],
                                                    log_data=False,
                                                    resample_t1=args.resample_t1,
                                                    rigid_only=args.rigid_only,
                                                    retrieve_mask=not args.create_mask)
workflow.base_dir = result_dir

if args.graph is True:
    generate_graph(workflow=workflow)
    exit(0)

qsubargs = '-l h_rt=02:00:00 -l tmem=2.9G -l h_vmem=2.9G -l vf=2.9G -l s_stack=10240 -j y -b y -S /bin/csh -V'
run_workflow(workflow=workflow,
             qsubargs=qsubargs)
