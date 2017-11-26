#! /usr/bin/env python
import sys
import os
import utility

data_path = os.environ['NiftyPipe_DATA_PATH']
script_file = 'perform_restingstate_preprocessing.py'
fmri = os.path.join(data_path, 'fmri', 'fmri.nii.gz')
t1 = os.path.join(data_path, 'fmri', 't1.nii.gz')
par = os.path.join(data_path, 'fmri', 't1_parcellation.nii.gz')
seg = os.path.join(data_path, 'fmri', 't1_segmentation.nii.gz')
outputs = ['atlas_fmri.nii.gz',
           'fmri_motion_corrected.nii.gz',
           'fmri_pp.nii.gz',
           'seg_csf_fmri.nii.gz',
           'seg_gm_fmri.nii.gz',
           'seg_wm_fmri.nii.gz',
           'fmri_qc.png']
result = 'seg_gm_fmri.nii.gz'
expected_result = os.path.join(data_path, 'expected_outputs', 'fmri_seg_gm_fmri.nii.gz')
cmd = script_file \
    + ' --remove_tmp ' \
    + ' -i ' + fmri \
    + ' -t ' + t1 \
    + ' -s ' + seg \
    + ' -p ' + par \
    + ' -o .'

if not utility.check_files([script_file, expected_result]):
    sys.exit(1)

# remove the potentially already existing result file:
if os.path.exists(os.path.abspath(result)):
    os.remove(os.path.abspath(result))

print cmd
os.system(cmd)
if not utility.check_files(outputs):
    sys.exit(1)
if not utility.compare_images(result, expected_result, 0.01, 0.01):
    print('output file %s is different from expected output %s, test failed' % (result, expected_result))
    sys.exit(1)
sys.exit(0)
