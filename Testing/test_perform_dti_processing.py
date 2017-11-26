#! /usr/bin/env python
import os
import sys
import utility

data_path = os.environ['NiftyPipe_DATA_PATH']
script_file = 'perform_dti_processing.py'
t1 = os.path.join(data_path, 't1.nii.gz')
dwi = os.path.join(data_path, 'diffusion', 'dwis_1.nii.gz')
bvals = os.path.join(data_path, 'diffusion', 'dwis_1.bval')
bvecs = os.path.join(data_path, 'diffusion', 'dwis_1.bvec')
mag = os.path.join(data_path, 'diffusion', 'fieldmap_m.nii.gz')
ph = os.path.join(data_path, 'diffusion', 'fieldmap_p.nii.gz')
outputs = ['dwis_1_b0.nii.gz',
           'dwis_1_dwis.nii.gz',
           'dwis_1_fa.nii.gz',
           'dwis_1_interslice_cc.png',
           'dwis_1_mask.nii.gz',
           'dwis_1_matrix_rot.png',
           'dwis_1_md.nii.gz',
           'dwis_1_predicted_tensors.nii.gz',
           'dwis_1_rgb.nii.gz',
           'dwis_1_t1_to_b0.txt',
           'dwis_1_tensor_residuals.nii.gz',
           'dwis_1_tensors.nii.gz',
           'dwis_1_v1.nii.gz']
result = 'dwis_1_tensors.nii.gz'
expected_result = os.path.join(data_path, 'expected_outputs', 'dwis_1_tensors.nii.gz')
cmd = script_file \
    + ' --remove_tmp ' \
    + ' -i ' + dwi \
    + ' -a ' + bvals \
    + ' -e ' + bvecs \
    + ' -t ' + t1 \
    + ' -m ' + mag \
    + ' -p ' + ph \
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
