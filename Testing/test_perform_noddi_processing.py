#! /usr/bin/env python
import os
import sys
import utility

print ('WARNING: the noddi output result and expected result have not been set for the test')
sys.exit(0)

data_path = os.environ['NiftyPipe_DATA_PATH']
script_file = 'perform_noddi_processing.py'
t1 = os.path.join(data_path, 't1.nii.gz')
dwi1 = os.path.join(data_path, 'diffusion', 'dwis_1.nii.gz')
bvals1 = os.path.join(data_path, 'diffusion', 'dwis_1.bval')
bvecs1 = os.path.join(data_path, 'diffusion', 'dwis_1.bvec')
dwi2 = os.path.join(data_path, 'diffusion', 'dwis_2.nii.gz')
bvals2 = os.path.join(data_path, 'diffusion', 'dwis_2.bval')
bvecs2 = os.path.join(data_path, 'diffusion', 'dwis_2.bvec')
mag = os.path.join(data_path, 'diffusion', 'fieldmap_m.nii.gz')
ph = os.path.join(data_path, 'diffusion', 'fieldmap_p.nii.gz')
outputs = ['']
result = ''
expected_result = os.path.join(data_path, 'expected_outputs', '')
cmd = script_file \
    + ' --remove_tmp ' \
    + ' -i ' + dwi1 + ' ' + dwi1 \
    + ' -a ' + bvals1 + ' ' + bvals1 \
    + ' -e ' + bvecs1 + ' ' + bvecs1 \
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
