#! /usr/bin/env python
import sys
import os
import utility

data_path = os.environ['NiftyPipe_DATA_PATH']
script_file = 'perform_gif_propagation.py'
t1 = os.path.join(data_path, 't1.nii.gz')
db = os.path.join(data_path, 'gif', 'db.xml')
outputs = ['t1_bias_corrected.nii.gz',
           't1_brain.nii.gz',
           't1_labels.nii.gz',
           't1_prior.nii.gz',
           't1_seg.nii.gz',
           't1_tiv.nii.gz',
           't1_volumes.csv']

result = 't1_labels.nii.gz'
expected_result = os.path.join(data_path, 'expected_outputs', 't1_labels.nii.gz')
cmd = script_file \
    + ' --remove_tmp ' \
    + ' -i ' + t1 \
    + ' -d ' + db \
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
