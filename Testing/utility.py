# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import nibabel as nib
import numpy as np
import os


def check_files(in_files):
    for f in in_files:
        print ('checking file: %s' % f)
        if not os.path.exists(f):
            if not any([os.path.exists(os.path.join(p, f)) for p in os.environ['PATH'].split(os.pathsep)]):
                print('file %s is missing, test failed' % f)
                return False
    return True


def compare_images(file1, file2, change_mean_error_marging=0.01, change_std_error_marging=0.01):
    in_img1 = nib.load(file1).get_data().ravel()
    in_img2 = nib.load(file2).get_data().ravel()
    valid_ids = in_img1 ** 2 > 0
    difference = (in_img2 - in_img1)
    percentile_change = np.divide(difference[valid_ids], in_img1[valid_ids])
    data_mean = float(np.mean(percentile_change))
    data_std = float(np.std(percentile_change))
    data_min = np.min(percentile_change)
    data_max = np.max(percentile_change)

    print('%s: \t changes: %2.2f%% +- %2.2f%% \t range: [%2.2f%% -- %2.2f%%]' %
          (file1, 100*data_mean, 100*data_std, 100*data_min, 100*data_max))

    if np.abs(data_mean) > change_mean_error_marging:
        return False
    if data_std > change_std_error_marging:
        return False

    return True


def compare_tables(file1, file2, change_mean_error_marging=0.01, change_std_error_marging=0.01, delimiter=','):
    in_img1 = np.genfromtxt(file1, delimiter=delimiter).ravel()
    in_img2 = np.genfromtxt(file2, delimiter=delimiter).ravel()
    valid_ids = in_img1 ** 2 > 0
    difference = (in_img2 - in_img1)
    percentile_change = np.divide(difference[valid_ids], in_img1[valid_ids])
    data_mean = float(np.mean(percentile_change))
    data_std = float(np.std(percentile_change))
    data_min = np.min(percentile_change)
    data_max = np.max(percentile_change)

    print('%s: \t changes: %2.2f%% +- %2.2f%% \t range: [%2.2f%% -- %2.2f%%]' %
          (file1, 100*data_mean, 100*data_std, 100*data_min, 100*data_max))

    if np.abs(data_mean) > change_mean_error_marging:
        return False
    if data_std > change_std_error_marging:
        return False

    return True
