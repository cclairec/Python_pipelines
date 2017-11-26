# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
    Simple interface for Plotting statistical results
"""

import csv
import os
import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import nibabel as nib


def read_stats_csv(filename, no_lines, no_cols):
    stats = np.zeros((no_lines, no_cols))
    with open(filename, 'rU') as f:
        reader = csv.reader(f)
        row_index = 0
        for row in reader:
            for i in range(no_cols):
                stats[row_index, i] = row[i]
            row_index += 1
            # print stats
    return stats


def get_image_4th_dimension(in_file):
    image = nib.load(in_file).get_data()
    return image.shape[3]


def plot_results(result_base_directory, number_of_repeats, result_directory_prefix):

    # the base directory of the results sunk from nipype
    # the individual diffusion simulation outputs:
    simulation_output_directories = [os.path.basename(f)
                                     for f in glob.glob(os.path.join(result_base_directory,
                                                                     result_directory_prefix + '*'))]

    # How many directions were used in this simulation?
    # Find out by opening the first DWI
    dwis_example_file = os.path.join(result_base_directory, simulation_output_directories[0],
                                     'dwifit__syn_noisy_merged_maths.nii.gz')
    number_of_directions = get_image_4th_dimension(dwis_example_file)

    # How many variants, i.e. run-types, do we have per repeat?
    # Find out with the number of directories per repeat
    # This is assuming there is no other directories present there...
    no_variants = len(simulation_output_directories) / number_of_repeats
    variant_name_list = []

    # the number of labels is arbitrary there..
    # no_labels = 49  # todo change that to dynamically discover the amount of labels...
    no_labels = 2

    # the labels of interest are the following:
    # labels = ['genu', 'body', 'splenium', 'fornix', 'cingulum (l)', 'cingulum (r)']
    labels = ['brain']
    # labels_indices = [3, 4, 5, 6, 36, 35]
    labels_indices = [1]

    # Instantiate the resulting difference tables
    dwi_summary = np.zeros((no_variants, len(labels_indices), number_of_repeats * number_of_directions))
    proc_residual_summary = np.zeros((no_variants, len(labels_indices), number_of_repeats * number_of_directions))
    tensor_summary = np.zeros((no_variants, len(labels_indices), number_of_repeats))
    trans_summary = np.zeros((no_variants, number_of_repeats * number_of_directions))

    variant_index = 0
    for variant_log in ['_log', '']:
        for variant_interpolation in ['_LIN', '_CUB']:

            # This gives us the variant we want to test, we need to load the statistics from each of the
            # iterations, and compile them together, make some empty numpy arrays

            # instantiate the difference arrays
            dwi_difference_results = np.zeros((number_of_directions * number_of_repeats * no_labels, 4))
            tensor_residual_results = np.zeros((number_of_directions * number_of_repeats * no_labels, 4))
            transformation_difference_results = np.zeros((number_of_directions * number_of_repeats, 1))
            tensor_difference_res = np.zeros((number_of_repeats * no_labels, 4))

            # iterate over the repeats and fill the difference arrays
            for itr in range(number_of_repeats):
                # find the corresponding result directory
                results_dir = result_base_directory + os.sep + result_directory_prefix + variant_log \
                  + variant_interpolation + '_' + str(itr)
                print results_dir
                # find the relative offset for each result table
                x_offset = range(number_of_directions * itr * no_labels,
                                 number_of_directions * (itr + 1) * no_labels)
                transformations_offset = range(number_of_directions * itr,
                                               number_of_directions * (itr + 1))
                tensor_offset = range(itr * no_labels, (itr + 1) * no_labels)

                # fill the tables with the results in the csv
                dwi_difference_results[x_offset, :] = \
                    read_stats_csv(results_dir + '/dwi_stats.csv',
                                   number_of_directions * no_labels, 4)
                tensor_residual_results[x_offset, :] = \
                    read_stats_csv(results_dir + '/proc_residual_stats.csv',
                                   number_of_directions * no_labels, 4)
                transformation_difference_results[transformations_offset, :] = \
                    read_stats_csv(results_dir + '/affine_stats.csv',
                                   number_of_directions, 1)
                tensor_difference_res[tensor_offset, :] = \
                    read_stats_csv(results_dir + '/tensor_stats.csv',
                                   no_labels - 1, 4)
            # return dwi_difference_results, tensor_residual_results, transformation_difference_results, tensor_difference_res
            # Once we've accumulated all the results into an array,
            # find the indices of the rois we are interested in
            roi_index = 0
            for roi in labels_indices:
                indices = dwi_difference_results[:, 0] == roi
                dwi_summary[variant_index, roi_index, :] = \
                    dwi_difference_results[indices, 1] / dwi_difference_results[indices, 3]
                proc_residual_summary[variant_index, roi_index, :] = \
                    tensor_residual_results[indices, 1] / tensor_residual_results[indices, 3]
                indices = tensor_difference_res[:, 0] == roi
                tensor_summary[variant_index, roi_index, :] = \
                    tensor_difference_res[indices, 1] / tensor_difference_res[indices, 3]
                roi_index += 1
            trans_summary[variant_index, :] = transformation_difference_results[:, 0]

            # print dwi_res[range(number_of_directions*itr*no_labels, number_of_directions*(itr+1)*no_labels),:]

            variant_name_list.append(variant_log + variant_interpolation)
            variant_index += 1

    return dwi_summary, proc_residual_summary, tensor_summary, trans_summary
    # sys.exit()

    row_no = 3  # organisation for the graphs
    col_no = 2  # organisation for the graphs

    plt.figure()
    for i in range(len(labels_indices)):
        ax = plt.subplot(row_no, col_no, (i % (row_no * col_no)) + 1)
        ax.boxplot(dwi_summary[:, i, :].transpose())
        plt.setp(ax, xticklabels=variant_name_list)
        plt.title('MSE of forward model ' + labels[i])

    plt.figure()
    for i in range(len(labels_indices)):
        ax = plt.subplot(row_no, col_no, (i % (row_no * col_no)) + 1)
        ax.boxplot(proc_residual_summary[:, i, :].transpose())
        plt.setp(ax, xticklabels=variant_name_list)
        plt.title('MSE of forward model ' + labels[i])

    plt.figure()
    ax = plt.subplot(111)
    ax.boxplot(trans_summary[range(4), :].transpose())
    plt.setp(ax, xticklabels=(variant_name_list[0:4]))
    plt.title('log trans l2 ' + labels[len(labels_indices)])

    plt.figure()
    for i in range(len(labels_indices)):
        ax = plt.subplot(row_no, col_no, (i % (row_no * col_no)) + 1)
        ax.boxplot(tensor_summary[:, i, :].transpose())
        plt.setp(ax, xticklabels=variant_name_list)
        plt.title('log l2 tensor ' + labels[i])

    plt.figure()
    for i in range(len(labels_indices)):
        plt.subplot(row_no, col_no, (i % (row_no * col_no)) + 1)
        for j in range(number_of_repeats):
            # Use black for the original data MSE and red for the processed data MSE
            plt.plot(tensor_summary[:, i, j],
                     np.mean(dwi_summary[:, i, range(j * number_of_directions, (j + 1) * number_of_directions)], 1),
                     'xk',
                     markersize=10)
            plt.plot(tensor_summary[:, i, j],
                     np.mean(
                         proc_residual_summary[:, i, range(j * number_of_directions, (j + 1) * number_of_directions)],
                         1),
                     'xr',
                     markersize=10)
            plt.xlabel('Tensor distance (L2 norm log tensor)')
            plt.ylabel('MSE')
            plt.tight_layout()

    # Do the same plot again, but don't separate out by different ROIs (easier to see trends)
    plt.figure()
    x_total = np.zeros((no_variants * len(labels_indices) * number_of_repeats))
    y1_total = np.zeros((no_variants * len(labels_indices) * number_of_repeats))
    y2_total = np.zeros((no_variants * len(labels_indices) * number_of_repeats))

    ind = 0
    for i in range(len(labels_indices)):
        for j in range(number_of_repeats):
            # Use black for the original data MSE and red for the processed data MSE
            plt.plot(tensor_summary[:, i, j],
                     np.mean(dwi_summary[:, i, range(j * number_of_directions, (j + 1) * number_of_directions)], 1),
                     'xk',
                     markersize=10)
            plt.plot(tensor_summary[:, i, j],
                     np.mean(
                         proc_residual_summary[:, i, range(j * number_of_directions, (j + 1) * number_of_directions)],
                         1),
                     'xr',
                     markersize=10)
            tmp1 = np.mean(dwi_summary[:, i, range(j * number_of_directions, (j + 1) * number_of_directions)], 1)
            tmp2 = np.mean(proc_residual_summary[:, i, range(j * number_of_directions, (j + 1) * number_of_directions)],
                           1)
            for k in range(no_variants):
                x_total[ind] = tensor_summary[k, i, j]
                y1_total[ind] = tmp1[k]
                y2_total[ind] = tmp2[k]
                ind += 1
            plt.xlabel('Tensor error (L2 norm log tensor)')
            plt.ylabel('MSE of forward model')
            plt.legend(['Original data', 'Processed data'])
            plt.tight_layout()
    # Run a linear regression of the error of the tensor with respect to the original data MSE
    lr1 = ss.linregress(x_total[np.isnan(x_total) is False],
                        y1_total[np.isnan(x_total) is False])
    # and the processed data
    lr2 = ss.linregress(x_total[np.isnan(x_total) is False],
                        y2_total[np.isnan(x_total) is False])
    print "The slope of the observed data MSE with respect to the tensor error is " + str(
        lr1[0]) + " with a p-value of " + str(lr1[3])
    print "The slope of the processed data MSE with respect to the tensor error is " + str(
        lr2[0]) + " with a p-value of " + str(lr2[3])

    plt.figure()
    x_total = np.zeros((no_variants * number_of_directions * number_of_repeats))
    y1_total = np.zeros((no_variants * number_of_directions * number_of_repeats))
    y2_total = np.zeros((no_variants * number_of_directions * number_of_repeats))

    ind = 0
    # Plot the transformation error against the data likelihoods
    for j in range(number_of_repeats):
        x = trans_summary[:, range(j * number_of_directions, (j + 1) * number_of_directions)].flatten()
        y1 = np.mean(dwi_summary[:, :, range(j * number_of_directions, (j + 1) * number_of_directions)], 1).flatten()
        y2 = np.mean(proc_residual_summary[:, :, range(j * number_of_directions, (j + 1) * number_of_directions)],
                     1).flatten()
        plt.plot(x, y1, 'xk', markersize=10)
        plt.plot(x, y2, 'xr', markersize=10)
        for k in range(no_variants * number_of_directions):
            x_total[ind] = x[k]
            y1_total[ind] = y1[k]
            y2_total[ind] = y2[k]
            ind += 1
        plt.xlabel('Transformation error (L2 norm log matrix)')
        plt.ylabel('MSE of the forward model')
        plt.legend(['Original data', 'Processed data'])

    # Run a linear regression of the error of the tensor with respect to the original data MSE
    lr1 = ss.linregress(x_total[np.isnan(x_total) is False], y1_total[np.isnan(x_total) is False])
    # and the processed data
    lr2 = ss.linregress(x_total[np.isnan(x_total) is False], y2_total[np.isnan(x_total) is False])
    print "The slope of the observed data MSE with respect to the transformation error is " + str(
        lr1[0]) + " with a p-value of " + str(lr1[3])
    print "The slope of the processed data MSE with respect to the transformation error is " + str(
        lr2[0]) + " with a p-value of " + str(lr2[3])

    plt.show()
