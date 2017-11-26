#! /usr/bin/env python

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def gen_substitutions(subject_id):
    subs = [('graph', subject_id + '_hippocampus_volumes'), ('profile_1', subject_id + '_profile_left'),
            ('profile_2', subject_id + '_profile_right')]
    return subs


parser = argparse.ArgumentParser(description='Gather all profiles into a single graph')

parser.add_argument('-i11', '--input_profiles_11',
                    dest='input_profiles_11',
                    metavar='input_profiles_11',
                    help='Input profiles 11',
                    nargs='+',
                    required=True)
parser.add_argument('-i12', '--input_profiles_12',
                    dest='input_profiles_12',
                    metavar='input_profiles_12',
                    help='Input profiles 12',
                    nargs='+',
                    required=True)
parser.add_argument('-i21', '--input_profiles_21',
                    dest='input_profiles_21',
                    metavar='input_profiles_21',
                    help='Input profiles 21',
                    nargs='+',
                    required=True)
parser.add_argument('-i22', '--input_profiles_22',
                    dest='input_profiles_22',
                    metavar='input_profiles_22',
                    help='Input profiles 22',
                    nargs='+',
                    required=True)
parser.add_argument('-o', '--output',
                    dest='output',
                    metavar='output',
                    help='output directory to which the outputs are stored',
                    required=False)

args = parser.parse_args()

result_dir = os.path.abspath(args.output)
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

basedir = os.getcwd()

profiles_11 = [np.loadtxt(os.path.abspath(f)) for f in args.input_profiles_11]
profiles_12 = [np.loadtxt(os.path.abspath(f)) for f in args.input_profiles_12]
profiles_21 = [np.loadtxt(os.path.abspath(f)) for f in args.input_profiles_21]
profiles_22 = [np.loadtxt(os.path.abspath(f)) for f in args.input_profiles_22]

p11 = np.array(profiles_11)
p12 = np.array(profiles_12)
p21 = np.array(profiles_21)
p22 = np.array(profiles_22)

p1 = p11 + p12
p2 = p21 + p22
p1_mean = p1.mean(0)
p2_mean = p2.mean(0)
p1_e = p1.std(0)
p2_e = p2.std(0)

numberofbins = p1_mean.shape[0]

fig = plt.figure(figsize=(8, 6))
x_axis = np.linspace(0, 1, numberofbins)

xlabel = 'Normalised hippocampal abscissa ( Anterior $\longleftrightarrow$ Posterior )'
ylabel_1 = 'FTD'
ylabel_2 = 'Alzheimer\'s Disease'
title = 'Hippocampal volume (% of TIV)'

plt.errorbar(x_axis, p1_mean[:, 1], yerr=p1_e[:, 1] / 2, color='0.5', errorevery=4, label=ylabel_1)
plt.errorbar(x_axis, p2_mean[:, 1], yerr=p2_e[:, 1] / 2, color='0.0', errorevery=4, label=ylabel_2)

plt.plot(x_axis, np.mean(p1_mean[:, 1]) * np.ones(numberofbins), color='0.5', linestyle='--')
plt.plot(x_axis, np.mean(p2_mean[:, 1]) * np.ones(numberofbins), color='0.0', linestyle='--')

ymax = 7000

plt.text(0.65, 0.7332 * ymax,
         'total volume: %.1f (%.1f)' % (np.mean(p1_mean[:, 1]), np.mean(p1_e[:, 1])),
#         'total volume: 3.52 (0.36)',
         verticalalignment='center',
         color='0.5')
plt.text(0.65, 0.6666 * ymax,
         'total volume: %.1f (%.1f)' % (np.mean(p2_mean[:, 1]), np.mean(p2_e[:, 1])),
#         'total volume: 2.94 (0.45)',
         verticalalignment='center',
         color='0.0')

#for p11, p12, p21, p22 in zip(profiles_11, profiles_12, profiles_21, profiles_22):
#    plt.plot(x_axis, p11[:, 1], 'g--')
#    plt.plot(x_axis, p12[:, 1], 'g--')
#    plt.plot(x_axis, p21[:, 1], 'r--')
#    plt.plot(x_axis, p22[:, 1], 'r--')

plt.title(title)
plt.xlabel(xlabel)
plt.legend(loc='best', fontsize='small')
plt.ylim([0, ymax])
plt.grid(which='major', axis='both')
plt.tight_layout()
fig.savefig('graph.png', format='PNG')
plt.close()

