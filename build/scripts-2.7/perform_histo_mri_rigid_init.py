#!/home/claicury/Code/Install/niftypipe/bin/python

import nipype.interfaces.utility as niu  # utility
import nipype.interfaces.io as nio  # Input Output
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.interfaces.niftyreg as niftyreg  # pypeline engine
import sys
import os
import textwrap
import argparse
import niftk
from nipype import config


class HistoMRVariables():
    def __init__(self):
        self.input_histo = None
        self.input_mri = None
        self.output_folder = None
        self.graph = False
        self.parser = None
        self.mri_histo_create_parser()

    def mri_histo_create_parser(self):
        pipeline_description = textwrap.dedent('''
            Jonas to add text here.
        ''')
        self.parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                              description=pipeline_description)
        """ Input images """
        self.parser.add_argument('--histo',
                                 dest='input_histo',
                                 type=str,
                                 metavar='image',
                                 help='MRI file',
                                 required=True)
        self.parser.add_argument('--histo_mask',
                                 dest='input_histo_mask',
                                 type=str,
                                 metavar='image',
                                 help='MRI mask file',
                                 default=None,
                                 required=False)
        self.parser.add_argument('--mri',
                                 dest='input_mri',
                                 type=str,
                                 metavar='image',
                                 help='Histo file',
                                 required=True)
        self.parser.add_argument('--mri_mask',
                                 dest='input_mri_mask',
                                 type=str,
                                 metavar='image',
                                 help='Histo mask file',
                                 default=None,
                                 required=False)
        """ Output argument """
        self.parser.add_argument('--output_dir',
                                 dest='output_folder',
                                 type=str,
                                 metavar='directory',
                                 help='Output directory containing the pipeline result',
                                 required=True)
        self.parser.add_argument('--res_iso',
                                 dest='resolution',
                                 type=str,
                                 metavar='resolution_iso',
                                 help='Some blabla for Jonas to complete',
                                 required=True)
        self.parser.add_argument('--sim',
                                 dest='sim_measure',
                                 type=str,
                                 metavar='similarity_measure',
                                 help='Some blabla for Jonas to complete',
                                 required=True)
        self.parser.add_argument('--nosym',
                                 dest='nosym',
                                 help='Some blabla for Jonas to complete',
                                 default=False,
                                 action='store_true')
        """ Others argument """
        self.parser.add_argument('-g',
                                 '--graph',
                                 dest='graph',
                                 help='Print a graph describing the node connections and exit',
                                 action='store_true',
                                 default=False)


def initialise_headers(in_file):
    import nibabel as nib
    import os
    from nipype.utils.filemanip import split_filename

    nii_image = nib.load(in_file)
    size_x = nii_image.get_header()['dim'][1] * nii_image.get_header()['pixdim'][1]
    size_y = nii_image.get_header()['dim'][2] * nii_image.get_header()['pixdim'][2]

    nii_image.get_header()['qform_code'] = 1
    nii_image.get_header()['sform_code'] = 0

    nii_image.get_header()['quatern_b'] = 0
    nii_image.get_header()['quatern_c'] = 0
    nii_image.get_header()['quatern_d'] = 0

    nii_image.get_header()['qoffset_x'] = -size_x / 2
    nii_image.get_header()['qoffset_y'] = -size_y / 2
    nii_image.get_header()['qoffset_z'] = 0

    _, bn, ext = split_filename(in_file)
    out_file = os.getcwd() + os.sep + bn + '_init' + ext
    new_nii_image = nib.Nifti1Image(nii_image.get_data(), None, nii_image.get_header())
    nib.save(new_nii_image, out_file)

    return out_file


def generate_matrices(histo_file, mri_file):
    import nibabel as nib
    import numpy as np
    import os
    histo_image = nib.load(histo_file)
    mri_image = nib.load(mri_file)
    max_value = np.max((histo_image.get_header()['dim'][2]-mri_image.get_header()['dim'][2],
                        mri_image.get_header()['dim'][2]-histo_image.get_header()['dim'][2]))
    increment = histo_image.get_header()['pixdim'][2]
    if histo_image.get_header()['dim'][2] < mri_image.get_header()['dim'][2]:
        increment = -increment

    matrix_files = []

    for i in range(0, max_value):
        offset = i * increment
        matrix = np.eye(4, 4)
        matrix[2][3] = offset
        filename = os.getcwd() + os.sep + 'matrix_' + str(i).zfill(3) + '.txt'
        np.savetxt(filename, matrix)
        matrix_files.append(filename)
    return matrix_files


def generate_plotsim(measures, in_mat, out_mat):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    val_number = len(measures)
    indices = np.zeros((val_number, 1))
    sim_values = np.zeros((val_number, 1))
    off_values = np.zeros((val_number, 1))
    for i in range(0, val_number):
        input_matrix = np.loadtxt(in_mat[i])
        output_matrix = np.loadtxt(out_mat[i])
        indices[i] = input_matrix[2][3]
        sim_values[i] = np.loadtxt(measures[i])
        off_values[i] = output_matrix[2][3]
    plt.subplot(1, 2, 1)
    plt.plot(indices, sim_values, 'o')
    plt.subplot(1, 2, 2)
    plt.plot(indices, off_values, 'x')
    out_file = os.getcwd() + os.sep + 'plot.pdf'
    plt.savefig(out_file)
    return out_file


"""
Main
"""


def main():
    # Initialise the pipeline variables and the argument parsing
    mri_histo_value_var = HistoMRVariables()
    # Parse the input arguments
    input_variables = mri_histo_value_var.parser.parse_args()

    # Create the output folder if it does not exists
    if not os.path.exists(os.path.abspath(input_variables.output_folder)):
        os.mkdir(os.path.abspath(input_variables.output_folder))

    # Create the workflow
    name = 'mri_histo_rigid'
    workflow = pe.Workflow(name=name)
    workflow.base_dir = os.path.abspath(input_variables.output_folder)
    workflow.base_output_dir = name

    # Create the input node interface
    input_node = pe.Node(
        interface=niu.IdentityInterface(
            fields=['input_histo',
                    'input_histo_mask',
                    'input_mri',
                    'input_mri_mask',
                    'resolution',
                    'sim_measure']),
        name='input_node')
    input_node.inputs.input_histo = os.path.abspath(input_variables.input_histo)
    input_node.inputs.input_histo_mask = os.path.abspath(input_variables.input_histo_mask)
    input_node.inputs.input_mri = os.path.abspath(input_variables.input_mri)
    input_node.inputs.input_mri_mask = os.path.abspath(input_variables.input_mri_mask)
    input_node.inputs.resolution = float(input_variables.resolution)
    input_node.inputs.sim_measure = input_variables.sim_measure

    # Alter the input image headers
    alter_mri_header = pe.Node(interface=niu.Function(function=initialise_headers,
                                                        input_names=['in_file'],
                                                        output_names=['out_file']),
                                 name='alter_mri_header')
    workflow.connect(input_node, 'input_mri', alter_mri_header, 'in_file')
    alter_histo_header = pe.Node(interface=niu.Function(function=initialise_headers,
                                                  input_names=['in_file'],
                                                  output_names=['out_file']),
                           name='alter_histo_header')
    workflow.connect(input_node, 'input_histo', alter_histo_header, 'in_file')

    alter_histo_mask_header = pe.Node(interface=niu.Function(function=initialise_headers,
                                                           input_names=['in_file'],
                                                           output_names=['out_file']),
                                    name='alter_histo_mask_header')

    alter_mri_mask_header = pe.Node(interface=niu.Function(function=initialise_headers,
                                                             input_names=['in_file'],
                                                             output_names=['out_file']),
                                      name='alter_mri_mask_header')

    # Resample to isotropic images
    iso_mri = pe.Node(interface=niftyreg.RegTools(),
                        name='iso_mri')
    iso_mri.inputs.chg_res_val = (input_node.inputs.resolution,
                                    input_node.inputs.resolution,
                                    input_node.inputs.resolution)
    workflow.connect(alter_mri_header, 'out_file', iso_mri, 'in_file')
    iso_histo = pe.Node(interface=niftyreg.RegTools(),
                        name='iso_histo')
    iso_histo.inputs.chg_res_val = (input_node.inputs.resolution,
                                    input_node.inputs.resolution,
                                    input_node.inputs.resolution)
    workflow.connect(alter_histo_header, 'out_file', iso_histo, 'in_file')

    iso_histo_mask = pe.Node(interface=niftyreg.RegTools(),
                           name='iso_histo_mask')
    iso_histo_mask.inputs.chg_res_val = (input_node.inputs.resolution,
                                       input_node.inputs.resolution,
                                       input_node.inputs.resolution)
    iso_mri_mask = pe.Node(interface=niftyreg.RegTools(),
                             name='iso_mri_mask')
    iso_mri_mask.inputs.chg_res_val = (input_node.inputs.resolution,
                                         input_node.inputs.resolution,
                                         input_node.inputs.resolution)

    # Generate matrices
    create_matrices = pe.Node(interface=niu.Function(function=generate_matrices,
                                                     input_names=['histo_file', 'mri_file'],
                                                     output_names=['matrix_files']),
                              name='create_matrices')
    workflow.connect(iso_histo, 'out_file', create_matrices, 'histo_file')
    workflow.connect(iso_mri, 'out_file', create_matrices, 'mri_file')

    # Run all the registrations
    rigid = pe.MapNode(interface=niftyreg.RegAladin(),
                       iterfield=['in_aff_file'],
                       name='rigid')
    rigid.inputs.rig_only_flag = True
    rigid.inputs.verbosity_off_flag = True
    if input_variables.nosym == True:
        rigid.inputs.nosym_flag = True
    workflow.connect(create_matrices, 'matrix_files', rigid, 'in_aff_file')
    workflow.connect(iso_histo, 'out_file', rigid, 'ref_file')
    workflow.connect(iso_mri, 'out_file', rigid, 'flo_file')
    if input_variables.input_histo_mask is not None:
        workflow.connect(input_node, 'input_histo_mask', alter_histo_mask_header, 'in_file')
        workflow.connect(alter_histo_header, 'out_file', iso_histo_mask, 'in_file')
        workflow.connect(iso_histo_mask, 'out_file', rigid, 'rmask_file')
    if input_variables.input_mri_mask is not None:
        workflow.connect(input_node, 'input_mri_mask', alter_mri_mask_header, 'in_file')
        workflow.connect(alter_mri_header, 'out_file', iso_mri_mask, 'in_file')
        workflow.connect(iso_mri_mask, 'out_file', rigid, 'fmask_file')

    # Run all the similarity measures
    similarity = pe.MapNode(interface=niftyreg.RegMeasure(),
                            iterfield=['flo_file'],
                            name='similarity')
    workflow.connect(input_node, 'sim_measure', similarity, 'measure_type')
    workflow.connect(iso_histo, 'out_file', similarity, 'ref_file')
    workflow.connect(rigid, 'res_file', similarity, 'flo_file')

    # Display similarity measures
    disp_sim = pe.Node(interface=niu.Function(function=generate_plotsim,
                                              input_names=['measures',
                                                           'in_mat',
                                                           'out_mat'],
                                              output_names=['out_file']),
                       name='disp_sim')
    workflow.connect(similarity, 'out_file', disp_sim, 'measures')
    workflow.connect(create_matrices, 'matrix_files', disp_sim, 'in_mat')
    workflow.connect(rigid, 'aff_file', disp_sim, 'out_mat')

    # Create a data sink
    ds = pe.Node(nio.DataSink(parameterization=False),
                 name='data_sink')
    ds.inputs.base_directory = workflow.base_dir
    workflow.connect(disp_sim, 'out_file', ds, '@plot')
    workflow.connect(alter_histo_header, 'out_file', ds, '@histo')
    workflow.connect(alter_mri_header, 'out_file', ds, '@mri')

    # output the graph if required
    if input_variables.graph is True:
        niftk.generate_graph(workflow=workflow)
        return

    # Run the workflow
    qsub_args = '-l h_rt=01:00:00 -l tmem=1.9G -l h_vmem=1.9G -l vf=1.9G -l s_stack=10240 -j y -b y -S /bin/csh -V'
    niftk.run_workflow(workflow=workflow,
                       qsubargs=qsub_args)

if __name__ == "__main__":
    main()
