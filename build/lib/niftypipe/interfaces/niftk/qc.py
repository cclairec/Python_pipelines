# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import sys
import os.path
import numpy as np
import nibabel as nib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from nipype.interfaces.base import (TraitedSpec, File, traits, BaseInterface, BaseInterfaceInputSpec)
from nipype.utils.filemanip import split_filename
import nipype.pipeline.engine as pe


class InterSliceCorrelationPlotInputSpec(BaseInterfaceInputSpec):
    in_file = File(argstr="%s", exists=True, mandatory=True,
                   desc="Input image")
    bval_file = File(argstr="%s", exists=True, mandatory=False,
                     desc="Input bval file")


class InterSliceCorrelationPlotOutputSpec(TraitedSpec):
    out_file = File(exists=False, genfile=True,
                    desc="Interslice correlation plot")


class InterSliceCorrelationPlot(BaseInterface):
    input_spec = InterSliceCorrelationPlotInputSpec
    output_spec = InterSliceCorrelationPlotOutputSpec
    _suffix = "_interslice_ncc"

    def _gen_output_filename(self, in_file):
        _, base, _ = split_filename(in_file)
        outfile = base + self._suffix + '.png'
        return os.path.abspath(outfile)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.out_file
        return outputs

    def _run_interface(self, runtime):
        # Load the original image
        nib_image = nib.load(self.inputs.in_file)
        data = nib_image.get_data()
        dim = data.shape
        vol_number = 1
        if len(dim) > 3:
            vol_number = dim[3]
        if self.inputs.bval_file:
            bvalues = np.loadtxt(self.inputs.bval_file)
            if len(bvalues) != vol_number:
                sys.exit()
        cc_values = np.zeros((vol_number, len(range(1, dim[2] - 1))))
        b0_values = np.zeros((vol_number, len(range(1, dim[2] - 1))))
        for v in range(vol_number):
            if vol_number > 1:
                volume = data[:, :, :, v]
            else:
                volume = data[:, :, :]
            current_b0 = False
            if self.inputs.bval_file:
                if bvalues[v] < 20:
                    current_b0 = True
            temp_array1 = volume[:, :, 0]
            voxel_number = temp_array1.size
            temp_array1 = np.reshape(temp_array1, voxel_number)
            temp_array1 = (temp_array1 - np.mean(temp_array1)) / np.std(temp_array1)
            for z in range(1, dim[2] - 1):
                temp_array2 = np.reshape(volume[:, :, z], voxel_number)
                temp_array2 = (temp_array2 - np.mean(temp_array2)) / np.std(temp_array2)
                if current_b0:
                    cc_values[v, z - 1] = np.nan
                    b0_values[v, z - 1] = (np.correlate(temp_array1, temp_array2) / np.double(voxel_number))
                else:
                    b0_values[v, z - 1] = np.nan
                    cc_values[v, z - 1] = (np.correlate(temp_array1, temp_array2) / np.double(voxel_number))
                temp_array1 = np.copy(temp_array2)
        fig = plt.figure(figsize=(13, 6))
        mask_cc_values = np.ma.masked_array(cc_values, np.isnan(cc_values))
        mask_b0_values = np.ma.masked_array(b0_values, np.isnan(b0_values))
        mean_cc_values = np.mean(mask_cc_values[:, :], axis=0)
        mean_b0_values = np.mean(mask_b0_values[:, :], axis=0)
        std_cc_values = np.std(mask_cc_values[:, :], axis=0)
        std_b0_values = np.std(mask_b0_values[:, :], axis=0)
        x_axis = np.array(range(1, dim[2] - 1))
        mpl.rcParams['text.latex.unicode'] = True
        plt.plot(x_axis, mean_cc_values, 'b-',
                 label='Mean non B0 $(\pm 3.5 \sigma)$')
        plt.plot(x_axis, mean_b0_values, 'r-',
                 label='Mean B0 $(\pm 3.5 \sigma)$')
        sigma_mul = 3.5
        plt.fill_between(x_axis,
                         mean_cc_values - sigma_mul * std_cc_values,
                         mean_cc_values + sigma_mul * std_cc_values,
                         facecolor='b',
                         linestyle='dashed',
                         alpha=0.2)
        plt.fill_between(x_axis,
                         mean_b0_values - sigma_mul * std_b0_values,
                         mean_b0_values + sigma_mul * std_b0_values,
                         facecolor='r',
                         linestyle='dashed',
                         alpha=0.2)
        for v in range(vol_number):
            current_b0 = False
            current_label = False
            if self.inputs.bval_file:
                if bvalues[v] < 20:
                    current_b0 = True
            if current_b0:
                for z in range(1, dim[2] - 1):
                    if b0_values[v, z - 1] < mean_b0_values[z - 1] - sigma_mul * std_b0_values[z - 1]:
                        if current_label:
                            plt.plot(z, b0_values[v, z - 1], '.',
                                     color=(plt.cm.jet(v * 255 / vol_number)[0:3]))
                        else:
                            current_label = True
                            plt.plot(z, b0_values[v, z - 1], '.',
                                     color=(plt.cm.jet(v * 255 / vol_number)[0:3]),
                                     label='Volume ' + str(v))
            else:
                for z in range(1, dim[2] - 1):
                    if cc_values[v, z - 1] < mean_cc_values[z - 1] - sigma_mul * std_cc_values[z - 1]:
                        if current_label:
                            plt.plot(z, cc_values[v, z - 1], '.',
                                     color=(plt.cm.jet(v * 255 / vol_number)[0:3]))
                        else:
                            current_label = True
                            plt.plot(z, cc_values[v, z - 1], '.',
                                     color=(plt.cm.jet(v * 255 / vol_number)[0:3]),
                                     label='Volume ' + str(v))
        plt.ylabel('Normalised cross-corelation')
        plt.xlabel('Slice Number ( Inferior $\longleftrightarrow$ Superior )')
        plt.title('Inter-slice normalised cross-correlation')
        plt.xticks(np.arange(0, dim[2], 2.0))
        plt.ylim([0, 1])
        plt.grid(which='major', axis='both')
        plt.legend(loc='best', numpoints=1, fontsize='small')
        plt.tight_layout()
        self.out_file = self._gen_output_filename(self.inputs.in_file)
        fig.savefig(self.out_file, format='PNG')
        plt.close()
        return runtime




def create_MatrixRotationPlot_workflow(name="matrixrotplot"):
    """
    Produce motion plot from input list of transform matrices.
    Bundles matrixprepplot and transrotationplot interfaces for convenience.
    Inputs ::

        input_node.in_files - list of affine matrices

    Outputs ::

        output_node.out_file - plot of motion
    """
    # Create the input and output nodes
    input_node = pe.Node(MatrixPrepPlot(), name='input_node')
    output_node = pe.Node(TransRotationPlot(), name='output_node')

    # Create the workflow
    pipeline = pe.Workflow(name=name)
    pipeline.base_output_dir = name
    pipeline.connect(input_node, 'out_xfms', output_node, 'in_xfms')
    return pipeline
        


class PrepPlotReadParamsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,
                   mandatory=True,
                   desc="Input parameters file")


class PrepPlotReadParamsOutputSpec(TraitedSpec):
    # It's not the most obvious, but needs to match MatrixPrepPlot
    out_xfms = traits.List(traits.Tuple((traits.Float,) * 6),
                           desc="List of tuples of x,y,z translations, x,y,z rotations")


class PrepPlotReadParams(BaseInterface):
    """
    Derive translations and rotations from matrix input to be used in
    plotting for QC. Use to create inputs to TransRotationPlot.
    """
    input_spec = PrepPlotReadParamsInputSpec
    output_spec = PrepPlotReadParamsOutputSpec
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_xfms'] = self.out_xfms
        return outputs

    def _run_interface(self, runtime):
        matrix = np.loadtxt(self.inputs.in_file)

        #self.out_xfms = zip(translations_x, translations_y, translations_z,
        #                    rotation_x, rotation_y, rotation_z)
        xfm_matrix = matrix[:, 0:6]
        self.out_xfms = [ tuple(row) for row in xfm_matrix.tolist()]
        return runtime




class MatrixPrepPlotInputSpec(BaseInterfaceInputSpec):
    in_files = traits.List(File(exists=True),
                           exists=True,
                           mandatory=True,
                           desc="List of input transformation matrix files")


class MatrixPrepPlotOutputSpec(TraitedSpec):
    out_xfms = traits.List(traits.Tuple((traits.Float,) * 6),
                           desc="List of tuples of x,y,z translations, x,y,z rotations")


class MatrixPrepPlot(BaseInterface):
    """
    Derive translations and rotations from matrix input to be used in
    plotting for QC. Use to create inputs to TransRotationPlot.
    """
    input_spec = MatrixPrepPlotInputSpec
    output_spec = MatrixPrepPlotOutputSpec
    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_xfms'] = self.out_xfms
        return outputs
    @staticmethod
    def get_max_norm_row(in_matrix):
        return np.max([
            np.sum(np.fabs(in_matrix[:][0])),
            np.sum(np.fabs(in_matrix[:][1])),
            np.sum(np.fabs(in_matrix[:][2]))
        ])
    @staticmethod
    def get_max_norm_col(in_matrix):
        return np.max([
            np.sum(np.fabs(in_matrix[0][:])),
            np.sum(np.fabs(in_matrix[1][:])),
            np.sum(np.fabs(in_matrix[2][:]))
        ])

    def polar_decomposition(self, in_matrix):
        gam = np.linalg.det(in_matrix)
        while gam == 0.0:
            gam = 0.00001 * (0.001 + self.get_max_norm_row(in_matrix))
            in_matrix[0][0] += gam
            in_matrix[1][1] += gam
            in_matrix[2][2] += gam
            gam = np.linalg.det(in_matrix)
        dif = 1
        k = 0
        while True:
            matrix_inv = np.linalg.inv(in_matrix)
            if dif > 0.3:
                alp = np.sqrt(self.get_max_norm_row(in_matrix) *
                              self.get_max_norm_col(in_matrix))
                bet = np.sqrt(self.get_max_norm_row(matrix_inv) *
                              self.get_max_norm_col(matrix_inv))
                gam = np.sqrt(bet / alp)
                gmi = 1 / gam
            else:
                gam = 1.0
                gmi = 1.0

            temp_matrix = 0.5 * (gam * in_matrix + gmi * np.transpose(matrix_inv))

            dif = np.fabs(temp_matrix[0][0] - in_matrix[0][0]) + np.fabs(temp_matrix[0][1] - in_matrix[0][1]) + \
                  np.fabs(temp_matrix[0][2] - in_matrix[0][2]) + np.fabs(temp_matrix[1][0] - in_matrix[1][0]) + \
                  np.fabs(temp_matrix[1][1] - in_matrix[1][1]) + np.fabs(temp_matrix[1][2] - in_matrix[1][2]) + \
                  np.fabs(temp_matrix[2][0] - in_matrix[2][0]) + np.fabs(temp_matrix[2][1] - in_matrix[2][1]) + \
                  np.fabs(temp_matrix[2][2] - in_matrix[2][2])
            k += 1
            if k > 100 or dif < 3.e-6:
                break
        return temp_matrix

    def extract_quaternion(self, in_matrix):
        xd = np.sqrt(np.sum(np.square(np.double(in_matrix[:][0]))))
        yd = np.sqrt(np.sum(np.square(np.double(in_matrix[:][1]))))
        zd = np.sqrt(np.sum(np.square(np.double(in_matrix[:][2]))))
        if xd == 0:
            in_matrix[0][0] = 1
            in_matrix[1][0] = 0
            in_matrix[2][0] = 0
        else:
            in_matrix[:][0] /= xd
        if yd == 0:
            in_matrix[0][1] = 0
            in_matrix[1][1] = 1
            in_matrix[2][1] = 0
        else:
            in_matrix[:][1] /= yd
        if zd == 0:
            in_matrix[0][2] = 0
            in_matrix[1][2] = 0
            in_matrix[2][2] = 1
        else:
            in_matrix[:][2] /= zd

        temp_matrix = self.polar_decomposition(in_matrix)
        det = np.linalg.det(temp_matrix)
        if det > 0:
            qfac = 1.0
        else:
            qfac = -1
            temp_matrix[0][2] = -temp_matrix[0][2]
            temp_matrix[1][2] = -temp_matrix[1][2]
            temp_matrix[2][2] = -temp_matrix[2][2]

        a = temp_matrix[0][0] + temp_matrix[1][1] + temp_matrix[2][2] + 1

        if a > 0.5:
            a = 0.5 * np.sqrt(a)
            b = 0.25 * (temp_matrix[2][1] - temp_matrix[1][2]) / a
            c = 0.25 * (temp_matrix[0][2] - temp_matrix[2][0]) / a
            d = 0.25 * (temp_matrix[1][0] - temp_matrix[0][1]) / a
        else:
            xd = 1 + temp_matrix[0][0] - (temp_matrix[1][1] + temp_matrix[2][2])
            yd = 1 + temp_matrix[1][1] - (temp_matrix[0][0] + temp_matrix[2][2])
            zd = 1 + temp_matrix[2][2] - (temp_matrix[0][0] + temp_matrix[1][1])
            if xd > 1.0:
                b = 0.5 * np.sqrt(xd)
                c = 0.25 * (temp_matrix[0][1] + temp_matrix[1][0]) / b
                d = 0.25 * (temp_matrix[0][2] + temp_matrix[2][0]) / b
                a = 0.25 * (temp_matrix[2][1] - temp_matrix[1][2]) / b
            elif yd > 1.0:
                c = 0.5 * np.sqrt(yd)
                b = 0.25 * (temp_matrix[0][1] + temp_matrix[1][0]) / c
                d = 0.25 * (temp_matrix[1][2] + temp_matrix[2][1]) / c
                a = 0.25 * (temp_matrix[0][2] - temp_matrix[2][0]) / c
            else:
                d = 0.50 * np.sqrt(zd)
                b = 0.25 * (temp_matrix[0][2] + temp_matrix[2][0]) / d
                c = 0.25 * (temp_matrix[1][2] + temp_matrix[2][1]) / d
                a = 0.25 * (temp_matrix[1][0] - temp_matrix[0][1]) / d
        if a < 0:
            b = -b
            c = -c
            d = -d
            a = -a
        out_values = {'a': a, 'b': b, 'c': c, 'd': d, 'qfac': qfac}
        return out_values

    def _run_interface(self, runtime):
        all_matrices = self.inputs.in_files
        num_matrices = len(all_matrices)
        rotation_x = np.zeros(num_matrices)
        rotation_y = np.zeros(num_matrices)
        rotation_z = np.zeros(num_matrices)
        translations_x = np.zeros(num_matrices)
        translations_y = np.zeros(num_matrices)
        translations_z = np.zeros(num_matrices)
        for i in range(0, num_matrices):
            matrix = np.loadtxt(all_matrices[i])
            values = self.extract_quaternion(matrix[np.ix_([0, 1, 2], [0, 1, 2])])
            rotation_x[i] = np.arctan(2 * (values['a'] * values['b'] + values['c'] * values['d']) / (
                1 - 2 * (np.square(values['b']) + np.square(values['c']))))
            rotation_y[i] = np.arcsin(2 * (values['a'] * values['c'] - values['b'] * values['d']))
            rotation_z[i] = np.arctan(2 * (values['a'] * values['d'] + values['b'] * values['c']) / (
                1 - 2 * (np.square(values['c']) + np.square(values['d']))))
            translations_x[i] = np.linalg.norm(matrix[0, 3])
            translations_y[i] = np.linalg.norm(matrix[1, 3])
            translations_z[i] = np.linalg.norm(matrix[2, 3])
        self.out_xfms = zip(translations_x, translations_y, translations_z,
                            rotation_x, rotation_y, rotation_z)
        return runtime



class TransRotationPlotInputSpec(BaseInterfaceInputSpec):
    in_xfms = traits.List(traits.Tuple((traits.Float,) * 6),
                          desc="List of tuples of x,y,z translations, x,y,z rotations",
                          mandatory=True)


class TransRotationPlotOutputSpec(TraitedSpec):
    out_file = File(exists=False, genfile=True,
                    desc="Matrix rotation plot")


class TransRotationPlot(BaseInterface):
    input_spec = TransRotationPlotInputSpec
    output_spec = TransRotationPlotOutputSpec
    _suffix = "_rotation"

    def _gen_output_filename(self, in_file):
        _, base, _ = split_filename(in_file)
        outfile = base + self._suffix + '.png'
        return os.path.abspath(outfile)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.out_file
        return outputs

    def _run_interface(self, runtime):
        num_matrices = len(self.inputs.in_xfms)
        trans_x, trans_y, trans_z, rot_x, rot_y, rot_z = zip(*self.inputs.in_xfms)
        rotation_x = rot_x - np.mean(rot_x)
        rotation_y = rot_y - np.mean(rot_y)
        rotation_z = rot_z - np.mean(rot_z)
        translations_x = trans_x - np.mean(trans_x)
        translations_y = trans_y - np.mean(trans_y)
        translations_z = trans_z - np.mean(trans_z)

        fig = plt.figure(figsize=(8, 6))
        x_axis = np.arange(0, num_matrices)
        plt.plot(x_axis, np.rad2deg(rotation_x), 'r-', label='x-axis rot.')
        plt.plot(x_axis, np.rad2deg(rotation_y), 'g-', label='y-axis rot.')
        plt.plot(x_axis, np.rad2deg(rotation_z), 'b-', label='z-axis rot.')
        plt.plot(x_axis, translations_x, 'r--', label='x-trans')
        plt.plot(x_axis, translations_y, 'g--', label='y-trans')
        plt.plot(x_axis, translations_z, 'b--', label='z-trans')

        plt.plot(x_axis,  2 * np.ones(num_matrices), color='0.0', linestyle='--')
        plt.plot(x_axis, -2 * np.ones(num_matrices), color='0.0', linestyle='--')
        plt.ylabel('Demeaned rotation (in deg.) and translation (in mm)')
        plt.xlabel('Volume Number')
        plt.title('Rotation  (degree) and translation (mm) per volume')
        plt.legend(loc='best', fontsize='small')
        plt.ylim([-5, 5])
        plt.grid(which='major', axis='both')
        plt.tight_layout()
        self.out_file = self._gen_output_filename('')
        fig.savefig(self.out_file, format='PNG')
        plt.close()
        return runtime


# @MH
class FmriQcPlotInputSpec(BaseInterfaceInputSpec):
    in_raw_fmri = File(exists=True,
                       mandatory=True,
                       desc="fmri image")
    in_raw_fmri_gm = File(exists=True,
                          mandatory=True,
                          desc="fmri gm image")
    in_raw_fmri_wm = File(exists=True,
                          mandatory=True,
                          desc="fmri wm image")
    in_raw_fmri_csf = File(exists=True,
                           mandatory=True,
                           desc="fmri csf image")
    in_mrp_file = File(exists=True,
                       mandatory=True,
                       desc="fmri mrp text file")
    in_spike_file = File(exists=True,
                         mandatory=True,
                         desc="fmri spike text file")
    in_rms_file = File(exists=True,
                       mandatory=True,
                       desc="fmri rms text file")


class FmriQcPlotOutputSpec(TraitedSpec):
    out_file = File(exists=False, genfile=True,
                    desc="fMRI QC plot")


class FmriQcPlot(BaseInterface):
    input_spec = FmriQcPlotInputSpec
    output_spec = FmriQcPlotOutputSpec
    _suffix = "_qc"

    def _gen_output_filename(self, in_file):
        _, base, _ = split_filename(in_file)
        outfile = base + self._suffix + '.png'
        return os.path.abspath(outfile)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.out_file
        return outputs

    def _run_interface(self, runtime):
        # read fmri file
        fmri_image = nib.load(self.inputs.in_raw_fmri)
        fmri_data = fmri_image.get_data()
        dim = fmri_data.shape
        fmri_data_flat = np.reshape(fmri_data, (dim[0] * dim[1] * dim[2], dim[3]))
        # read fmri segmentation file
        gm_image = nib.load(self.inputs.in_raw_fmri_gm)
        gm_data = gm_image.get_data()
        masked_gm = fmri_data_flat[gm_data.flatten() > 0, :]
        wm_image = nib.load(self.inputs.in_raw_fmri_wm)
        wm_data = wm_image.get_data()
        masked_wm = fmri_data_flat[wm_data.flatten() > 0, :]
        csf_image = nib.load(self.inputs.in_raw_fmri_csf)
        csf_data = csf_image.get_data()
        masked_csf = fmri_data_flat[csf_data.flatten() > 0, :]
        # normalise - Broadcasting is really good for this
        masked_gm_n = (masked_gm - masked_gm.mean(axis=1)[:, np.newaxis]) / masked_gm.std(axis=1)[:, np.newaxis]
        masked_wm_n = (masked_wm - masked_wm.mean(axis=1)[:, np.newaxis]) / masked_wm.std(axis=1)[:, np.newaxis]
        masked_csf_n = (masked_csf - masked_csf.mean(axis=1)[:, np.newaxis]) / masked_csf.std(axis=1)[:, np.newaxis]
        tissue = np.concatenate((masked_gm_n, masked_wm_n, masked_csf_n), axis=0)
        dvars = np.zeros([dim[3], 1])
        for i in range(1, dim[3]):
            dvars[i] = np.sqrt(np.sum((tissue[:, i] - tissue[:, i - 1]) ** 2) / tissue.shape[0])
        dvars = dvars / np.linalg.norm(dvars)
        # read mrp file
        x = np.linspace(1, dim[3], dim[3])
        mrp = np.loadtxt(self.inputs.in_mrp_file)
        spike = np.loadtxt(self.inputs.in_spike_file)
        rms = np.loadtxt(self.inputs.in_rms_file)

        # plot things
        font = {'family': 'serif',
                'weight': 'bold',
                'size': 8
                }
        mpl.rc('font', **font)
        fig = plt.figure()
        plt.subplot(4, 1, 1)
        line_roll, = plt.plot(x, mrp[:, 0], label='$Roll [mm]$')
        line_pitch, = plt.plot(x, mrp[:, 1], label='$Pitch [mm]$')
        line_yaw, = plt.plot(x, mrp[:, 2], label='$Yaw [mm]$')
        plt.legend(handles=[line_roll, line_pitch, line_yaw], prop={'size': 6})
        plt.ylabel('Rotations', fontdict=font)
        plt.title('fMRI QC', fontdict=font)
        plt.ylim([-4, 4])

        plt.subplot(4, 1, 2)
        line_is, = plt.plot(x, mrp[:, 3], label='$\Delta I - S [mm]$')
        line_rl, = plt.plot(x, mrp[:, 4], label='$\Delta R - L [mm]$')
        line_ap, = plt.plot(x, mrp[:, 5], label='$\Delta A - P [mm]$')
        plt.legend(handles=[line_is, line_rl, line_ap], prop={'size': 6})
        plt.ylabel('Translations', fontdict=font)
        plt.ylim([-4, 4])

        plt.subplot(4, 1, 3)
        line_rms, = plt.plot(x, rms, label='$RMS$')
        line_sp, = plt.plot(x, spike, label='$SP$')
        line_dv, = plt.plot(x, dvars, label='$DVARS$')
        plt.legend(handles=[line_rms, line_sp, line_dv], prop={'size': 6})
        plt.ylabel('Normalised Measures', fontdict=font)
        plt.ylim([0, 1])

        plt.subplot(4, 1, 4)
        plt.imshow(tissue, extent=[0, tissue.shape[1], 1, tissue.shape[0]], aspect='auto')
        plt.xlabel('fMRI Volume', fontdict=font)
        plt.ylabel('Tissue', fontdict=font)

        # Tweak spacing between subplots to prevent labels from overlapping
        plt.subplots_adjust(hspace=0.35)
        # save to file
        self.out_file = self._gen_output_filename(self.inputs.in_raw_fmri)
        fig.savefig(self.out_file, format='PNG')
        plt.close()
        return runtime
