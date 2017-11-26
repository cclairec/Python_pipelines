# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from nipype.interfaces.base import (TraitedSpec, File, BaseInterface, BaseInterfaceInputSpec)
from nipype.utils.filemanip import split_filename
import numpy as np
import os


def get_label_dictionary():
    label_names = dict()
    label_names['0'] = 'not assigned'
    label_names['1'] = 'lh_bankssts'
    label_names['2'] = 'lh_caudalanteriorcingulate'
    label_names['3'] = 'lh_caudalmiddlefrontal'
    label_names['4'] = 'lh_cuneus'
    label_names['5'] = 'lh_entorhinal'
    label_names['6'] = 'lh_frontalpole'
    label_names['7'] = 'lh_fusiform'
    label_names['8'] = 'lh_inferiorparietal'
    label_names['9'] = 'lh_inferiortemporal'
    label_names['10'] = 'lh_insula'
    label_names['11'] = 'lh_isthmuscingulate'
    label_names['12'] = 'lh_lateraloccipital'
    label_names['13'] = 'lh_lateralorbitofrontal'
    label_names['14'] = 'lh_lingual'
    label_names['15'] = 'lh_medialorbitofrontal'
    label_names['16'] = 'lh_middletemporal'
    label_names['17'] = 'lh_paracentral'
    label_names['18'] = 'lh_parahippocampal'
    label_names['19'] = 'lh_parsopercularis'
    label_names['20'] = 'lh_parsorbitalis'
    label_names['21'] = 'lh_parstriangularis'
    label_names['22'] = 'lh_pericalcarine'
    label_names['23'] = 'lh_postcentral'
    label_names['24'] = 'lh_posteriorcingulate'
    label_names['25'] = 'lh_precentral'
    label_names['26'] = 'lh_precuneus'
    label_names['27'] = 'lh_rostralanteriorcingulate'
    label_names['28'] = 'lh_rostralmiddlefrontal'
    label_names['29'] = 'lh_superiorfrontal'
    label_names['30'] = 'lh_superiorparietal'
    label_names['31'] = 'lh_superiortemporal'
    label_names['32'] = 'lh_supramarginal'
    label_names['33'] = 'lh_temporalpole'
    label_names['34'] = 'lh_transversetemporal'
    label_names['35'] = 'rh_bankssts'
    label_names['36'] = 'rh_caudalanteriorcingulate'
    label_names['37'] = 'rh_caudalmiddlefrontal'
    label_names['38'] = 'rh_cuneus'
    label_names['39'] = 'rh_entorhinal'
    label_names['40'] = 'rh_frontalpole'
    label_names['41'] = 'rh_fusiform'
    label_names['42'] = 'rh_inferiorparietal'
    label_names['43'] = 'rh_inferiortemporal'
    label_names['44'] = 'rh_insula'
    label_names['45'] = 'rh_isthmuscingulate'
    label_names['46'] = 'rh_lateraloccipital'
    label_names['47'] = 'rh_lateralorbitofrontal'
    label_names['48'] = 'rh_lingual'
    label_names['49'] = 'rh_medialorbitofrontal'
    label_names['50'] = 'rh_middletemporal'
    label_names['51'] = 'rh_paracentral'
    label_names['52'] = 'rh_parahippocampal'
    label_names['53'] = 'rh_parsopercularis'
    label_names['54'] = 'rh_parsorbitalis'
    label_names['55'] = 'rh_parstriangularis'
    label_names['56'] = 'rh_pericalcarine'
    label_names['57'] = 'rh_postcentral'
    label_names['58'] = 'rh_posteriorcingulate'
    label_names['59'] = 'rh_precentral'
    label_names['60'] = 'rh_precuneus'
    label_names['61'] = 'rh_rostralanteriorcingulate'
    label_names['62'] = 'rh_rostralmiddlefrontal'
    label_names['63'] = 'rh_superiorfrontal'
    label_names['64'] = 'rh_superiorparietal'
    label_names['65'] = 'rh_superiortemporal'
    label_names['66'] = 'rh_supramarginal'
    label_names['67'] = 'rh_temporalpole'
    label_names['68'] = 'rh_transversetemporal'
    return label_names


class FreesurferUpdateCsvFileWithLabelsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,
                   mandatory=True,
                   desc="Input csv file")


class FreesurferUpdateCsvFileWithLabelsOutputSpec(TraitedSpec):
    out_file = File(desc="Updated file")


class FreesurferUpdateCsvFileWithLabels(BaseInterface):
    """

    Examples
    --------

    """

    input_spec = FreesurferUpdateCsvFileWithLabelsInputSpec
    output_spec = FreesurferUpdateCsvFileWithLabelsOutputSpec

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        out_file = self._gen_output_filename(in_file)
        in_array = np.genfromtxt(in_file, delimiter=',')
        label_dict = get_label_dictionary()
        f = open(out_file, 'w+')
        s = in_array.shape
        for i in range(s[0]):
            label_value = np.int(in_array[i, 0])
            if str(label_value) in label_dict:
                f.write("%u," % label_value)
                f.write("%s" % repr(label_dict[str(label_value)]))
                for j in range(1, s[1]):
                    f.write(",%5.2e" % in_array[i, j])
                f.write("\n")
        f.close()
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_output_filename(self.inputs.in_file)
        return outputs

    @staticmethod
    def _gen_output_filename(in_file):
        _, bn, _ = split_filename(in_file)
        outfile = os.path.abspath(bn + '_fs.txt')
        return outfile
