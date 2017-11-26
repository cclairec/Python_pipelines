# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from nipype.interfaces.base import (TraitedSpec, File, BaseInterface, BaseInterfaceInputSpec)
from nipype.utils.filemanip import split_filename
import numpy as np
import os


def get_label_dictionary():
    label_names = dict()
    label_names['0'] = 'Unclassified'
    label_names['1'] = 'Middle cerebellar peduncle'
    label_names['2'] = 'Pontine crossing tract (a part of MCP)'
    label_names['3'] = 'Genu of corpus callosum'
    label_names['4'] = 'Body of corpus callosum'
    label_names['5'] = 'Splenium of corpus callosum'
    label_names['6'] = 'Fornix (column and body of fornix)'
    label_names['7'] = 'Corticospinal tract R'
    label_names['8'] = 'Corticospinal tract L'
    label_names['9'] = 'Medial lemniscus R'
    label_names['10'] = 'Medial lemniscus L'
    label_names['11'] = 'Inferior cerebellar peduncle R  '
    label_names['12'] = 'Inferior cerebellar peduncle L'
    label_names['13'] = 'Superior cerebellar peduncle R'
    label_names['14'] = 'Superior cerebellar peduncle L'
    label_names['15'] = 'Cerebral peduncle R'
    label_names['16'] = 'Cerebral peduncle L'
    label_names['17'] = 'Anterior limb of internal capsule R'
    label_names['18'] = 'Anterior limb of internal capsule L'
    label_names['19'] = 'Posterior limb of internal capsule R'
    label_names['20'] = 'Posterior limb of internal capsule L'
    label_names['21'] = 'Retrolenticular part of internal capsule R'
    label_names['22'] = 'Retrolenticular part of internal capsule L'
    label_names['23'] = 'Anterior corona radiata R'
    label_names['24'] = 'Anterior corona radiata L'
    label_names['25'] = 'Superior corona radiata R'
    label_names['26'] = 'Superior corona radiata L'
    label_names['27'] = 'Posterior corona radiata R'
    label_names['28'] = 'Posterior corona radiata L'
    label_names['29'] = 'Posterior thalamic radiation (include optic radiation) R'
    label_names['30'] = 'Posterior thalamic radiation (include optic radiation) L'
    label_names['31'] = 'Sagittal stratum (include inferior longitidinal fasciculus and ' + \
                        'inferior fronto-occipital fasciculus) R'
    label_names['32'] = 'Sagittal stratum (include inferior longitidinal fasciculus and ' + \
                        'inferior fronto-occipital fasciculus) L'
    label_names['33'] = 'External capsule R'
    label_names['34'] = 'External capsule L'
    label_names['35'] = 'Cingulum (cingulate gyrus) R'
    label_names['36'] = 'Cingulum (cingulate gyrus) L'
    label_names['37'] = 'Cingulum (hippocampus) R'
    label_names['38'] = 'Cingulum (hippocampus) L'
    label_names['39'] = 'Fornix (cres) / Stria terminalis (can not be resolved with current resolution) R'
    label_names['40'] = 'Fornix (cres) / Stria terminalis (can not be resolved with current resolution) L'
    label_names['41'] = 'Superior longitudinal fasciculus R'
    label_names['42'] = 'Superior longitudinal fasciculus L'
    label_names['43'] = 'Superior fronto-occipital fasciculus (could be a part of anterior internal capsule) R'
    label_names['44'] = 'Superior fronto-occipital fasciculus (could be a part of anterior internal capsule) L'
    label_names['45'] = 'Uncinate fasciculus R'
    label_names['46'] = 'Uncinate fasciculus L'
    label_names['47'] = 'Tapetum R'
    label_names['48'] = 'Tapetum L'
    return label_names


class JHUUpdateCsvFileWithLabelsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,
                   mandatory=True,
                   desc="Input csv file")


class JHUUpdateCsvFileWithLabelsOutputSpec(TraitedSpec):
    out_file = File(desc="Updated file")


class JHUUpdateCsvFileWithLabels(BaseInterface):
    """

    Examples
    --------

    """

    input_spec = JHUUpdateCsvFileWithLabelsInputSpec
    output_spec = JHUUpdateCsvFileWithLabelsOutputSpec

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
        outfile = os.path.abspath(bn + '_neuromorph.txt')
        return outfile


def jhu_write_labels_function(filename):
    dic = get_label_dictionary()
    keys = dic.keys()
    keys.sort(key=int)
    f = open(filename, 'w')
    for k in keys:
        f.write(k + ', ' + dic[k] + '\n')
    f.close()
    return os.path.abspath(filename)
