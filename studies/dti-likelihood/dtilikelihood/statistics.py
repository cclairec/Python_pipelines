# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
    Simple interface for Rician and Gaussian noise addition
"""

import os
import nibabel as nib
import scipy.stats as ss
import numpy as np

from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec
from nipype.interfaces.base import (TraitedSpec, File, traits, isdefined)
from nipype.utils.filemanip import split_filename


def apply_noise(original_image, mask, noise_type, sigma_val):
    nib_image = nib.load(original_image)
    data = nib_image.get_data()
    mask_data = nib.load(mask).get_data()
    output_data = np.zeros(data.shape)
    np.random.seed()
    r = np.random.standard_normal(size=data.shape)
    if noise_type == 'gaussian':
        output_data = data + sigma_val * r
    elif noise_type == 'rician':
        output_data = np.sqrt((data + sigma_val * r)**2 + (sigma_val*r)**2)
    output_data[mask_data <= 0] = 0
    output_data[output_data < 0] = 0
    nib_output = nib.Nifti1Image(output_data, nib_image.get_affine())
    return nib_output


class NoiseAdderInputSpec(BaseInterfaceInputSpec):
    
    in_file = File(argstr='%s',
                   exists=True,
                   mandatory=True,
                   desc='Input file to add noise to')
    mask_file = File(argstr='%s',
                     exists=False,
                     desc='Mask of input image space')
    _noise_type = ['gaussian', 'rician']
    noise_type = traits.Enum(*_noise_type,
                             argstr='%s',
                             desc='Type of added noise (gaussian or rician)',
                             mandatory=True)
    sigma_val = traits.Float(argstr='%f',
                             desc='Value of the added noise sigma',
                             mandatory=True)
    out_file = File(argstr='%s',
                    desc='Output image with added noise',
                    name_source=['in_file'],
                    name_template='%s_noisy')
    
    
class NoiseAdderOutputSpec(TraitedSpec):
    out_file = File(desc='Output image with added noise',
                    exists=True)


class NoiseAdder(BaseInterface):

    """

    Examples
    --------

    """

    _suffix = '_noisy'
    input_spec = NoiseAdderInputSpec  
    output_spec = NoiseAdderOutputSpec

    def _gen_output_filename(self, in_file):
        _, base, _ = split_filename(in_file)
        outfile = base + self._suffix + '.nii.gz'
        return outfile
        
    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_file):
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        else:
            outputs['out_file'] = os.path.abspath(self._gen_output_filename(self.inputs.in_file))
        return outputs

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        noise_type = self.inputs.noise_type
        sigma_val = self.inputs.sigma_val
        out_file = self._list_outputs()['out_file']
        mask_file = self.inputs.mask_file

        nib_output = apply_noise(in_file, mask_file, noise_type, sigma_val)
        nib.save(nib_output, out_file)
        return runtime
