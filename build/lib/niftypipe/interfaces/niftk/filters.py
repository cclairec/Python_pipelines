# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from nipype.interfaces.base import (TraitedSpec, File, traits)
from nipype.interfaces.fsl.base import FSLCommand as NIFTKCommand
from .base import NIFTKCommandInputSpec


class N4BiasCorrectionInputSpec(NIFTKCommandInputSpec):
    in_file = File(argstr="-i %s", exists=True, mandatory=True,
                   desc="Input target image filename")

    mask_file = File(exists=True, argstr="--inMask %s",
                     desc="Mask over the input image")

    out_file = File(argstr="-o %s",
                    desc="output bias corrected image",
                    name_source=['in_file'],
                    name_template='%s_corrected')

    in_levels = traits.BaseInt(desc='Number of Multi-Scale Levels - optional - default = 3',
                               argstr='--nlevels %d')

    in_downsampling = traits.BaseInt(desc='Level of Downsampling - optional - default = 1' +
                                          '(no downsampling), downsampling to level 2 is recommended',
                                     argstr='--sub %d')

    in_maxiter = traits.BaseInt(desc='Maximum number of Iterations - optional - default = 50',
                                argstr='--niters %d')

    out_biasfield_file = File(argstr="--outBiasField %s",
                              desc="output bias field file",
                              name_source=['in_file'],
                              name_template='%s_biasfield')

    in_convergence = traits.Float(desc='Convergence Threshold - optional - default = 0.001', argstr='--convergence %f')

    in_fwhm = traits.Float(
        desc='The full width at half maximum of the Gaussian used to model the bias field. (default: 0.15)',
        argstr='--FWHM %f')


class N4BiasCorrectionOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output bias corrected image")
    out_biasfield_file = File(desc="output bias field")


class N4BiasCorrection(NIFTKCommand):
    _cmd = "niftkN4BiasFieldCorrection"

    input_spec = N4BiasCorrectionInputSpec
    output_spec = N4BiasCorrectionOutputSpec


class CropImageInputSpec(NIFTKCommandInputSpec):
    in_file = File(argstr="-i %s", exists=True, mandatory=True,
                   desc="Input target image filename")

    mask_file = File(exists=True, argstr="-m %s",
                     desc="Mask over the input image")

    out_file = File(argstr="-o %s",
                    desc="output cropped image",
                    name_source=['in_file'],
                    name_template='%s_cropped')


class CropImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Output cropped image file")


class CropImage(NIFTKCommand):
    """
    Examples
    --------
    import niftk as niftk
    cropper = CropImage()
    cropper.inputs.in_file = "T1.nii.gz"
    cropper.inputs.mask_file = "mask.nii.gz"
    cropper.inputs.out_file = "T1_cropped.nii.gz"
    cropper.run()
    """
    _cmd = "niftkCropImage"

    input_spec = CropImageInputSpec
    output_spec = CropImageOutputSpec
