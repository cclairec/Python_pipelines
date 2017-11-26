# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
from nipype.interfaces.fsl.base import FSLCommand as RestingStatefMRIPreprocessCommand
from nipype.interfaces.fsl.base import FSLCommandInputSpec as RestingStatefMRIPreprocessCommandInputSpec
from nipype.interfaces.base import (TraitedSpec, File)


class RestingStatefMRIPreprocessInputSpec(RestingStatefMRIPreprocessCommandInputSpec):
    # fmri raw input file
    in_fmri = File(exists=True,
                   mandatory=True,
                   argstr="%s",
                   position=3,
                   desc="raw fmri input file")
    # anatomical t1 image file
    in_t1 = File(exists=True,
                 mandatory=True,
                 argstr="%s",
                 position=4,
                 desc="anatomical input file")
    # segmentation of the T1 scan (including white matter, csf and grey matter segmentation)
    in_tissue_segmentation = File(exists=True,
                                  mandatory=True,
                                  argstr="%s",
                                  position=7,
                                  desc="segmentation of the T1 scan (including csf pos1, " +
                                       "grey pos2 and white matter pos3)")
    # atlas 
    in_parcellation = File(exists=True,
                           mandatory=True,
                           argstr="%s",
                           position=8,
                           desc="segmentation of the T1 scan (including csf pos1 ,grey pos2 and white matter pos3)")


class RestingStatefMRIPreprocessOutputSpec(TraitedSpec):
    # preprocessed fmri scan in subject space
    out_corrected_fmri = File(exists=True, genfile=True, desc="preprocessed fMRI scan in subject space")
    out_fmri_to_t1_transformation = File(exists=True, genfile=True, desc="fMRI to T1 affine transformation")
    out_atlas_fmri = File(exists=True, genfile=True, desc="atlas in fmri space")
    out_raw_fmri_gm = File(exists=True,
                           mandatory=True,
                           desc="fmri gm image")
    out_raw_fmri_wm = File(exists=True,
                           mandatory=True,
                           desc="fmri wm image")
    out_raw_fmri_csf = File(exists=True,
                            mandatory=True,
                            desc="fmri csf image")
    out_mrp_file = File(exists=True,
                        mandatory=True,
                        desc="fmri mrp text file")
    out_spike_file = File(exists=True,
                          mandatory=True,
                          desc="fmri spike text file")
    out_rms_file = File(exists=True,
                        mandatory=True,
                        desc="fmri rms text file")
    out_motioncorrected_file = File(exists=True,
                                    mandatory=True,
                                    desc="fmri motion corrected file")


class RestingStatefMRIPreprocess(RestingStatefMRIPreprocessCommand):
    """

    Examples
    --------

    import restingstatefmri as fmri
    
    rs = fmri.RestingStatefMRIPreprocess()
    
    rs.inputs.in_fmri = "func.nii.gz"
    rs.inputs.in_t1 = "anat.nii.gz"
    rs.inputs.in_tissue_segmentation = "seg.nii.gz"
    rs.inputs.in_parcellation = "atlas.nii.gz"

    rs.run()

    """

    _cmd = "fmri_prep_single.sh"
    input_spec = RestingStatefMRIPreprocessInputSpec
    output_spec = RestingStatefMRIPreprocessOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_corrected_fmri'] = os.path.abspath('fmri_pp.nii.gz')
        outputs['out_atlas_fmri'] = os.path.abspath('atlas_fmri.nii.gz')
        outputs['out_fmri_to_t1_transformation'] = os.path.abspath('fmri_to_t1_transformation.txt')
        outputs['out_raw_fmri_gm'] = os.path.abspath('seg_gm_fmri.nii.gz')
        outputs['out_raw_fmri_wm'] = os.path.abspath('seg_wm_fmri.nii.gz')
        outputs['out_raw_fmri_csf'] = os.path.abspath('seg_csf_fmri.nii.gz')
        outputs['out_mrp_file'] = os.path.abspath('dfile_rall.1D')
        outputs['out_spike_file'] = os.path.abspath('outcount_rall.1D')
        outputs['out_rms_file'] = os.path.abspath('enorm.1D')
        outputs['out_motioncorrected_file'] = os.path.abspath('fmri_motion_corrected.nii.gz')
        return outputs
