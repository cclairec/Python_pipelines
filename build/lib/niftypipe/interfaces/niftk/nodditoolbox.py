import os
from string import Template
from nipype.interfaces.base import (TraitedSpec, File, traits, BaseInterface, BaseInterfaceInputSpec)
from nipype.interfaces.matlab import MatlabCommand


# A custom function for getting noddi version
def getNoddiPath():
    try:
        cmd = os.environ['NODDIDIR']
        return cmd
    except KeyError:
        return '.'


# A custom function for getting noddi toolbox version
def getNoddiToolBoxPath():
    try:
        cmd = os.environ['NODDITOOLBOXDIR']
        return cmd
    except KeyError:
        return '.'


# Generate optional arguments to pass to NODDI fitting script if necessary
def getExtraArgs(noise_scaling_factor, tissuetype, matlabpoolsize):
    """Generate optional arguments to pass to NODDI fitting script if necessary
    """

    extra_args = ""
    if noise_scaling_factor != -1:
        extra_args = "%s,\'noisescaling\',%1.3f" % (extra_args, noise_scaling_factor)
    if str.lower(tissuetype) != 'default':
        extra_args = "%s,\'tissuetype\',%s" % (extra_args, tissuetype)
    if matlabpoolsize != 1:
        extra_args = "%s,\'matlabpoolsize\',%d" % (extra_args, matlabpoolsize)

    return extra_args


class NoddiInputSpec(BaseInterfaceInputSpec):
    in_dwis = File(exists=True,
                   desc='The input 4D DWIs image file',
                   mandatory=True)
    in_mask = File(exists=True,
                   desc='The input mask image file',
                   mandatory=True)
    in_bvals = File(exists=True,
                    desc='The input bval file',
                    mandatory=True)
    in_bvecs = File(exists=True,
                    desc='The input bvec file',
                    mandatory=True)
    in_b0threshold = traits.Float(desc='The input B0 threshold (default = 5)',
                                  mandatory=False,
                                  default_value=5.0,
                                  usedefault=True)
    in_fname = traits.Str('noddi',
                          desc='The output fname to use',
                          usedefault=True)
    noise_scaling_factor = traits.Int(-1,
                                      desc='Scaling factor for sigma using in NODDI fitting',
                                      usedefault=True)
    tissue_type = traits.Str('default',
                             desc='tissuetype NODDI fitting',
                             usedefault=True)
    matlabpoolsize = traits.Int(desc='Pool size for matlab processing (default = 1 / not parallel)',
                                mandatory=False,
                                default_value=1,
                                usedefault=True)


class NoddiOutputSpec(TraitedSpec):
    out_neural_density = File(genfile=True, desc='The output neural density image file')
    out_orientation_dispersion_index = File(genfile=True, desc='The output orientation dispersion index image file')
    out_csf_volume_fraction = File(genfile=True, desc='The output csf volume fraction image file')
    out_objective_function = File(genfile=True, desc='The output objective function image file')
    out_kappa_concentration = File(genfile=True, desc='The output Kappa concentration image file')
    out_error = File(genfile=True, desc='The output estimation error image file')
    out_fibre_orientations_x = File(genfile=True, desc='The output fibre orientation (x) image file')
    out_fibre_orientations_y = File(genfile=True, desc='The output fibre orientation (y) image file')
    out_fibre_orientations_z = File(genfile=True, desc='The output fibre orientation (z) image file')

    matlab_output = traits.Str()


class Noddi(BaseInterface):
    """
    NODDI estimation interface for the MATLAB toolbox of the NODDI diffusion model fitting.

    NODDI (neurite orientation dispersion and density imaging) is  a practical diffusion MRI technique for estimating
    the microstructural complexity of dendrites and axons in vivo on clinical MRI scanners.

    Documentation and installation of the MATLAB toolbox can be found here:
    http://mig.cs.ucl.ac.uk/index.php?n=Tutorial.NODDImatlab

    More information about the model can be found in the NeuroImage paper:
    http://dx.doi.org/10.1016/j.neuroimage.2012.03.072


    Examples
    --------

    # >>> n = Noddi()
    # >>> n.inputs.in_dwis = subject1_dwis.nii.gz
    # >>> n.inputs.in_mask = subject1_mask.nii.gz
    # >>> n.inputs.in_bvals = subject1_bvals.bval
    # >>> n.inputs.in_bvecs = subject1_bvecs.bvec
    # >>> n.inputs.in_fname = 'subject1'
    # >>> n.inputs.noise_scaling_factor = 1
    # >>> out = n.run()
    # >>> print out.outputs
    """

    input_spec = NoddiInputSpec
    output_spec = NoddiOutputSpec

    def _run_interface(self, runtime):
        """This is where you implement your script"""
        d = dict(in_dwis=self.inputs.in_dwis,
                 in_mask=self.inputs.in_mask,
                 in_bvals=self.inputs.in_bvals,
                 in_bvecs=self.inputs.in_bvecs,
                 in_b0threshold=self.inputs.in_b0threshold,
                 in_fname=self.inputs.in_fname,
                 in_noddi_toolbox=getNoddiToolBoxPath(),
                 in_noddi_path=getNoddiPath(),
                 extra_noddi_args=getExtraArgs(self.inputs.noise_scaling_factor, self.inputs.tissue_type, self.inputs.matlabpoolsize))

        # this is your MATLAB code template
        script = Template("""
        addpath(genpath('$in_noddi_path'));
        addpath(genpath('$in_noddi_toolbox'));
        in_dwis = '$in_dwis';
        in_mask = '$in_mask';
        in_bvals = '$in_bvals';
        in_bvecs = '$in_bvecs';
        in_b0threshold = $in_b0threshold;
        in_fname = '$in_fname';
        [~,~,~,~,~,~,~] = noddi_fitting(in_dwis, in_mask, in_bvals, in_bvecs, in_b0threshold, in_fname $extra_noddi_args);
        exit;
        """).substitute(d)

        mlab = MatlabCommand(script=script, mfile=True)
        result = mlab.run()
        return result.runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        basename = self.inputs.in_fname
        outputs['out_neural_density'] = os.path.join(os.getcwd(), basename + '_ficvf.nii')
        outputs['out_orientation_dispersion_index'] = os.path.join(os.getcwd(), basename + '_odi.nii')
        outputs['out_csf_volume_fraction'] = os.path.join(os.getcwd(), basename + '_fiso.nii')
        outputs['out_objective_function'] = os.path.join(os.getcwd(), basename + '_fmin.nii')
        outputs['out_kappa_concentration'] = os.path.join(os.getcwd(), basename + '_kappa.nii')
        outputs['out_error'] = os.path.join(os.getcwd(), basename + '_error_code.nii')
        outputs['out_fibre_orientations_x'] = os.path.join(os.getcwd(), basename + '_fibredirs_xvec.nii')
        outputs['out_fibre_orientations_y'] = os.path.join(os.getcwd(), basename + '_fibredirs_yvec.nii')
        outputs['out_fibre_orientations_z'] = os.path.join(os.getcwd(), basename + '_fibredirs_zvec.nii')
        return outputs

