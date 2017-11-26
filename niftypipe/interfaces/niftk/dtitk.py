# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
import shutil
from nipype.interfaces.base import (TraitedSpec, File, traits, isdefined,
                                    CommandLine, CommandLineInputSpec, BaseInterface,
                                    BaseInterfaceInputSpec)
from nipype.utils.filemanip import split_filename


def get_dtitk_path():
    try:
        os.environ['DTITK_ROOT']
    except KeyError:
        return ''
    return os.environ['DTITK_ROOT']


"""
    DTITKBinaryThresholdImageFilter interface
"""


class DTITKBinaryThresholdImageFilterInputSpec(CommandLineInputSpec):
    in_file = File(argstr="%s",
                   exists=True,
                   mandatory=True,
                   position=1,
                   desc="Text file containing a list of the input files")
    out_file = File(genfile=True,
                    argstr='%s',
                    position=2,
                    desc='image to write')
    lower_val = traits.Float(argstr='%f',
                             mandatory=True,
                             default=0.01,
                             position=3,
                             desc='Lower threshold value')
    upper_val = traits.Float(argstr='%f',
                             mandatory=True,
                             default=10,
                             position=4,
                             desc='Upper threshold value')
    inside = traits.Float(argstr='%f',
                          mandatory=True,
                          default=0,
                          position=5,
                          desc='Inside value')
    outside = traits.Float(argstr='%f',
                           mandatory=True,
                           default=1,
                           position=6,
                           desc='Outside value')


class DTITKBinaryThresholdImageFilterOutputSpec(TraitedSpec):
    out_file = File(desc="Output file")


class DTITKBinaryThresholdImageFilter(CommandLine):
    _cmd = get_dtitk_path() + '/utilities/' + 'BinaryThresholdImageFilter'
    input_spec = DTITKBinaryThresholdImageFilterInputSpec
    output_spec = DTITKBinaryThresholdImageFilterOutputSpec
    _suffix = '_BinaryThresholdImageFilter'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            _, name, ext = split_filename(self.inputs.in_file)
            outputs['out_file'] = os.path.abspath(name + self._suffix + '.nii.gz')
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None

"""
    DfToInverse interface
"""


class DfToInverseInputSpec(CommandLineInputSpec):
    in_file = File(argstr="-in %s",
                   exists=True,
                   mandatory=True,
                   desc="Input deformation field to invert")
    out_file = File(genfile=True,
                    argstr='-out %s',
                    desc='Inverted deformation field')


class DfToInverseOutputSpec(TraitedSpec):
    out_file = File(desc="Inverted deformation field")


class DfToInverse(CommandLine):
    _cmd = get_dtitk_path() + '/bin/' + 'dfToInverse'
    input_spec = DfToInverseInputSpec
    output_spec = DfToInverseOutputSpec
    _suffix = '_dfToInverse'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            _, name, ext = split_filename(self.inputs.in_file)
            outputs['out_file'] = os.path.abspath(name + self._suffix + '.nii.gz')
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None


"""
    TVMean interface
"""


class TVMeanInputSpec(CommandLineInputSpec):
    in_file_list = File(argstr="-in %s",
                        exists=True,
                        mandatory=True,
                        desc="Text file containing a list of the input files")
    out_file = File(genfile=True,
                    argstr='-out %s',
                    desc='image to write')


class TVMeanOutputSpec(TraitedSpec):
    out_file = File(desc="Output file")


class TVMean(CommandLine):
    _cmd = get_dtitk_path() + '/bin/' + 'TVMean'
    input_spec = TVMeanInputSpec
    output_spec = TVMeanOutputSpec
    _suffix = '_tvMean'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            _, name, ext = split_filename(self.inputs.in_file_list)
            outputs['out_file'] = os.path.abspath(name + self._suffix + '.nii.gz')
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None


"""
    VVMean interface
"""


class VVMeanInputSpec(CommandLineInputSpec):
    in_file_list = File(argstr="-in %s",
                        exists=True,
                        mandatory=True,
                        desc="Text file containing a list of the input files")
    out_file = File(genfile=True,
                    argstr='-out %s',
                    desc='image to write')


class VVMeanOutputSpec(TraitedSpec):
    out_file = File(desc="Output file")


class VVMean(CommandLine):
    _cmd = get_dtitk_path() + '/bin/' + 'VVMean'
    input_spec = VVMeanInputSpec
    output_spec = VVMeanOutputSpec
    _suffix = '_vvMean'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            _, name, ext = split_filename(self.inputs.in_file_list)
            outputs['out_file'] = os.path.abspath(name + self._suffix + '.nii.gz')
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None

"""
    TVtool interface
"""


class TVtoolInputSpec(CommandLineInputSpec):
    in_file = File(argstr="-in %s",
                   exists=True,
                   mandatory=True,
                   desc="Input file")
    operation = traits.Enum('fa', 'tr', 'ad', 'rd', 'norm', 'spd',
                            argstr='-%s',
                            desc='Type of operation to do')
    scale_value = traits.Float(argstr='-scale %f',
                               desc='Scaling value')
    mask_file = File(argstr='-mask %s',
                     desc='File to use to mask the input image')
    out_file = File(genfile=True,
                    argstr='-out %s',
                    desc='image to write')


class TVtoolOutputSpec(TraitedSpec):
    out_file = File(desc="Output file")


class TVtool(CommandLine):
    _cmd = get_dtitk_path() + '/bin/' + 'TVtool'
    input_spec = TVtoolInputSpec
    output_spec = TVtoolOutputSpec
    _suffix = '_tvtool'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            _, name, ext = split_filename(self.inputs.in_file)
            outputs['out_file'] = os.path.abspath(name + self._suffix + ext)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None

"""
    TVAdjustVoxelspace interface
"""


class TVAdjustVoxelspaceInputSpec(CommandLineInputSpec):
    in_file = File(argstr="-in %s",
                   exists=True,
                   mandatory=True,
                   desc="Input file")
    orig_val_x = traits.Float(argstr='-origin %f',
                              mandatory=True,
                              position=1,
                              desc='Origin along the x axis')
    orig_val_y = traits.Float(argstr='%f',
                              mandatory=True,
                              position=2,
                              desc='Origin along the y axis')
    orig_val_z = traits.Float(argstr='%f',
                              mandatory=True,
                              position=3,
                              desc='Origin along the z axis')
    out_file = File(genfile=True,
                    argstr='-out %s',
                    desc='image to write')


class TVAdjustVoxelspaceOutputSpec(TraitedSpec):
    out_file = File(desc="Output file")


class TVAdjustVoxelspace(CommandLine):
    _cmd = get_dtitk_path() + '/bin/' + 'TVAdjustVoxelspace'
    input_spec = TVAdjustVoxelspaceInputSpec
    output_spec = TVAdjustVoxelspaceOutputSpec
    _suffix = '_TVAdjustVoxelspace'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            _, name, ext = split_filename(self.inputs.in_file)
            outputs['out_file'] = os.path.abspath(name + self._suffix + ext)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None

"""
    TVResample interface
"""


class TVResampleInputSpec(CommandLineInputSpec):

    in_file = File(argstr="-in %s",
                   exists=True,
                   mandatory=True,
                   desc="Input file")
    size_val_x = traits.Int(argstr='-size %i',
                            mandatory=True,
                            position=1,
                            desc='Size along the x axis')
    size_val_y = traits.Int(argstr='%i',
                            mandatory=True,
                            position=2,
                            desc='Size along the y axis')
    size_val_z = traits.Int(argstr='%i',
                            mandatory=True,
                            position=3,
                            desc='Size along the z axis')
    vsize_val_x = traits.Float(argstr='-vsize %f',
                               mandatory=True,
                               position=4,
                               desc='Voxel size along the x axis')
    vsize_val_y = traits.Float(argstr='%f',
                               mandatory=True,
                               position=5,
                               desc='Voxel size along the y axis')
    vsize_val_z = traits.Float(argstr='%f',
                               mandatory=True,
                               position=6,
                               desc='Voxel size along the z axis')
    out_file = File(genfile=True,
                    argstr='-out %s',
                    desc='image to write')


class TVResampleOutputSpec(TraitedSpec):
    out_file = File(desc="Output file")


class TVResample(CommandLine):
    _cmd = get_dtitk_path() + '/bin/' + 'TVResample'
    input_spec = TVResampleInputSpec
    output_spec = TVResampleOutputSpec
    _suffix = '_TVResample'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            _, name, ext = split_filename(self.inputs.in_file)
            outputs['out_file'] = os.path.abspath(name + self._suffix + ext)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None


"""
    AffineSymTensor3DVolume interface
"""


class AffineSymTensor3DVolumeInputSpec(CommandLineInputSpec):
    ref_file = File(argstr="-target %s", exists=True, desc="Reference file")
    flo_file = File(argstr="-in %s", exists=True, desc="floating file")
    in_trans = File(argstr="-trans %s", exist=True, desc="Input transformation")
    sm_option_val = traits.Enum('LEI', 'EI',
                                desc='Interpolation type',
                                default='LEI',
                                argstr="-interp %s")
    out_file = File(genfile=True,
                    argstr='-out %s',
                    desc='image to write')


class AffineSymTensor3DVolumeOutputSpec(TraitedSpec):
    out_file = File(desc='image to write')


class AffineSymTensor3DVolume(CommandLine):
    _cmd = get_dtitk_path() + '/bin/' + 'affineSymTensor3DVolume'
    input_spec = AffineSymTensor3DVolumeInputSpec
    output_spec = AffineSymTensor3DVolumeOutputSpec
    _suffix = '_AffineSymTensor3DVolume'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            _, name, ext = split_filename(self.inputs.flo_file)
            outputs['out_file'] = os.path.abspath(name + self._suffix + ext)
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None


"""
    AffineSymTensor3DVolume interface
"""


class Affine3DShapeAverageInputSpec(CommandLineInputSpec):
    file_list = File(argstr="%s", exists=True, desc="Text file containing all affine", position=1)
    ref_file = File(argstr="%s", exists=True, desc="Reference file", position=2)
    out_file = File(genfile=True, argstr='%s', desc='Output affine transformation', position=3)
    inverse_flag = traits.Bool(argstr="%i", exists=True, desc="average (0) or inverse (1)", default=0, position=4)


class Affine3DShapeAverageOutputSpec(TraitedSpec):
    out_file = File(desc='Output affine transformation')


class Affine3DShapeAverage(CommandLine):
    _cmd = get_dtitk_path() + '/bin/' + 'affine3DShapeAverage'
    input_spec = Affine3DShapeAverageInputSpec
    output_spec = Affine3DShapeAverageOutputSpec
    _suffix = '_Affine3DShapeAverage'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            _, name, ext = split_filename(self.inputs.file_list)
            outputs['out_file'] = os.path.abspath(name + self._suffix + '.aff')
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None


"""
    DeformationSymTensor3DVolume interface
"""


class DeformationSymTensor3DVolumeInputSpec(CommandLineInputSpec):
    flo_file = File(argstr="-in %s", exists=True, desc="floating image", mandatory=True)
    in_trans = File(argstr="-trans %s", exists=True, desc="transformation file", mandatory=True)
    out_file = File(genfile=True, argstr=' -out %s', desc='Warped image')


class DeformationSymTensor3DVolumeOutputSpec(TraitedSpec):
    out_file = File(desc='Warped image')


class DeformationSymTensor3DVolume(CommandLine):
    _cmd = get_dtitk_path() + '/bin/' + 'deformationSymTensor3DVolume'
    input_spec = DeformationSymTensor3DVolumeInputSpec
    output_spec = DeformationSymTensor3DVolumeOutputSpec
    _suffix = '_DeformationSym3DVolume'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            _, name, ext = split_filename(self.inputs.flo_file)
            outputs['out_file'] = os.path.abspath(name + self._suffix + '.nii.gz')
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None


"""
    Affine3Dtool interface
"""


class Affine3DtoolInputSpec(CommandLineInputSpec):
    in_file = File(argstr="-in %s", exists=True, desc="Input transformation")
    comp_file = File(argstr="-compose %s", exists=True, desc="transformation to compose")
    out_file = File(genfile=True, argstr='-out %s', desc='Output affine transformation')


class Affine3DtoolOutputSpec(TraitedSpec):
    out_file = File(desc='Output affine transformation')


class Affine3Dtool(CommandLine):
    _cmd = get_dtitk_path() + '/bin/' + 'affine3Dtool'
    input_spec = Affine3DtoolInputSpec
    output_spec = Affine3DtoolOutputSpec
    _suffix = '_Affine3Dtool'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            _, name, ext = split_filename(self.inputs.in_file)
            outputs['out_file'] = os.path.abspath(name + self._suffix + '.aff')
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None

"""
    DfRightComposeAffine interface
"""


class DfRightComposeAffineInputSpec(CommandLineInputSpec):
    aff_file = File(argstr="-aff %s", exists=True, mandatory=True, desc="Input affine transformation")
    def_file = File(argstr="-df %s", exists=True, mandatory=True, desc="Input nonrigid transformation")
    out_file = File(genfile=True, argstr='-out %s', desc='Output deformation field')


class DfRightComposeAffineOutputSpec(TraitedSpec):
    out_file = File(desc='Output affine transformation')


class DfRightComposeAffine(CommandLine):
    _cmd = get_dtitk_path() + '/bin/' + 'dfRightComposeAffine'
    input_spec = DfRightComposeAffineInputSpec
    output_spec = DfRightComposeAffineOutputSpec
    _suffix = '_DfRightComposeAffine'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_file):
            _, name, ext = split_filename(self.inputs.def_file)
            outputs['out_file'] = os.path.abspath(name + self._suffix + '.nii.gz')
        else:
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._list_outputs()[name]
        return None


"""
    RTVCGM interface
"""


class RTVCGMInputSpec(CommandLineInputSpec):
    ref_file = File(argstr="-template %s", exists=True, desc="Reference file",
                    position=1)
    flo_file = File(argstr="-subject %s", exists=True, desc="floating file",
                    position=2)
    sm_option_val = traits.Enum('EDS', 'GDS', 'DDS', desc='Similarity measure',
                                argstr="-SMOption %s", position=3)
    sep_x_val = traits.Float(argstr="-sep %f", mandatory=False, default=2,
                             position=4)
    sep_y_val = traits.Float(argstr="%f", mandatory=False, default=2,
                             position=5)
    sep_z_val = traits.Float(argstr="%f", mandatory=False, default=2,
                             position=6)
    ftol_val = traits.Float(argstr="-ftol %f", mandatory=False, default=0.005,
                            position=7)
    in_trans = File(mandatory=False,
                    desc="Input transformation")
    out_trans = File(genfile=True,
                     argstr='-outTrans %s',
                     desc="Output transformation")


class RTVCGMOutputSpec(TraitedSpec):
    out_trans = File(desc="Output transformation")


class RTVCGM(CommandLine):
    _cmd = get_dtitk_path() + '/bin/' + 'rtvCGM'
    input_spec = RTVCGMInputSpec
    output_spec = RTVCGMOutputSpec
    _suffix = '_rtvCGM'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_trans):
            _, name, ext = split_filename(self.inputs.flo_file)
            outputs['out_trans'] = os.path.abspath(name + self._suffix + '.aff')
        else:
            outputs['out_trans'] = os.path.abspath(self.inputs.out_trans)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_trans':
            return self._list_outputs()[name]
        return None


"""
    ATVCGM interface
"""


class ATVCGMInputSpec(CommandLineInputSpec):
    ref_file = File(argstr="-template %s", exists=True, desc="Reference file",
                    position=1)
    flo_file = File(argstr="-subject %s", exists=True, desc="floating file",
                    position=2)
    sm_option_val = traits.Enum('EDS', 'GDS', 'DDS', desc='Similarity measure',
                                argstr="-SMOption %s", position=3)
    sep_x_val = traits.Float(argstr="-sep %f", mandatory=False, default=2,
                             position=4)
    sep_y_val = traits.Float(argstr="%f", mandatory=False, default=2,
                             position=5)
    sep_z_val = traits.Float(argstr="%f", mandatory=False, default=2,
                             position=6)
    ftol_val = traits.Float(argstr="-ftol %f", mandatory=False, default=0.005,
                            position=7)
    in_trans = File(mandatory=False,
                    desc="Input transformation")
    out_trans = File(genfile=True,
                     argstr='-outTrans %s',
                     desc="Output transformation")


class ATVCGMOutputSpec(TraitedSpec):
    out_trans = File(desc="Output transformation")


class ATVCGM(CommandLine):
    _cmd = get_dtitk_path() + '/bin/' + 'atvCGM'
    input_spec = ATVCGMInputSpec
    output_spec = ATVCGMOutputSpec
    _suffix = '_atvCGM'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_trans):
            _, name, ext = split_filename(self.inputs.flo_file)
            outputs['out_trans'] = os.path.abspath(name + self._suffix + '.aff')
        else:
            outputs['out_trans'] = os.path.abspath(self.inputs.out_trans)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_trans':
            return self._list_outputs()[name]
        return None


"""
    DTIDiffeomorphicReg interface
"""


class DtiDiffeomorphicRegInputSpec(CommandLineInputSpec):
    ref_file = File(argstr="%s", exists=True, desc="Reference file", mandatory=True, position=1)
    flo_file = File(argstr="%s", exists=True, desc="floating file", mandatory=True, position=2)
    mask_file = File(argstr="%s", exists=True, desc="floating file", mandatory=True, position=3)
    initial_val = traits.Int(argstr="%i", default=1, desc="Initial value", mandatory=True, position=4)
    iteration_val = traits.Int(argstr="%i", default=1, desc="iteration number", mandatory=True, position=5)
    ftol_val = traits.Float(argstr="%f", default=0.005, position=6)


class DtiDiffeomorphicRegOutputSpec(TraitedSpec):
    out_trans = File(desc="Output transformation")


class DtiDiffeomorphicReg(CommandLine):
    _cmd = get_dtitk_path() + '/scripts/' + 'dti_diffeomorphic_reg'
    input_spec = DtiDiffeomorphicRegInputSpec
    output_spec = DtiDiffeomorphicRegOutputSpec
    _suffix = '_diffeo.df'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        bn, name, ext = split_filename(self.inputs.flo_file)
        source = bn + os.sep + name + self._suffix + '.nii.gz'
        destination = os.path.abspath(name + self._suffix + '.nii.gz')
        shutil.move(source, destination)
        outputs['out_trans'] = destination
        return outputs

    def _gen_filename(self, name):
        if name == 'out_trans':
            return self._list_outputs()[name]
        return None


class FSLEddyAcqInputSpec(BaseInterfaceInputSpec):
    ped = traits.Enum("x","y","z","-x","-y","-z",mandatory=True,
                        desc='Phase encode direction [-](x|y|z)')
    rot = traits.Float(mandatory=True,
                       desc='Readout time (ms)')
    dticount = traits.Int(mandatory=True,
                          desc='Number of acquisitions')

class FSLEddyAcqOutputSpec(TraitedSpec):
    out_acqp = File(exists=True, desc="Output acquisition parameters file")
    out_index = File(exists =True, desc="Indices for all volumes")


class FSLEddyAcq(BaseInterface):
    input_spec = FSLEddyAcqInputSpec
    output_spec = FSLEddyAcqOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_acqp'] = os.path.abspath("acqp.txt")
        outputs['out_index'] = os.path.abspath("index.txt")
        return outputs

    def _run_interface(self, runtime):
        ped = self.inputs.ped
        rot = self.inputs.rot
        dticount = self.inputs.dticount
        dircos = [(-1 if "-" in ped else 1) * int(ax in ped) for ax in ["x", "y", "z" ]]
        acqvals = dircos + [rot/1e3]
        with open("acqp.txt", "w") as acqpf:
            acqpf.write(" ".join(map(str,acqvals))+"\n")
            acqpf.close()
        indices = dticount*[1]
        with open("index.txt", "w") as indf:
            indf.write(" ".join(map(str,indices)))
            indf.close()
        return runtime
