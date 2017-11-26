# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
import os
from copy import deepcopy
import re
from nipype.utils.filemanip import split_filename
import nibabel as nb
from nipype.interfaces.base import (CommandLine, CommandLineInputSpec,
                                    OutputMultiPath, isdefined,
                                    TraitedSpec, File, Directory,
                                    traits, InputMultiPath, BaseInterface,
                                    BaseInterfaceInputSpec)


class Dtitk2FslInputSpec(BaseInterfaceInputSpec):
    in_file = File(argstr="%s", exists=True, mandatory=True,
                   desc="Input tensor image filename in the FSL format")


class Dtitk2FslOutputSpec(TraitedSpec):
    out_file = File(argstr="%s", name_source=['in_file'],
                    name_template='%s_nii',
                    desc="Output tensor image filename (in the DTITK format)")


class Dtitk2Fsl(BaseInterface):
    """
    Converts DTI-TK compatible tensor image file to FSL tensor image.

    Example
    --------
    """

    input_spec = Dtitk2FslInputSpec
    output_spec = Dtitk2FslOutputSpec

    def _run_interface(self, runtime):
        self.out_file = self.convert_dtitk2fsl(self.inputs.in_file)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.out_file
        return outputs

    @staticmethod
    def convert_dtitk2fsl(in_file, out_file='dtitk2fsl.nii.gz'):

        in_image = nb.load(in_file)
        header = in_image.get_header()
#        data = 0.001 * in_image.get_data()
        data = in_image.get_data()
        shape = list(data.shape[0:3])
        zooms = list(header.get_zooms()[0:3])
        shape.append(data.shape[4])
        zooms.append(header.get_zooms()[4])
        header.set_data_shape(shape)
        header.set_zooms(zooms)
        out_image = nb.Nifti1Image(data.reshape(shape),
                                   affine=in_image.get_affine(),
                                   header=header)
        nb.save(out_image, out_file)
        return os.path.abspath(out_file)


class Fsl2DtitkInputSpec(BaseInterfaceInputSpec):
    in_file = File(argstr="%s", exists=True, mandatory=True,
                   desc="Input tensor image filename in the FSL format")


class Fsl2DtitkOutputSpec(TraitedSpec):
    out_file = File(argstr="%s", name_source=['in_file'],
                    name_template='%s_nii',
                    desc="Output tensor image filename (in the DTITK format)")


class Fsl2Dtitk(BaseInterface):
    """
    Converts FSL tensor image into DTI-TK compatible tensor image.

    Example
    --------
    """

    input_spec = Fsl2DtitkInputSpec
    output_spec = Fsl2DtitkOutputSpec

    def _run_interface(self, runtime):
        self.out_file = self.convert_fsl2dtitk(self.inputs.in_file)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self.out_file
        return outputs

    @staticmethod
    def convert_fsl2dtitk(in_file, out_file='fsl2dtitk.nii.gz'):

        in_image = nb.load(in_file)
        header = in_image.get_header()
#        data = 1000 * in_image.get_data()
        data = in_image.get_data()
        shape = list(data.shape[0:3])
        zooms = list(header.get_zooms()[0:3])
        shape.append(1)
        shape.append(data.shape[3])
        zooms.append(1)
        zooms.append(header.get_zooms()[3])
        header.set_data_shape(shape)
        header.set_zooms(zooms)
        out_image = nb.Nifti1Image(data.reshape(shape),
                                   affine=in_image.get_affine(),
                                   header=header)
        nb.save(out_image, out_file)
        return os.path.abspath(out_file)


class GetAxisOrientationInputSpec(BaseInterfaceInputSpec):
    in_file = File(argstr='%s', exists=True, mandatory=True,
                   desc="Input target image filename")


class GetAxisOrientationOutputSpec(TraitedSpec):
    out_dict = traits.Dict(desc='Dictionnary containing the axis orientation.')


class GetAxisOrientation(BaseInterface):
    input_spec = GetAxisOrientationInputSpec
    output_spec = GetAxisOrientationOutputSpec

    def _run_interface(self, runtime):
        input_file = self.inputs.in_file
        self.out_dict = self.get_axis_orientation(input_file)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_dict'] = self.out_dict
        return outputs

    @staticmethod
    def get_axis_orientation(input_file):
        outfile = os.path.abspath('fslhd_out.txt')
        f = open(outfile, 'w+')
        os.system('fslhd -x ' + input_file + ' > ' + outfile)
        fsldict = {}
        for line in f:
            listedline = line.strip().split('=')  # split around the = sign
            if len(listedline) > 1:  # we have the = sign in there
                fsldict[listedline[0].replace(" ", "")] = listedline[1].replace(" ", "")
        f.close()
        out_dict = dict()
        if fsldict['sform_code'] > 0:
            out_dict['i'] = fsldict['sform_i_orientation']
            out_dict['j'] = fsldict['sform_j_orientation']
            out_dict['k'] = fsldict['sform_k_orientation']
        else:
            out_dict['i'] = fsldict['qform_i_orientation']
            out_dict['j'] = fsldict['qform_j_orientation']
            out_dict['k'] = fsldict['qform_k_orientation']
        return out_dict


class Dcm2niiPhilipsInputSpec(CommandLineInputSpec):
    source_names = InputMultiPath(File(exists=True), argstr="%s", position=-1,
                                  copyfile=False, mandatory=True, xor=['source_dir'])
    source_dir = Directory(exists=True, argstr="%s", position=-1, mandatory=True,
                           xor=['source_names'])
    anonymize = traits.Bool(True, argstr='-a', usedefault=True)
    config_file = File(exists=True, argstr="-b %s", genfile=True)
    collapse_folders = traits.Bool(True, argstr='-c', usedefault=True)
    date_in_filename = traits.Bool(True, argstr='-d', usedefault=True)
    events_in_filename = traits.Bool(True, argstr='-e', usedefault=True)
    source_in_filename = traits.Bool(False, argstr='-f', usedefault=True)
    gzip_output = traits.Bool(False, argstr='-g', usedefault=True)
    id_in_filename = traits.Bool(False, argstr='-i', usedefault=True)
    nii_output = traits.Bool(True, argstr='-n', usedefault=True)
    output_dir = Directory(exists=True, argstr='-o %s', genfile=True)
    protocol_in_filename = traits.Bool(True, argstr='-p', usedefault=True)
    reorient = traits.Bool(argstr='-r')
    spm_analyze = traits.Bool(argstr='-s', xor=['nii_output'])
    convert_all_pars = traits.Bool(True, argstr='-v', usedefault=True)
    reorient_and_crop = traits.Bool(False, argstr='-x', usedefault=True)
    philips_precise = traits.Bool(False, argstr='', usedefault=True)


class Dcm2niiPhilipsOutputSpec(TraitedSpec):
    converted_files = OutputMultiPath(File(exists=True))
    reoriented_files = OutputMultiPath(File(exists=True))
    reoriented_and_cropped_files = OutputMultiPath(File(exists=True))
    philips_dwi = OutputMultiPath(File(exists=True))
    bvecs = OutputMultiPath(File(exists=True))
    bvals = OutputMultiPath(File(exists=True))
    outputdir = Directory(exists=True)

    t13d = OutputMultiPath(File(exists=True))
    t12d = OutputMultiPath(File(exists=True))
    pdt2 = OutputMultiPath(File(exists=True))
    flair = OutputMultiPath(File(exists=True))
    psir = OutputMultiPath(File(exists=True))
    mtr = OutputMultiPath(File(exists=True))
    b0map = OutputMultiPath(File(exists=True))
    dirsense = OutputMultiPath(File(exists=True))
    

class Dcm2niiPhilips(CommandLine):
    """Uses MRICRON's dcm2nii to convert dicom files

    Examples
    ========

    >>> from nipype.interfaces.dcm2nii import Dcm2nii # doctest: +SKIP
    >>> converter = Dcm2nii() # doctest: +SKIP
    >>> converter.inputs.source_names = ['functional_1.dcm', 'functional_2.dcm'] # doctest: +SKIP
    >>> converter.inputs.gzip_output = True # doctest: +SKIP
    >>> converter.inputs.output_dir = '.' # doctest: +SKIP
    >>> converter.cmdline # doctest: +SKIP
    'dcm2nii -a y -c y -b config.ini -v y -d y -e y -g y -i n -n y -o . -p y -x n -f n functional_1.dcm'
    >>> converter.run() # doctest: +SKIP

    """

    input_spec = Dcm2niiPhilipsInputSpec
    output_spec = Dcm2niiPhilipsOutputSpec

    _cmd = 'dcm2nii'

    def _format_arg(self, opt, spec, val):
        if opt in ['anonymize', 'collapse_folders', 'date_in_filename', 'events_in_filename',
                   'source_in_filename', 'gzip_output', 'id_in_filename', 'nii_output',
                   'protocol_in_filename', 'reorient', 'spm_analyze', 'convert_all_pars',
                   'reorient_and_crop']:
            spec = deepcopy(spec)
            if val:
                spec.argstr += ' y'
            else:
                spec.argstr += ' n'
                val = True
        if opt == 'source_names':
            return spec.argstr % val[0]
        return super(Dcm2niiPhilips, self)._format_arg(opt, spec, val)

    def _run_interface(self, runtime):

        new_runtime = super(Dcm2niiPhilips, self)._run_interface(runtime)

        [self.output_files,
         self.reoriented_files,
         self.reoriented_and_cropped_files,
         self.philips_dwi,
         self.bvecs, 
         self.bvals,
         self.outputdir,
         self.t13d,
         self.t12d,
         self.pdt2,
         self.flair,
         self.psir,
         self.mtr,
         self.b0map,
         self.dirsense] = self._parse_stdout(new_runtime.stdout)

        return new_runtime

    def _parse_stdout(self, stdout):
        files = []
        reoriented_files = []
        reoriented_and_cropped_files = []
        philips_dwi = []
        bvecs = []
        bvals = []
        skip = False
        last_added_file = None
        t13d = []
        t12d = []
        pdt2 = []
        flair = []
        psir = []
        mtr = []
        b0map = []
        dirsense = []

        if isdefined(self.inputs.output_dir):
            output_dir = self.inputs.output_dir
        else:
            output_dir = self._gen_filename('output_dir')

        for line in stdout.split("\n"):
            if not skip:
                m_file = None
                if line.startswith("Saving "):
                    m_file = line[len("Saving "):]
                elif line.startswith("GZip..."):
                    # for gzipped outpus files are not absolute
                    m_file = os.path.abspath(os.path.join(output_dir,
                                                          line[len("GZip..."):]))
                elif line.startswith("Number of diffusion directions "):
                    if last_added_file:
                        base, filename, ext = split_filename(last_added_file)
                        bvecs.append(os.path.join(base, filename + ".bvec"))
                        bvals.append(os.path.join(base, filename + ".bval"))
                elif re.search('.*-->(.*)', line):
                    val = re.search('.*-->(.*)', line)
                    val = val.groups()[0]
                    if isdefined(self.inputs.output_dir):
                        output_dir = self.inputs.output_dir
                    else:
                        output_dir = self._gen_filename('output_dir')
                    val = os.path.join(output_dir, val)
                    m_file = val

                if m_file:
                    files.append(m_file)
                    last_added_file = m_file
                    continue

                if line.startswith("Reorienting as "):
                    reoriented_files.append(line[len("Reorienting as "):])
                    skip = True
                    continue
                elif line.startswith("Cropping NIfTI/Analyze image "):
                    # base, filename = os.path.split(line[len("Cropping NIfTI/Analyze image "):])
                    # filename = "c" + filename
                    # We don't need at all
                    # reoriented_and_cropped_files.append(os.path.join(base, filename))
                    skip = True
                    continue
                elif line.startswith("Removed DWI from DTI scan "):
                    # remove xFILENAME from converted list and add to philips_dwi
                    file_ind = files.index(last_added_file)
                    del files[file_ind]
                    philips_dwi.append(last_added_file)

                    # when converting from Philips scanner, bvec and bval files, the nifti files is
                    # cropped to remove eADC image (last image)
                    # bval and bvec files are therefore associated with the xFILENAME not the FILENAME
                    
                    # get indices for original bvec and bval files from list (must be xFILENAME version)
                    base, filename, ext = split_filename(last_added_file)
                    orgfilename = filename[1:]  # remove 'x' from filename
                    bvec_ind = bvecs.index(os.path.join(base, orgfilename + ".bvec"))
                    bval_ind = bvals.index(os.path.join(base, orgfilename + ".bval"))
                    del bvecs[bvec_ind]
                    del bvals[bval_ind]

                    # replace bvec/bval with xFILE version
                    bvecs.append(os.path.join(base, filename+'.bvec'))
                    bvals.append(os.path.join(base, filename+'.bval'))
                    
                    skip = True
                    continue
                elif re.search('.*-->(.*)T13D(.*)', line):
                    t13d.append(last_added_file)
                    skip = True
                    continue
                elif re.search('.*-->(.*)T1FASTCLEAR(.*)', line):
                    t12d.append(last_added_file)
                    skip = True
                    continue
                elif re.search('.*-->(.*)PDT2(.*)', line):
                    pdt2.append(last_added_file)
                    skip = True
                    continue
                elif re.search('.*-->(.*)PSIR(.*)', line):
                    psir.append(last_added_file)
                    skip = True
                    continue
                elif re.search('.*-->(.*)FLAIR(.*)', line):
                    flair.append(last_added_file)
                    skip = True
                    continue
                elif re.search('.*-->(.*)MTR(.*)', line):
                    mtr.append(last_added_file)
                    skip = True
                    continue
                elif re.search('.*-->(.*)BOMap(.*)', line):
                    b0map.append(last_added_file)
                    skip = True
                    continue
                elif re.search('.*-->(.*)DIRSENSE(.*)', line):
                    dirsense.append(last_added_file)
                    skip = True
                    continue

            skip = False
        return files, reoriented_files, reoriented_and_cropped_files, philips_dwi, bvecs, bvals, \
               output_dir, t13d, t12d, pdt2, flair, psir, mtr, b0map, dirsense

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['converted_files'] = self.output_files
        outputs['reoriented_files'] = self.reoriented_files
        outputs['reoriented_and_cropped_files'] = self.reoriented_and_cropped_files
        outputs['philips_dwi'] = self.philips_dwi
        outputs['bvecs'] = self.bvecs
        outputs['bvals'] = self.bvals
        outputs['outputdir'] = self.outputdir
        outputs['t13d'] = self.t13d
        outputs['t12d'] = self.t12d
        outputs['pdt2'] = self.pdt2
        outputs['flair'] = self.flair
        outputs['psir'] = self.psir
        outputs['mtr'] = self.mtr
        outputs['b0map'] = self.b0map
        outputs['dirsense'] = self.dirsense
        return outputs

    def _gen_filename(self, name):
        if name == 'output_dir':
            return os.getcwd()
        elif name == 'config_file':
            config_file = "config.ini"
            f = open(config_file, "w")
            # disable interactive mode
            f.write("[BOOL]\nManualNIfTIConv=0\n")
            f.close()
            return config_file
        return None


class Pct2DcmInputSpec(CommandLineInputSpec):
    in_umap_file = File(desc="umap nifti file (e.g. pCT umap)",
                        exists=True,
                        mandatory=True,
                        argstr="%s",
                        position=0)
    in_ute_umap_dir = Directory(desc="UTE umap - dicom folder",
                                exists=True,
                                mandatory=True,
                                argstr="%s",
                                position=1)
    in_umap_name = traits.String(desc="Name of the umap to convert",
                                 mandatory=True,
                                 argstr="%s",
                                 position=2)


class Pct2DcmOutputSpec(TraitedSpec):
    output_file = File(desc="Output zip file with the pseudo CT umap DICOMs",
                       exists=True)


class Pct2Dcm(CommandLine):
    input_spec = Pct2DcmInputSpec
    output_spec = Pct2DcmOutputSpec
    _cmd = 'Pct2Dcm.sh'

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['output_file'] = os.path.abspath(self.inputs.in_umap_name + '_DICOM.zip')
        return outputs


class Image2VtkMeshInputSpec(CommandLineInputSpec):
    in_file = File(argstr="-i %s", exists=True, mandatory=True,
                   desc="Input binary image filename")

    matrix_file = File(exists=True, argstr="-m %s",
                       desc="optional matrix to use for extra transformation ")
    out_file = File(argstr="-o %s",
                    desc="output mesh file in VTK format",
                    name_source=['in_file'],
                    name_template='%s_mesh.vtk',
                    keep_extension=False)
    in_reductionRate = traits.Float(0.2,
                                    argstr="-r %s",
                                    mandatory=False,
                                    desc='rate of mesh reduction: if r=0.20: 20pc reduction' +
                                         '(if there was 100 triangles, now there will be 80)',
                                    use_default=True)


class Image2VtkMeshOutputSpec(TraitedSpec):
    out_file = File(desc="Output mesh file in VTK format", exists=True)


class Image2VtkMesh(CommandLine):
    """
    Examples
    --------
    """
    _cmd = "image2mesh"
    input_spec = Image2VtkMeshInputSpec
    output_spec = Image2VtkMeshOutputSpec

