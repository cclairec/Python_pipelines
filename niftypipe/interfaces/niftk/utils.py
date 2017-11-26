# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from nipype.interfaces.base import (TraitedSpec, File, traits, BaseInterface, BaseInterfaceInputSpec,
                                    OutputMultiPath, isdefined,CommandLine,CommandLineInputSpec)
from nipype.interfaces import (niftyreg, niftyseg, fsl)
import nipype.interfaces.utility as niu
from nipype.utils.filemanip import split_filename
import nibabel as nib
import numpy as np
import os


class WriteArrayToCsvInputSpec(BaseInterfaceInputSpec):
    in_array = traits.Array(exists=True, mandatory=True,
                            desc="array")
    in_name = traits.String(mandatory=True, desc="Name of the output file")


class WriteArrayToCsvOutputSpec(TraitedSpec):
    out_file = File(desc="Output file")


class WriteArrayToCsv(BaseInterface):
    """

    Examples
    --------

    """

    input_spec = WriteArrayToCsvInputSpec
    output_spec = WriteArrayToCsvOutputSpec

    def _run_interface(self, runtime):
        in_array = self.inputs.in_array
        in_name = self.inputs.in_name
        out_file = self._gen_output_filename(in_name)
        np.savetxt(out_file,
                   in_array,
                   delimiter=',')
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_output_filename(self.inputs.in_name)
        return outputs

    @staticmethod
    def _gen_output_filename(in_name):
        _, bn, _ = split_filename(in_name)
        outfile = os.path.abspath(bn + '.csv')
        return outfile


class MergeLabelsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,
                   mandatory=True,
                   desc="Input roi file")
    roi_list = traits.ListInt(mandatory=True,
                              desc="List containing the label index to use")


class MergeLabelsOutputSpec(TraitedSpec):
    out_file = File(desc="Output file")


class MergeLabels(BaseInterface):
    """

    Examples
    --------

    """

    input_spec = MergeLabelsInputSpec
    output_spec = MergeLabelsOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_output_filename(self.inputs.in_file, self.inputs.roi_list)
        return outputs

    @staticmethod
    def _gen_output_filename(in_file, roi_list):
        _, bn, ext = split_filename(in_file)
        labels = ''
        for l in roi_list:
            labels = labels + '_' + str(l)
        outfile = os.path.abspath(bn + '_merged' + labels + ext)
        return outfile

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        roi_list = self.inputs.roi_list
        # Load the parcelation
        parcelation = nib.load(in_file)
        data = parcelation.get_data()
        # Create a new empty label file
        new_roi_data = np.zeros(data.shape)
        # Extract the relevant roi from the initial parcelation and accumulate the value
        for i in roi_list:
            new_roi_data += np.equal(data, i * np.ones(data.shape))
        # binarise the image
        new_roi_data = (new_roi_data != 0)
        # Create a new image based on the initial parcelation
        out_img = nib.Nifti1Image(np.uint8(new_roi_data), parcelation.get_affine())
        out_img.set_data_dtype('uint8')
        out_img.set_qform(parcelation.get_qform())
        out_img.set_sform(parcelation.get_sform())
        out_file_name = self._gen_output_filename(in_file, roi_list)
        out_img.set_filename(out_file_name)
        out_img.to_filename(out_file_name)
        return runtime


class NormaliseImageWithROIInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,
                   mandatory=True,
                   desc="Input file to normalise")
    roi_file = File(mandatory=True,
                    desc="ROI file to use for normalisation")


class NormaliseImageWithROIOutputSpec(TraitedSpec):
    out_file = File(desc="Output file")


class NormaliseImageWithROI(BaseInterface):
    """

    Examples
    --------

    """

    input_spec = NormaliseImageWithROIInputSpec
    output_spec = NormaliseImageWithROIOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_output_filename(self.inputs.in_file)
        return outputs

    @staticmethod
    def _gen_output_filename(in_file):
        _, bn, ext = split_filename(in_file)
        outfile = os.path.abspath(bn + '_norm' + ext)
        return outfile

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        roi_file = self.inputs.roi_file
        # Load the input image
        image = nib.load(in_file)
        data = image.get_data()
        # Load the roi
        roi = nib.load(roi_file).get_data()
        # Extract the normalisation value
        mask = (roi != 0)
        normalisation_value = np.mean(data[mask])
        # Normalise the input image
        norm_data = np.divide(data, normalisation_value)
        # Create a new image based on the initial parcelation
        out_img = nib.Nifti1Image(norm_data, image.get_affine())
        out_img.set_data_dtype(image.get_data_dtype())
        [mat, code] = image.get_qform(coded=True)
        out_img.set_qform(mat, code=np.int(code))
        [mat, code] = image.get_sform(coded=True)
        out_img.set_sform(mat, code=np.int(code))
        out_file_name = self._gen_output_filename(in_file)
        out_img.set_filename(out_file_name)
        out_img.to_filename(out_file_name)
        return runtime


class FilenamesToTextFileInputSpec(BaseInterfaceInputSpec):
    in_files = traits.List(File(exists=True))


class FilenamesToTextFileOutputSpec(TraitedSpec):
    out_file = traits.File(desc='Output text file containing all filenames')


class FilenamesToTextFile(BaseInterface):
    input_spec = FilenamesToTextFileInputSpec
    output_spec = FilenamesToTextFileOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_output_filename(self.inputs.in_files)
        return outputs

    @staticmethod
    def _gen_output_filename(in_files):
        _, bn, _ = split_filename(in_files[0])
        outfile = os.path.abspath(bn + '_list.txt')
        return outfile

    def _run_interface(self, runtime):
        files = self.inputs.in_files
        filename = self._gen_output_filename(files)
        out_file = open(filename, 'w')
        for i in range(len(files)):
            out_file.write("%s\n" % files[i])
        out_file.close()
        return runtime


class CombineTextFilesInputSpec(BaseInterfaceInputSpec):
    in_files = traits.List(File(exists=True))


class CombineTextFilesOutputSpec(TraitedSpec):
    out_file = traits.File(desc='Output text file')


class CombineTextFiles(BaseInterface):
    input_spec = CombineTextFilesInputSpec
    output_spec = CombineTextFilesOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._gen_output_filename(self.inputs.in_files)
        return outputs

    @staticmethod
    def _gen_output_filename(in_files):
        _, bn, _ = split_filename(in_files[0])
        outfile = os.path.abspath(bn + '_concat.txt')
        return outfile

    def _run_interface(self, runtime):
        files_to_read = self.inputs.in_files
        out_filename = self._gen_output_filename(files_to_read)
        out_file = open(out_filename, 'w')
        for i in range(len(files_to_read)):
            in_file = open(files_to_read[i], 'r')
            out_file.write("%s\n" % in_file.read())
        out_file.close()
        return runtime


class ProduceMaskInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True)
    use_nrr = traits.Bool(desc='Use non-linear registration to propagate the mask. Only affine is used by default',
                          default_value=False)


class ProduceMaskOutputSpec(TraitedSpec):
    out_file = traits.File()


class ProduceMask(BaseInterface):

    input_spec = ProduceMaskInputSpec
    output_spec = ProduceMaskOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = os.path.abspath('MNI152_T1_2mm_brain_mask_res_fill.nii.gz')
        return outputs

    def _run_interface(self, runtime):
        mni_template = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm.nii.gz')
        mni_template_mask = os.path.join(os.environ['FSLDIR'], 'data', 'standard', 'MNI152_T1_2mm_brain_mask.nii.gz')
        in_file = self.inputs.in_file
        mni2input = niftyreg.RegAladin()
        mni2input.inputs.verbosity_off_flag = True
        mni2input.inputs.ref_file = in_file
        mni2input.inputs.flo_file = mni_template
        mni2input_res = mni2input.run()
        mask_resample = niftyreg.RegResample(inter_val='NN')
        if self.inputs.use_nrr:
            mni2input_nrr = niftyreg.RegF3D()
            mni2input_nrr.inputs.verbosity_off_flag = True
            mni2input_nrr.inputs.ref_file = in_file
            mni2input_nrr.inputs.flo_file = mni_template
            mni2input_nrr.inputs.aff_file = mni2input_res.outputs.aff_file
            mni2input_nrr.inputs.vel_flag = True
            mni2input_nrr_res = mni2input_nrr.run()
            mask_resample.inputs.trans_file = mni2input_nrr_res.outputs.cpp_file
        else:
            mask_resample.inputs.trans_file = mni2input_res.outputs.aff_file
        mask_resample.inputs.ref_file = in_file
        mask_resample.inputs.flo_file = mni_template_mask
        mask_resample_res = mask_resample.run()
        fill_mask = niftyseg.UnaryMaths(operation='fill')
        fill_mask.inputs.in_file = mask_resample_res.outputs.out_file
        fill_mask.run()
        return runtime


class SplitB0DWIsFromFileInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,
                   mandatory=True,
                   desc="Input 4D file")
    in_bval = File(exists=True,
                   mandatory=True,
                   desc="Input bval file")
    in_bvec = File(exists=True,
                   mandatory=True,
                   desc="Input bvec file")
    in_bval_threshold = traits.Float(mandatory=False,
                                     desc="bvalue threshold",
                                     default_value=10.0,
                                     usedefault=True)


class SplitB0DWIsFromFileOutputSpec(TraitedSpec):
    out_B0s = OutputMultiPath(File(exists=True))
    out_DWIs = OutputMultiPath(File(exists=True))
    out_all = OutputMultiPath(File(exists=True))
    out_indices = traits.Array(desc="B0 Indices in the table")


class SplitB0DWIsFromFile(BaseInterface):
    """

    Examples
    --------

    """

    input_spec = SplitB0DWIsFromFileInputSpec
    output_spec = SplitB0DWIsFromFileOutputSpec

    def _run_interface(self, runtime):
        from dipy.core import gradients
        splitter = fsl.Split(dimension='t', out_base_name='dwi_', in_file=self.inputs.in_file)
        gtab = gradients.gradient_table(bvals=self.inputs.in_bval, bvecs=self.inputs.in_bvec,
                                        b0_threshold=self.inputs.in_bval_threshold)
        masklist = list(gtab.b0s_mask)
        split_outputs = splitter.run().outputs.out_files
        self._b0s, self._dwis, self._all, self._indices = self.extract_sublists(split_outputs, masklist)
        return runtime

    @staticmethod
    def extract_sublists(array, mask_array):
        l1 = [array[i] for i in range(len(array)) if mask_array[i] == True]
        l2 = [array[i] for i in range(len(array)) if mask_array[i] == False]
        l3 = array
        ids = [i for i in range(len(array)) if mask_array[i] == True]
        return l1, l2, l3, ids

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_B0s'] = self._b0s
        outputs['out_DWIs'] = self._dwis
        outputs['out_all'] = self._all
        outputs['out_indices'] = self._indices
        return outputs


class SplitAndSelectInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True,
                   mandatory=True,
                   desc="Input 4D file")
    in_id = traits.Int(mandatory=False,
                       desc="Index",
                       default_value=0,
                       usedefault=True)


class SplitAndSelectOutputSpec(TraitedSpec):
    out_file = File(exists=True)


class SplitAndSelect(BaseInterface):
    """

    Examples
    --------

    """

    input_spec = SplitAndSelectInputSpec
    output_spec = SplitAndSelectOutputSpec

    def _run_interface(self, runtime):
        splitter = fsl.Split(dimension='t', out_base_name='out_', in_file=self.inputs.in_file)
        split_outputs = splitter.run().outputs.out_files
        if len(split_outputs[0]) == 1:
            split_outputs = [split_outputs]
        self._out_file = self.extract_index(split_outputs, self.inputs.in_id)
        return runtime

    @staticmethod
    def extract_index(array, index):
        return array[index]

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = self._out_file
        return outputs


class ExtractBaseNameInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=False,
                   desc="Input file")


class ExtractBaseNameOutputSpec(TraitedSpec):
    out_basename = traits.String()


class ExtractBaseName(BaseInterface):
    input_spec = ExtractBaseNameInputSpec
    output_spec = ExtractBaseNameOutputSpec

    def _run_interface(self, runtime):
        self._out_basename = split_filename(self.inputs.in_file)[1]
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_basename'] = self._out_basename
        return outputs



class IdentityMatrixInputSpec(BaseInterfaceInputSpec):
    flo_file = File(exists=True)


class IdentityMatrixOutputSpec(TraitedSpec):
    aff_file = traits.File()


class IdentityMatrix(BaseInterface):
    """Create an identity affine matrix. If the flo_file input is
    provided then the output aff_file is named in the same manner
    as the RegAladin interface.
    """
    input_spec = IdentityMatrixInputSpec
    output_spec = IdentityMatrixOutputSpec

    def _gen_filename(self, name):
        if name == 'aff_file':
            if (isdefined(self.inputs.flo_file)):
                _, final_bn, final_ext = split_filename(self.inputs.flo_file)
            else:
                final_bn = "identity"
            return os.path.join(os.getcwd(),
                                final_bn + "_aff.txt")
        else:
            return None

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['aff_file'] = self._gen_filename('aff_file')
        return outputs

    def _run_interface(self, runtime):
        outputs = self._list_outputs()
        aff_file = outputs['aff_file']
        aff = np.diag(4*[1])
        np.savetxt(aff_file, aff, "%g")
        return runtime


class NoRegAladinInputSpec(niftyreg.reg.RegAladinInputSpec):
    pass


class NoRegAladinOutputSpec(niftyreg.reg.RegAladinOutputSpec):
    pass


class NoRegAladin(BaseInterface):
    """Interface provides an alternative to RegAladin, but simply copies float image
    and outputs identity file. Note that copying the float image means outputs will
    not necessarily share the same resolution and dimensions as the reference image."""
    input_spec = NoRegAladinInputSpec
    output_spec = NoRegAladinOutputSpec

    def _gen_fname(self, basename, out_dir=None, suffix=None, ext=None):
        if basename == '':
            msg = 'Unable to generate filename for command %s. ' % self.cmd
            msg += 'basename is not set!'
            raise ValueError(msg)
        _, final_bn, final_ext = split_filename(basename)
        if out_dir is None:
            out_dir = os.getcwd()
        if ext is not None:
            final_ext = ext
        if suffix is not None:
            final_bn = ''.join((final_bn, suffix))
        return os.path.abspath(os.path.join(out_dir, final_bn + final_ext))

    def _gen_filename(self, name):
        if name == 'aff_file':
            return self._gen_fname(self.inputs.flo_file, suffix='_aff', ext='.txt')
        if name == 'res_file':
            return self._gen_fname(self.inputs.flo_file, suffix='_res', ext='.nii.gz')
        return None


    def _list_outputs(self):
        outputs = self.output_spec().get()

        if isdefined(self.inputs.aff_file):
            outputs['aff_file'] = self.inputs.aff_file
        else:
            outputs['aff_file'] = self._gen_filename('aff_file')
        if isdefined(self.inputs.res_file):
            outputs['res_file'] = self.inputs.aff_file
        else:
            outputs['res_file'] = self._gen_filename('res_file')
        # Make a list of the linear transformation file and the input image
        outputs['avg_output'] = os.path.abspath(outputs['aff_file']) + ' ' + os.path.abspath(self.inputs.flo_file)
        return outputs

    def _run_interface(self, runtime):
        outputs = self._list_outputs()
        aff_file = outputs['aff_file']
        aff = np.diag(4*[1])
        np.savetxt(aff_file, aff, "%g")
        rename = niu.Rename()
        rename.inputs.in_file = self.inputs.flo_file
        rename.inputs.format_string = outputs['res_file']
        result = rename.run()
        return result.runtime



class CopyFileInputSpec(CommandLineInputSpec):
    in_file = File(mandatory=True, exists=True,
                    desc="File to be copied", position=1)

    out_file = File(desc="name of copied file")


class CopyFileOutputSpec(TraitedSpec):
    out_file = File()


class CopyFile(CommandLine):
    _cmd = "cp"
    input_spec = CopyFileInputSpec
    output_spec = CopyFileOutputSpec


class RemoveFileInputSpec(CommandLineInputSpec):
    in_file = File(mandatory=True, exists=True,
                    desc="File to be deleted", position=1)
    wait_node = File(desc='waiting this output before delete in_file')


class RemoveFile(CommandLine):
    _cmd = "rm -rf"
    input_spec = RemoveFileInputSpec


class SwapOrientImageInputSpec(CommandLineInputSpec):
    image2reorient = File(argstr="-swaporient %s", mandatory=True, exists=True,
                          desc="Image to reorient. Change the image, not the header.")


class SwapOrientImageOutputSpec(TraitedSpec):
    reoriented_image = File(exists=True)


class SwapOrientImage(CommandLine):
    input_spec = SwapOrientImageInputSpec
    output_spec = SwapOrientImageOutputSpec
    _cmd = "fslorient"

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['reoriented_image'] = os.path.abspath(self.inputs.image2reorient)
        return outputs


class SwapDimImageInputSpec(CommandLineInputSpec):
    image2reorient = File(mandatory=True, exists=True,
                          desc="Input image to be flipped. The image is swapped, not the header")
    axe2flip = traits.Enum('LR', 'IS', 'AP', mandatory=True,
                           desc="Input image to be flipped. The image is swapped, not the header")


class SwapDimImageOutputSpec(TraitedSpec):
    flipped_image = File(desc="flipped image", exists=True)


class SwapDimImage(BaseInterface):
    input_spec = SwapDimImageInputSpec
    output_spec = SwapDimImageOutputSpec

    def _run_interface(self, runtime):
        in_image = nib.load(self.inputs.image2reorient)
        axe = self.inputs.axe2flip
        matrix = in_image.get_data()
        matrix_swapdim = np.zeros(matrix.shape)
        print matrix.shape
        # Need to check which axe x,y,z corresponds to this axe:
        if axe == 'LR':
            out_shell = os.popen('fslhd ' + os.path.abspath(self.inputs.image2reorient) + ' | grep "Left" ').read()

        elif axe == 'IS':
            out_shell = os.popen('fslhd ' + os.path.abspath(self.inputs.image2reorient) + ' | grep "Inferior" ').read()

        elif axe == 'AP':
            out_shell=os.popen('fslhd ' + os.path.abspath(self.inputs.image2reorient) + ' | grep "Posterior" ').read()

        ind_orient = out_shell.find('orient')
        axe_xyz = out_shell[ind_orient-1]
        if axe_xyz == 'x':
            size_axe_swap = matrix.shape[0]
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    for k in range(matrix.shape[2]):
                        matrix_swapdim[i, j, k] = matrix[size_axe_swap - 1 - i, j, k]
        elif axe_xyz == 'y':
            print 'Swapping axe y'
            size_axe_swap = matrix.shape[1]
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    for k in range(matrix.shape[2]):
                        matrix_swapdim[i, j, k] = matrix[i, size_axe_swap - 1 - j, k]
        elif axe_xyz == 'z':
            print 'Swapping axe z'
            size_axe_swap = int(matrix.shape[2])
            print "size_axe_swap = " + str(size_axe_swap)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    for k in range(matrix.shape[2]):
                        matrix_swapdim[i, j, k] = matrix[i, j, size_axe_swap - 1 - k]
        else:
            print "image = " + os.path.abspath(self.inputs.image2reorient)
            print "out_shell = " + out_shell
            print "ind_orient = " + str(ind_orient)
            raise ValueError

        swapped_image = nib.Nifti1Image(matrix_swapdim, in_image.get_affine())
        self.outputfile = self._gen_output_filename(self.inputs.image2reorient, axe_xyz)
        nib.save(swapped_image, self.outputfile)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['flipped_image'] = self.outputfile
        return outputs

    @staticmethod
    def _gen_output_filename(in_file, axe):
        p, bn, ext = split_filename(in_file)
        outfile = os.path.abspath(bn + '_swapDim_axe_' + axe + ext)
        return outfile


class isListOfListInputSpec(BaseInterfaceInputSpec):
    in_list = traits.List(mandatory=True)


class isListOfListOutputSpec(TraitedSpec):
    bool_listOfList = traits.Bool()


class isListOfList(BaseInterface):

    input_spec = isListOfListInputSpec
    output_spec = isListOfListOutputSpec

    def _run_interface(self, runtime):
        self.bool_listOfList = None
        if isinstance(self.inputs.in_list, list):
            if isinstance(self.inputs.in_list[0], list):
                self.bool_listOfList = True
            else:
                self.bool_listOfList = False
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['bool_listOfList'] = self.bool_listOfList
        return outputs


class extractSubListInputSpec(BaseInterfaceInputSpec):
    in_list = traits.List(mandatory=True)
    k = traits.Int(mandatory=True, desc="length of the sub list to be extracted")
    sorting_reference = traits.List(desc='if given, will use the k elements of in_list with the smallest ' +
                                         'values in sorting_reference. Otherwise, the node will return ' +
                                         'the k firsts elements of in_list.')


class extractSubListOutputSpec(TraitedSpec):
    out_sublist = traits.List()
    sorted_list_ref = traits.List()

class extractSubList(BaseInterface):

    input_spec = extractSubListInputSpec
    output_spec = extractSubListOutputSpec

    def _run_interface(self, runtime):
        self.out_sublist = []
        list_in = self.inputs.in_list
        if self.inputs.k > len(list_in):
            if isdefined(self.inputs.sorting_reference):
                list_ref = self.inputs.sorting_reference
                self.sorted_list_ref = sorted(list_ref)
                sorted_index_ref = [list_ref.index(x) for x in self.sorted_list_ref]
                self.out_sublist = [list_in[index] for index in sorted_index_ref]
                print self.out_sublist
            else:
                self.out_sublist = self.inputs.in_list
        else:
            if isdefined(self.inputs.sorting_reference):
                list_ref = self.inputs.sorting_reference
                self.sorted_list_ref = sorted(list_ref)
                sorted_index_ref = [list_ref.index(x) for x in self.sorted_list_ref]
                c = [list_in[index] for index in sorted_index_ref]
                self.out_sublist = [c[index] for index in range(self.inputs.k)]
            else:
                self.out_sublist = [list_in[index] for index in range(self.inputs.k)]
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_sublist'] = self.out_sublist
        outputs['sorted_list_ref'] = self.sorted_list_ref
        print outputs['out_sublist']
        return outputs
