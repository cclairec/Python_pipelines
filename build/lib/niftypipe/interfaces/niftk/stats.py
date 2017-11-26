# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from nipype.interfaces.base import (TraitedSpec, File, traits, BaseInterface, BaseInterfaceInputSpec)
import numpy as np
import nibabel as nib
from scipy import linalg as la


class ComputeDiceScoreInputSpec(BaseInterfaceInputSpec):
    
    in_file1 = File(argstr="%s",
                    exists=True,
                    mandatory=True,
                    desc="First roi image")
    in_file2 = File(argstr="%s",
                    exists=True,
                    mandatory=True,
                    desc="Second roi image")


class ComputeDiceScoreOutputSpec(TraitedSpec):
    out_dict = traits.Dict(desc='Output array containing the dice score.\n' +
                                'The first value is the label index and the second is the Dice score')


class ComputeDiceScore(BaseInterface):
    input_spec = ComputeDiceScoreInputSpec
    output_spec = ComputeDiceScoreOutputSpec

    def _run_interface(self, runtime):
        roi_file1 = self.inputs.in_file1
        roi_file2 = self.inputs.in_file2
        self.out_dict = self.compute_dice_score(roi_file1, roi_file2)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_dict'] = self.out_dict
        return outputs

    @staticmethod
    def compute_dice_score(roi_file1,
                           roi_file2):

        # Read the input images
        in_img1 = nib.load(roi_file1).get_data()
        in_img2 = nib.load(roi_file2).get_data()

        # Get the min and max label values
        min_label_value = np.int32(np.min([np.min(in_img1), np.min(in_img2)]))
        max_label_value = np.int32(np.max([np.max(in_img1), np.max(in_img2)]))

        # Iterate over all label values
        out_dict = dict()
        for l in range(int(min_label_value), int(max_label_value) + 1):
            mask1 = in_img1 == l
            mask2 = in_img2 == l
            mask3 = np.multiply(in_img1 == l, in_img2 == l)
            if np.sum(mask1) + np.sum(mask2) != 0:
                out_dict[l] = 2 * float(np.sum(mask3)) / float(np.sum(mask1) + np.sum(mask2))

        return out_dict


class CalculateAffineDistancesInputSpec(BaseInterfaceInputSpec):
    transformation1_list = traits.List(traits.File(exists=True), mandatory=True, desc='List of affine transformations')
    transformation2_list = traits.List(traits.File(exists=True), mandatory=True, desc='List of affine transformations')


class CalculateAffineDistancesOutputSpec(TraitedSpec):
    out_array = traits.Array(desc='Array of distances between the paired transformations')


class CalculateAffineDistances(BaseInterface):
    """

    Examples
    --------

    """
    input_spec = CalculateAffineDistancesInputSpec
    output_spec = CalculateAffineDistancesOutputSpec

    def _run_interface(self, runtime):
        transformation1_list = self.inputs.transformation1_list
        transformation2_list = self.inputs.transformation2_list
        self.distances = self.calculate_distance_between_affines(transformation1_list,
                                                                 transformation2_list)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_array'] = self.distances
        return outputs

    @staticmethod
    def read_file_to_matrix(file_name):

        return np.genfromtxt(file_name)

    def calculate_distance_between_affines(self, list1_aff, list2_aff):
        distances = np.zeros((len(list1_aff), 1))
        for i in range(len(list1_aff)):
            # Read affine matrices
            file1 = list1_aff[i]
            file2 = list2_aff[i]
            mat1 = self.read_file_to_matrix(file1)
            mat2 = self.read_file_to_matrix(file2)
            log_mat1 = la.logm(mat1)
            log_mat2 = la.logm(mat2)

            distances[i, 0] = ((log_mat2 - log_mat1)*(log_mat2 - log_mat1)).sum()

        return distances
        

class ExtractRoiStatisticsInputSpec(BaseInterfaceInputSpec):
    
    in_file = File(argstr="%s",
                   exists=True,
                   mandatory=True,
                   desc="Input image to extract the statistics from")
    roi_file = File(argstr="%s",
                    exists=True,
                    mandatory=True,
                    desc="Input image that contains the different roi")
    weight_file = File(argstr="%s",
                       exists=True,
                       mandatory=False,
                       desc="Input weight image (0 to 1)")
    in_label = traits.List(traits.BaseInt,
                           mandatory=False,
                           desc="Label value(s) to extract")
    in_threshold = traits.Float(argstr='%f',
                                mandatory=False,
                                desc='Input threshold to use for the weight file to compute the statistics',
                                default=None)


class ExtractRoiStatisticsOutputSpec(TraitedSpec):
    out_array = traits.Array(desc="Output array organised as follow: " +
                                  "label index, mean value, std value, roi volume in mm")


class ExtractRoiStatistics(BaseInterface):
    """

    Examples
    --------

    """

    input_spec = ExtractRoiStatisticsInputSpec
    output_spec = ExtractRoiStatisticsOutputSpec

    def _run_interface(self, runtime):
        in_file = self.inputs.in_file
        roi_file = self.inputs.roi_file
        weight_file = self.inputs.weight_file
        label = self.inputs.in_label
        threshold = self.inputs.in_threshold
        self.stats = self.extract_roi_statistics(in_file, roi_file, label, weight_file, threshold)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_array'] = self.stats
        return outputs

    @staticmethod
    def extract_roi_statistics(in_file, roi_file, label, weight_file, threshold=None):

        # Load the input image
        in_image = np.array(nib.load(in_file).get_data().ravel())
        in_image_valid = in_image[np.isnan(in_image) == False]

        # load the parcelation
        roi_image = np.array(nib.load(roi_file).get_data().ravel())
        roi_image_valid = roi_image[np.isnan(in_image) == False]

        # load the weights
        weights_image_valid = None
        if weight_file:
            weights_image = np.array(nib.load(weight_file).get_data().ravel())
            weights_image_valid = weights_image[np.isnan(in_image) is False]

        # extract the number of label
        if label:
            unique_values = np.asarray(label)
        else:
            unique_values = np.unique(roi_image_valid)

        # Create an array to accumulate values
        regional_value = np.zeros((unique_values.size, 4))

        vox_dim = np.product(nib.load(in_file).get_header().get_zooms())

        # Loop over all labels
        index = 0
        for i in unique_values:
            mask = (roi_image_valid == i)
            n = np.sum(mask)

            values = in_image_valid[mask]
            weights = np.ones(n)
            if weight_file:
                weights = weights_image_valid[mask]
                if threshold:
                    weights[weights < threshold] = 0
                    weights[weights >= threshold] = 1
                    
            regional_value[index, 0] = i
            m = 0
            sd = 0
            if n > 0 and np.sum(weights) > 0:
                m = np.average(values, weights=weights)
                sd = np.sqrt(np.average((values-m)**2, weights=weights))

            regional_value[index, 1] = m
            regional_value[index, 2] = sd
            regional_value[index, 3] = n * vox_dim
            index += 1
        return regional_value
