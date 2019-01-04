"""
tf_model_input_test.py

This file contains all of the functions for preprocessing and augmenting the
test data before it is input into the network.
"""

import os
import numpy as np
from KS_lib.image import KSimage

###############################################################
def generate_linear_gradient_for_merger(output_patch_size, stride):
    """
    generate_linear_gradient_for_merger calculates the relative ratios
    of overlapping pixels which are combined when all of the patches
    are stitched together

    param: output_patch
    param: stride
    return: factor_left, factor_right
    """
    start_left = 0
    center_left = int(np.ceil(output_patch_size / 2.0) - 1)
    end_left = output_patch_size - 1

    start_right = stride
    center_right = int(stride + np.ceil(output_patch_size / 2.0) - 1)
    end_right = stride + output_patch_size - 1

    if stride >= output_patch_size:
        factor_left = np.ones(output_patch_size)
        factor_right = np.ones(output_patch_size)
    else:
        factor_left = np.zeros(output_patch_size)
        factor_left[start_left:center_left + 1] = 1

        if start_right > center_left:
            factor_left[center_left:start_right] = 1
            overlap_start = start_right
        else:
            overlap_start = center_left + 1

        if center_right <= end_left:
            factor_left[center_right:end_left+1] = 0
            overlap_end = center_right - 1
        else:
            overlap_end = end_left
        seq = np.arange(1,overlap_end - overlap_start + 2,1)
        factor_left[overlap_start:overlap_end+1] = seq[::-1] \
                                                   / float(overlap_end - overlap_start + 2)

        factor_right = np.zeros(end_right+1)
        factor_right[center_right:(end_right+1)] = 1
        if end_left < center_right:
            factor_right[end_left + 1:center_right+1] = 1
            overlap_end = end_left
        else:
            overlap_end = center_right - 1

        if start_right <= center_left:
            factor_right[start_right:center_left+1] = 0
            overlap_start = center_left + 1
        else:
            overlap_start = start_right

        factor_right[overlap_start:overlap_end+1] = np.arange(1,overlap_end - overlap_start + 2,1) \
                                                    / float(overlap_end - overlap_start + 2)

        factor_right = factor_right[start_right:end_right+1]

    return factor_left, factor_right

#####################################################################################
def MergePatches_test(patches, stride, image_size, sizeInputPatch, sizeOutputPatch, flags):
    """
    MergePatches_test stitches all of the overlapping patches together using the
    provided stride value, resulting in the final output image

    param: patches
    param: stride
    param: image_size
    param: sizeInputPatch
    param: sizeOutputPatch
    param: size_input_patches
    return: image
    """

    patches = np.float32(patches)

    ntimes_row = int(np.floor((image_size[0] - sizeInputPatch[0]) / float(stride[0])) + 1)
    ntimes_col = int(np.floor((image_size[1] - sizeInputPatch[1]) / float(stride[1])) + 1)
    rowRange = range(0, ntimes_row * stride[0], stride[0])
    colRange = range(0, ntimes_col * stride[1], stride[1])

    displacement_row = int(round((sizeInputPatch[0] - sizeOutputPatch[0]) / 2.0))
    displacement_col = int(round((sizeInputPatch[1] - sizeOutputPatch[1]) / 2.0))

    image = np.zeros([image_size[0], image_size[1], patches.shape[3]], dtype=np.float32)

    factor_up_row, factor_down_row = generate_linear_gradient_for_merger(sizeOutputPatch[0], stride[0])
    factor_left_col, factor_right_col = generate_linear_gradient_for_merger(sizeOutputPatch[1], stride[1])

    factor_left_col = factor_left_col.reshape(1, len(factor_left_col), 1)
    factor_right_col = factor_right_col.reshape(1, len(factor_right_col), 1)
    factor_up_row = factor_up_row.reshape(len(factor_up_row), 1, 1)
    factor_down_row = factor_down_row.reshape(len(factor_down_row), 1, 1)

    factor_left_col = np.tile(factor_left_col, [sizeOutputPatch[0], 1, patches.shape[3]])
    factor_right_col = np.tile(factor_right_col, [sizeOutputPatch[0], 1, patches.shape[3]])
    factor_up_row = np.tile(factor_up_row, [1, image_size[1], patches.shape[3]])
    factor_down_row = np.tile(factor_down_row, [1, image_size[1], patches.shape[3]])

    ####################################################################################################################
    for index1, row in enumerate(rowRange):

        row_strip = np.zeros([sizeOutputPatch[0], image_size[1], patches.shape[3]], dtype=np.float32)

        for index2, col in enumerate(colRange):

            temp = patches[(index1 * len(colRange)) + index2, :, :, :]
            if index2 != 0:
                temp = temp * factor_right_col

            row_strip[:, col + displacement_col: col + displacement_col + sizeOutputPatch[1], :] += temp

            if index2 != len(colRange):
                row_strip[:, col + displacement_col : col + displacement_col + sizeOutputPatch[1],
                :] = row_strip[:, col + displacement_col : col + displacement_col + sizeOutputPatch[1],
                    :] * factor_left_col

        if index1 != 0:
            row_strip = row_strip * factor_down_row

        image[row + displacement_row: row + displacement_row + sizeOutputPatch[0], :, :] += row_strip

        if index1 != len(rowRange):
            image[row + displacement_row : row + displacement_row + sizeOutputPatch[0], :, :] = \
            image[row + displacement_row : row + displacement_row + sizeOutputPatch[0], :, :] * factor_up_row

    ################################################################################################################

    image = image[flags['size_input_patch'][0]:image.shape[0] - flags['size_input_patch'][0],
            flags['size_input_patch'][1]:image.shape[1] - flags['size_input_patch'][1], :]
    return image

#####################################################################################
def ExtractPatches_test(sizeInputPatch, stride, image):
    """
    ExtractPatches_test extracts patches from the original image
    using the specified input size and stride

    param: sizeInputPatch
    param: stride
    param: image
    return: generator with patches in it
    """

    ntimes_row = int(np.floor((image.shape[0] - sizeInputPatch[0])/float(stride[0])) + 1)
    ntimes_col = int(np.floor((image.shape[1] - sizeInputPatch[1])/float(stride[1])) + 1)
    rowRange = range(0, ntimes_row*stride[0], stride[0])
    colRange = range(0, ntimes_col*stride[1], stride[1])

    for index1, row in enumerate(rowRange):
        for index2, col in enumerate(colRange):
            yield (image[row:row + sizeInputPatch[0],col:col + sizeInputPatch[1], :])


#####################################################################################
def read_data_test(filename, flags):
    """
    read_data_test reads in the test image, resizes it
    from 10x to 2.5x, and extracts patches from the image

    param: filename
    param: stride_test
    return: patches, patches_mask, image.shape, nPatches
    """

    stride = flags['stride_test']

    image = KSimage.imread(filename)    
    if image.ndim == 2:
        image = np.expand_dims(image, axis=3)

    padrow = flags['size_input_patch'][0]
    padcol = flags['size_input_patch'][1]

    image = np.lib.pad(image, ((padrow, padrow), (padcol, padcol), (0, 0)), 'symmetric')


    print("INPUT SHAPE: " + str(image.shape))

    # extract patches
    patches = ExtractPatches_test(flags['size_input_patch'], stride, image)
    

    ntimes_row = int(np.floor((image.shape[0] - flags['size_input_patch'][0]) / float(stride[0])) + 1)
    ntimes_col = int(np.floor((image.shape[1] - flags['size_input_patch'][1]) / float(stride[1])) + 1)
    rowRange = range(0, ntimes_row * stride[0], stride[0])
    colRange = range(0, ntimes_col * stride[1], stride[1])

    nPatches = len(rowRange) * len(colRange)

    return patches, image.shape, nPatches


