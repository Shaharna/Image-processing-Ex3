#------------- Imports -----------
import numpy as np
import scipy.ndimage.filters
import scipy.signal
from imageio import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import os
#------------- Constants -----------
GRAYSCALE_SHAPE = 2
RGB_SHAPE = 3
GRAY_MAX_VALUE = 255

#------------- Functions -----------

def read_image(filename, representation):
    """
    3.1 Reading an image into a given representation.
    :param filename: read_image(filename, representation).
    :param representation: representation code, either 1 or 2 defining
    whether the output should be a grayscale image (1) or an RGB image (2).
    If the input image is grayscale, we won’t call it with representation = 2.
    :return: This function returns an image, make sure the output image
    is represented by a matrix of type np.float64 with intensities
    (either grayscale or RGB channel intensities)
    normalized to the range [0, 1].
    """
    im = (imread(filename).astype(np.float64)) / GRAY_MAX_VALUE

    if representation == 1:

        return rgb2gray(im)

    elif representation == 2:

        return (im)


def build_binomial_array(binomial_size):
    """
    This function returns a 1D array which is the binomial array
    including binomial size coefficients.
    :param binumial_size: The number of binomial coefficients
    :return:  1D array which is the binomial array
    including binomial size coefficients.
    """
    if binomial_size == 1:
        return np.array([1])

    base_conv = np.array([1, 1]).reshape(1,2)
    filter_array = np.array([1, 1]).reshape(1,2)

    while filter_array.size < binomial_size:
        filter_array = scipy.signal.convolve2d(filter_array, base_conv)

    return filter_array.astype(np.float64)


def blur(im, filter_1D):
    """
    This function blurs the image with a gaussian filter
    :return: the image after it was blur
    """
    im = scipy.ndimage.filters.convolve(im, filter_1D)

    return scipy.ndimage.filters.convolve(im, filter_1D.T)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    This function constructs a gaussian_pyramid from the given image with
    max_levels levels and gaussian filter size at filter_size.
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels1 in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that
    represents a squared filter) to be used in constructing the pyramid filter
    (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]).
    :return: pyr, filter_vec.
    pyramid pyr as a standard python array (i.e. not numpy’s array)
    with maximum length of max_levels, where each element of the array is a
    grayscale image.
    The functions should also output filter_vec which is row vector of shape
    (1, filter_size) used for the pyramid construction.
    This filter should be built using a consequent 1D convolutions of [1 1]
    with itself in order to derive a row of the binomial coefficients which
    is a good approximation to the Gaussian profile.
    The filter_vec is be normalized.
    """
    new_im = np.copy(im)

    gaussian_pyramid = [im]

    filter_1D = build_binomial_array(filter_size)

    normalize_coefficient = np.sum(filter_1D)

    filter = (filter_1D /normalize_coefficient)

    # main loop
    while new_im.shape[0] >= 16 and new_im.shape[1] >= 16 and \
            len(gaussian_pyramid) < max_levels:

        new_im = reduce_image(filter, new_im)

        gaussian_pyramid.append(new_im)

    return gaussian_pyramid, filter


def reduce_image(filter, im):
    """
    This function reduces the size of a given image after blurring it with filter.
    :param filter: The filter to blur the image with
    :param new_im: a RGB or grayscale image
    :return: The reduced image
    """
    x = im.shape[0] // 2
    y = im.shape[1] // 2

    if len(im.shape) == RGB_SHAPE:
        new_im = np.zeros((x, y, RGB_SHAPE), dtype=im.dtype)

        for i in range(RGB_SHAPE):
            blurred_color = blur(im[:, :, i], filter)
            new_im[:, :, i] = (blurred_color[::2, ::2])

    elif len(im.shape) == GRAYSCALE_SHAPE:
        blurred_im = blur(im, filter)
        new_im = blurred_im[::2, ::2]

    return new_im

def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    This function constructs a laplacian_pyramid from the given image with
    max_levels levels and gaussian_filter size at filter_size.
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels1 in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter (an odd scalar that
    represents a squared filter) to be used in constructing the pyramid filter
    (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]).
    :return: pyr, filter_vec.
    pyramid pyr as a standard python array (i.e. not numpy’s array)
    with maximum length of max_levels, where each element of the array is a
    grayscale image.
    The functions should also output filter_vec which is row vector of shape
    (1, filter_size) used for the pyramid construction.
    This filter should be built using a consequent 1D convolutions of [1 1]
    with itself in order to derive a row of the binomial coefficients which
    is a good approximation to the Gaussian profile.
    The filter_vec is be normalized.
    """
    gaussian_pyramid, filter_1D = build_gaussian_pyramid(im, max_levels, filter_size)

    laplacian_pyramid = []

    for i in range(len(gaussian_pyramid) - 1):

        expanded_im = expand_image(gaussian_pyramid[i+1], 2 * filter_1D)

        laplacian_pyramid.append(gaussian_pyramid[i] - expanded_im)

    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid, filter_1D

def expand_image(im, filter):
    """
    This function expands the given image (adds 0 at odd indexes)
    :return: The expanded image
    """
    new_x_size = 2 * im.shape[0]
    new_y_size = 2 * im.shape[1]
    expanded_im = []

    if len(im.shape) == RGB_SHAPE:
        expanded_im = np.zeros((new_x_size, new_y_size, RGB_SHAPE), dtype=im.dtype)
        # adding 0 in the odd places = taking the original even places of each color
        expanded_im[::2, ::2, :] = im

        for i in range(RGB_SHAPE):
            expanded_im[:, :, i] = blur(expanded_im[:, :, i], filter)

    elif len(im.shape) == GRAYSCALE_SHAPE:
        expanded_im = np.zeros((new_x_size, new_y_size), dtype=im.dtype)
        # adding 0 in the odd places = taking the original even places
        expanded_im[::2, ::2] = im
        expanded_im = blur(expanded_im, filter)

    return expanded_im

def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    This function implement the reconstruction of an image from its Laplacian
    Pyramid.
    :param lpyr: is the Laplacian pyramid is generated by the function -
    build_laplacian_pyramid.
    :param filter_vec: is the filter that is generated by the function -
    build_laplacian_pyramid
    :param coeff: a python list.
    The list length is the same as the number of levels in the pyramid lpyr.
    Before reconstructing the image img I multiply each level i of the
    laplacian pyramid by its corresponding coefficient coeff[i].
    :return: The image as it was constructed from summing all the laplcian
     levels.
    """
    expended_im = lpyr[-1] * coeff[-1]

    for i in range(len(lpyr)-2, 0, -1):

        expended_im = coeff[i] * lpyr[i] + expand_image(expended_im, filter_vec)

    return scale_image(lpyr[0] + expand_image(expended_im, filter_vec))

def render_pyramid(pyr, levels):
    """
    This function render the pyramids.
    :param pyr: is either a Gaussian or Laplacian pyramid.
    :param levels: is the number of levels to present in the result ≤ max_levels.
    :return: a single black image in which the pyramid levels of the given
    pyramid pyr are stacked horizontally (after stretching the values to [0, 1]).
    """
    if levels == 1:
        return scale_image(pyr[0])

    new_pyr = [scale_image(pyr[0])]
    orig_x = pyr[0].shape[0]

    for i in range(1, levels):
        x = orig_x - pyr[i].shape[0]
        scaled_im = scale_image(pyr[i])
        new_im = np.pad(scaled_im, ((0, x),(0, 0)), 'constant', constant_values=(0))
        new_pyr.append(new_im)

    res = np.hstack(new_pyr[:levels])
    return res

def scale_image(im, range = (0, 1)):
    """
    This function scale the image pixels to the given range.
    :param im: The image to scale
    :param range: The destination range
    :return: The scaled image
    """
    im_min = im.min()
    im_max = im.max()

    temp_im = im - im_min

    return (temp_im / im_max)

def display_pyramid(pyr, levels):
    """
    This function use render_pyramid to internally render and then display
    the stacked pyramid image using plt.imshow().
    :param pyr: is either a Gaussian or Laplacian pyramid.
    :param levels: is the number of levels to present in the result ≤ max_levels.
    :return:
    """
    im_stack = render_pyramid(pyr, levels)

    plt.figure()

    if len(pyr[0].shape) == GRAYSCALE_SHAPE:
        plt.imshow(im_stack, cmap="gray")

    elif len(pyr[0].shape) == RGB_SHAPE:
        plt.imshow(im_stack)

    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    Implement pyramid blending as described in the lecture.
    :param im1: input grayscale images to be blended.
    :param im2: input grayscale images to be blended.
    :param mask: is a boolean (i.e. dtype == np.bool)
    mask containing True and False representing which parts of im1 and im2
    should appear in the resulting im_blend. Note that a value of
    True corresponds to 1, and False corresponds to 0.
    :param max_levels: is the max_levels parameter you should use when
    generating the Gaussian and Laplacian pyramids.
    :param filter_size_im:  is the size of the Gaussian filter
    (an odd scalar that represents a squared filter) which defining the filter
    used in the construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: is the size of the Gaussian filter
    (an odd scalar that represents a squared filter) which defining the filter
    used in the construction of the Gaussian pyramid of mask.
    :return: grayscale image in the range [0, 1].
    """
    lypr1, filter1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lypr2, filter2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)

    gypr_mask, mask_filter = build_gaussian_pyramid(mask.astype(np.double),
                                                    max_levels,
                                                 filter_size_mask)
    blended_lypr = []

    for k in range(len(gypr_mask)):
        blended_lypr.append(
            np.multiply(gypr_mask[k],lypr1[k]) + np.multiply(1 - gypr_mask[k], lypr2[k]))

    new_filter_vec = 2 * filter1
    coeff_filter = np.ones(len(blended_lypr))
    new_im = laplacian_to_image(blended_lypr, new_filter_vec, coeff_filter)
    scaled_im = scale_image(new_im)

    return scaled_im.astype(np.float64)


def blending_example1():
    """
    This function is performing pyramid blending on two sets of image pairs and
    masks I find nice.
    :return:
    im1, im2, mask, im_blend : the two images (im1 and im2),
    the mask (mask) and the resulting blended image (im_blend).
    """
    im1 = read_image(relpath("Lenna.jpg"), 2)
    im2 = read_image(relpath("voldemort5.jpg"), 2)
    mask = read_image(relpath("mask1.jpg"), 1).astype(np.bool)


    blended_im = np.zeros(im1.shape, dtype=np.float64)

    for i in range(RGB_SHAPE) :

        im1_color = im1[:, :, i]
        im2_color = im2[:, :, i]
        blended_im[:, :, i] = pyramid_blending(im1_color, im2_color, mask, 4, 5, 5).astype(im1.dtype)

    display_plot(im1, im2, mask, blended_im)

    return im1, im2, mask, blended_im


def blending_example2():
    """
    This function is performing pyramid blending on two sets of image pairs and
    masks I find nice.
    :return:
    im1, im2, mask, im_blend : the two images (im1 and im2),
    the mask (mask) and the resulting blended image (im_blend).
    """
    im1 = read_image(relpath("21.jpg"), 2)
    im2 = read_image(relpath("22.jpg"), 2)
    mask = read_image(relpath("mask2.jpg"), 1).astype(np.bool)


    blended_im = np.zeros((im1.shape[0], im1.shape[1], RGB_SHAPE), dtype=np.float64)

    for i in range(RGB_SHAPE) :

        im1_color = im1[:, :, i]
        im2_color = im2[:, :, i]
        blended_im[:, :, i] = pyramid_blending(im1_color, im2_color, mask, 10, 5, 5)

    display_plot(im1, im2, mask, blended_im)

    return im1, im2, mask, blended_im


def display_plot(im1, im2, mask, blended_im):
    """
    This function plot 4 images at the same figure
    :param im1:
    :param im2:
    :param mask:
    :param blended_im:
    :return:
    """
    plt.figure()

    plt.subplot(221)
    plt.imshow(im1)

    plt.subplot(222)
    plt.imshow(im2)

    plt.subplot(223)
    plt.imshow(mask, cmap="gray")

    plt.subplot(224)
    plt.imshow(blended_im)
    plt.imsave('blended.jpg', blended_im)
    plt.show()


def relpath(filename):
    """

    :param filename:
    :return:
    """
    return os.path.join(os.path.dirname(__file__), filename)

if __name__ == '__main__':

    im1, im2, mask, blended_im = blending_example1()
    # im1, im2, mask, blended_im = blending_example2()