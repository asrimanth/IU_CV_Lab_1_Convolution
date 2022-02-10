#Import the Image and ImageFilter classes from PIL (Pillow)
from PIL import Image
from PIL import ImageFilter
import sys

import numpy as np
from scipy import signal

def image_to_np_array(image_path):
    return np.asarray(Image.open(image_path))


def resize_image(image_np_array, shape):
    PIL_image = Image.fromarray(np.uint8(image_np_array)).convert('RGB')
    PIL_image = PIL_image.resize(shape)
    return np.array(PIL_image)


def pad_image_with_white(image_np_array, border_size):
    max_row = image_np_array.shape[0]
    max_col = image_np_array.shape[1]
    new_image = np.ones((max_row + 2*border_size, max_col + 2*border_size, 3), dtype='uint8') * 255
    new_image[border_size:border_size+max_row, border_size:border_size+max_col, :] = image_np_array
    return new_image


def custom_convolution(resized_image_np, kernel=np.ones((3,3)), output_filename = "demo_output.jpg"):
    # Decide the border size. Assume a kernel is square and has odd_number * odd_number of elements.
    kernel_size = kernel.shape[0]
    if(kernel_size <= 1):
        border_size = 0
    else:
        border_size = kernel_size // 2
    padded_image = pad_image_with_white(resized_image_np, border_size)

    # print(f"Before filtering, resized and padded image shape: {padded_image.shape}, border size: {border_size}")
    # plot_image(padded_image)
    target_row = padded_image.shape[0] - (2*border_size)
    target_col = padded_image.shape[1] - (2*border_size)
    result_image = np.zeros((target_row, target_col, 3))

    for rgb_index in range(3):
        for i in range(target_row):
            for j in range(target_col):
                mat = padded_image[i:i+kernel_size, j:j+kernel_size, rgb_index]
                result_image[i][j][rgb_index] = np.sum(np.multiply(mat, kernel))
    
    if np.min(result_image) < 0 or np.max(result_image) > 255:
        np.clip(result_image, 0, 255, out=result_image)

    # print(f"Target shape is {result_image.shape}")
    result_image = result_image.astype('uint8')
    PIL_result_image = Image.fromarray(np.uint8(result_image)).convert('RGB')
    PIL_result_image.save(output_filename)


def get_sharpen_filter(alpha, identity_filter, approximated_gaussian):
    sharpen = np.zeros((5,5))
    sharpen[1:4, 1:4] = identity_filter
    sharpen = ((1+alpha) * sharpen) - approximated_gaussian
    return sharpen


def get_all_kernels():
    kernel_dict = dict()
    identity_filter = np.zeros((3,3))
    identity_filter[1][1] = 1
    kernel_dict['a.jpg'] = identity_filter

    box_blur = np.ones((3,3)) * 1/9
    kernel_dict['b.jpg'] = box_blur

    horizontal_derivative = np.zeros((3,3))
    horizontal_derivative[1][0] = -1
    horizontal_derivative[1][2] = 1
    kernel_dict['c.jpg'] = horizontal_derivative

    approximated_gaussian = np.array([
        [0.003, 0.013, 0.022, 0.013, 0.003],
        [0.013, 0.059, 0.097, 0.059, 0.013],
        [0.022, 0.097, 0.159, 0.097, 0.022],
        [0.013, 0.059, 0.097, 0.059, 0.013],
        [0.003, 0.013, 0.022, 0.013, 0.003]
    ])
    kernel_dict['d.jpg'] = approximated_gaussian

    kernel_dict['e.jpg'] = get_sharpen_filter(0.9, identity_filter, approximated_gaussian)

    # gaussian_derivative = np.zeros((3,3))
    # gaussian_derivative[1:4, 1:4] = horizontal_derivative
    gaussian_derivative = signal.convolve2d(horizontal_derivative, approximated_gaussian)
    gaussian_derivative = gaussian_derivative[1:6, 1:6]
    kernel_dict['f.jpg'] = gaussian_derivative

    return kernel_dict


if __name__ == '__main__':
    # Load an image 
    image_path = sys.argv[1]
    kernels = get_all_kernels()

    image_np = image_to_np_array(image_path)
    resize_shape = (600, 800)
    resized_image_np = resize_image(image_np, resize_shape)

    for filename, kernel in kernels.items():
        custom_convolution(resized_image_np, kernel, filename)

