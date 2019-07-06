import numpy as np
from skimage import io, viewer, exposure
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib
import easygui
import math

def convolution_zoom_in(img_matrix, mask_matrix):
    flat_img = img_matrix.flatten()
    flat_mask = mask_matrix.flatten()
    return sum([a*b for a, b in zip(flat_img, flat_mask)])


def convolution(input_img, mask_matrix):
    x_size, y_size = input_img.shape
    output_img = np.empty(input_img.shape)
    for x_pix in range(0, x_size):         # handling image corners and edges
        if x_pix == 0:
            output_img[x_pix:] = 0
            continue
        elif x_pix == x_size-1:
            output_img[x_pix:] = 0
            break
        for y_pix in range(0, y_size):
            if y_pix == 0:
                output_img[x_pix, y_pix] = 0
                continue
            elif y_pix == y_size-1:
                output_img[x_pix, y_pix] = 0
                break
            else:
                output_img[x_pix, y_pix] = convolution_zoom_in(input_img[x_pix-1:x_pix+2, y_pix-1:y_pix+2], mask_matrix)
    return output_img


def produce_x_grad(input_img):
    x_diff = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    return convolution(input_img, x_diff)


def produce_y_grad(input_img):
    y_diff = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    return convolution(input_img, y_diff)


def produce_overall_grad(input_img):
    x_grad = produce_x_grad(input_img)
    print('finished x grad')
    y_grad = produce_y_grad(input_img)
    print('finished y grad')

    return (x_grad**2+y_grad**2)**0.5

def main():
    my_file = easygui.fileopenbox()
    my_img = io.imread(my_file, as_gray=True)

    x_size, y_size = my_img.shape
    size = x_size * y_size
    goal_size = 300**2      # chosen so computations run in ~1s
    scaling_factor = (goal_size / size)**(0.5)

    my_img = resize(my_img,
        (math.floor(x_size * scaling_factor), math.floor(y_size * scaling_factor)),
        anti_aliasing=True)

    x_grad = exposure.equalize_hist(produce_x_grad(my_img))
    y_grad = exposure.equalize_hist(produce_y_grad(my_img))
    overall_grad_img = exposure.equalize_hist(produce_overall_grad(my_img))
    print('finished overall_grad')

    output_img = np.empty(my_img.shape)

    for column in range(0, len(overall_grad_img)):
        for value in range(0, len(overall_grad_img[column])):
            output_img[column,value] = 1-overall_grad_img[column,value]

    final_output = np.concatenate([my_img, output_img], axis=1)
    viewer.ImageViewer(final_output).show()

if __name__ == '__main__':
    main()
