import numpy as np
import cv2
import math
import skimage.util as skp
from tqdm import tqdm


# Noise reduction using bilateral filtering
#
# Explain the theory behind the method.
# Make some illustrative experiments with your own images. Remember to add noise first.
# Try different types of noise.
# Compare with at least one other method (simple filtering).


def bilateral_filter(IMAGE, kernel_size, SIGMA_DISTANCE, SIGMA_RANGE):
    """
    Bilateral filters an Image
    :param IMAGE: The source Image
    :param kernel_size: Size of the kernel window

    :param SIGMA_RANGE: Sigma of the color parameter
    :param SIGMA_DISTANCE: Sigma of the Distance Parameter
    :return: A Bilateral Filtered Image
    """
    d = kernel_size // 2
    img_filtered = np.zeros(IMAGE.shape, dtype=np.uint8)
    IMAGE = np.pad(IMAGE, d, 'mean')


    def filter(i, j, d):
        def calculate_weight(i, j, k, l):
            t_distance = - (np.square(i-k) + np.square(j-l))\
                         / 2 / np.square(SIGMA_DISTANCE)
            t_range = - np.square(IMAGE[i, j] - IMAGE[k, l])\
                      / 2 / np.square(SIGMA_RANGE)
            # print(t_distance, t_range, np.exp(t_distance + t_range))
            return np.exp(t_distance + t_range)

        sum_w = 0
        sum_iw = 0
        for k in range(i - d, i + d + 1):
            for l in range(j - d, j + d + 1):
                w = calculate_weight(i, j, k, l)
                sum_iw += IMAGE[k, l] * w
                sum_w += w
        return int(sum_iw / sum_w)

    for i in tqdm(range(d, IMAGE.shape[0] - d)):
        for j in range(d, IMAGE.shape[1] - d):
            new_val = filter(i, j, d)
            img_filtered[i - d, j - d] = new_val
    return img_filtered


src = cv2.imread('images/qcirchirp.bmp', 0)
cv2.imshow('source image', src)

image_with_noise = skp.random_noise(src, mode='s&p', amount=0.1, clip=True)
image_with_noise = (image_with_noise * 255).astype(np.uint8)
cv2.imshow('Noisy image', image_with_noise)

filtered_image = bilateral_filter(image_with_noise, 5, 70, 70)
cv2.imshow('filtered image', filtered_image)

filtered_image_cv = cv2.bilateralFilter(image_with_noise, 5, 70, 70)
cv2.imshow('filtered image cv', filtered_image_cv)

cv2.waitKey(0)
cv2.destroyAllWindows()
