import math

from PIL import Image
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np


def clamp(value, minimum, maximum):
    return max(min(value, maximum), minimum)


def warp(image, D):
    result = np.zeros(image.shape, dtype=np.uint8)

    for h in range(image.shape[0]):
        for w in range(image.shape[1]):

            x_pixel = clamp((w - D[h, w]), 0, image.shape[1] - 1)

            result[h, w] = image[h, int(x_pixel)]
    return result


# image = imread(r"D:\Programowanie\DL\Inzynierka\DenseNet.jpg")
# D = np.random.randint(10, 50, image.shape[:2], dtype=np.uint8)
# im_out = warp(image, D)
# plt.imshow(im_out, interpolation='nearest')
# plt.show()
# im = Image.fromarray(im_out)
# im.save("out.jpg")
