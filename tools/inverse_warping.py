import math

import torch
from PIL import Image
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np


def clamp(value, minimum, maximum):
    return max(min(value, maximum), minimum)


def warp(image, D):
    result = torch.zeros(image.shape, requires_grad=True).to('cuda')

    for b in range(D.shape[0]):
        for h in range(D.shape[-2]):
            for w in range(D.shape[-1]):

                x_pixel = clamp((w - D[b, :, h, w]), 0, image.shape[-1] - 1)

                result[b, :, h, w] = image[b, :, h, int(x_pixel)]
    return result


# image = imread(r"D:\Programowanie\DL\Inzynierka\DenseNet.jpg")
image = torch.rand((4,3,5,5))
D = torch.rand((4,1,3,3))
out = torch.nn.functional.interpolate(D,size=image.shape[2:4], mode='bilinear')
print(out.shape)
# im_out = warp(image, D)
# print(im_out)
# plt.imshow(im_out, interpolation='nearest')
# plt.show()
# im = Image.fromarray(im_out)
# im.save("out.jpg")
