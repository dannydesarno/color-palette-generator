"""
ddesarn@uwo.ca this file is having tools to go from RGB to HSL and vice versa
"""
import numpy as np
import matplotlib.pyplot as plt
from colorsys import rgb_to_hls, hls_to_rgb


def normalize_rgb(r, g, b):
    return r / 255, g / 255, b / 255


def hex2rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    rgb = [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]
    return np.array(normalize_rgb(*rgb))


# # RGB to HSL (Note: RGB values must be in [0, 1])
# h, l, s = rgb_to_hls(r, g, b)
#
# # HSL to RGB (HSL are all from [0, 1])
# r, g, b = hls_to_rgb(h, l, s)