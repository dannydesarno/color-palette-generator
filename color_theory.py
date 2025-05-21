import numpy as np

from rgb2hls import *
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def cgradient(colors, n_steps=255*4, cmap_name='gcmap'):
    """
    Color map gradient that connects each color in the list
    :param colors: list or iterable of R G B values from 0 to 1. Example, [[0, 0, 1], [1, 1, 0], ...]
    :return:
    """
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, n_steps)
    return cmap


def interpolate_hls(colors, n=256, is_hls=True):
    """
    Allows us to interpolate from color to color
    :param n: number of different colors in total mapping (if we have 3 colors, n // 2 are assigned for each)
    """
    if not is_hls:
        colors_hls = [rgb_to_hls(*color) for color in colors]
    else:
        colors_hls = colors
    color_pairs = [[colors_hls[ii], colors_hls[ii+1]] for ii in range(len(colors_hls) - 1)]
    m = int(n // len(color_pairs))

    cmap = []
    for c1, c2 in color_pairs:
        interpolated_colors = []
        for t in np.linspace(0, 1, m, endpoint=False):
            h = (1 - t) * c1[0] + t * c2[0]
            l = (1 - t) * c1[1] + t * c2[1]
            s = (1 - t) * c1[2] + t * c2[2]
            interpolated_colors.append(hls_to_rgb(h, l, s))
        cmap += interpolated_colors
    return cmap


def plot_color_from_rgb(rgb_colors):
  """
  Plots a single color patch from RGB values.

  Args:
    rgb_color: A tuple of three floats (R, G, B), each in the range [0, 1].
  """
  fig, axs = plt.subplots(len(rgb_colors))
  axs = axs.ravel()
  for ii, ax in enumerate(axs):
      ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor=rgb_colors[ii]))
      ax.set_xlim(0, 1)
      ax.set_ylim(0, 1)
      ax.set_xticks([])
      ax.set_yticks([])
      ax.set_title(f'R:{rgb_colors[ii][0]:.3f}, G:{rgb_colors[ii][1]:.3f}, B:{rgb_colors[ii][2]:.3f}')
  plt.show()
  plt.close()


def palette_complimentary(rgb: list|tuple|np.ndarray, is_hsl=False, return_rgb=True):
    """
    Given as input the
    :param rgb:
    :param is_hsl:
    :return:
    """
    if not isinstance(rgb, np.ndarray):
        rgb = np.array(rgb)
    if is_hsl:
        h, s, l = rgb[0], rgb[1], rgb[2]
    else:
        if np.sum(rgb > 1) > 1:
            rgb = normalize_rgb(*rgb)
        h, l, s = rgb_to_hls(*rgb)

    h2 = (h + 180/360) % 1
    if return_rgb:
        return hls_to_rgb(h2, l, s)
    else:
        return h2, l, s


def palette_tetrad(rgb: list|tuple|np.ndarray, angle=31.41592, is_hsl=False, return_rgb=True):
    """

    :param rgb:
    :param angle: Angle of hue is in degrees that we are choosing our tetrad color scheme
    :param is_hsl:
    :param return_rgb:
    :return:
    """
    if not isinstance(rgb, np.ndarray):
        rgb = np.array(rgb)
    if not is_hsl:
        if np.sum(rgb > 1) > 1:
            rgb = normalize_rgb(*rgb)
    hsl = np.array(rgb_to_hls(*rgb))
    hsl_180 =  np.array(palette_complimentary(rgb, return_rgb=False))

    hsl_2 = hsl.copy()
    hsl_2[0] = (hsl_2[0] + angle / 360) % 1

    hsl_4 = hsl_180.copy()
    hsl_4[0] = (hsl_4[0] + angle / 360) % 1

    if return_rgb:
        rgb_2 = hls_to_rgb(*hsl_2)
        rgb_180 = hls_to_rgb(*hsl_180)
        rgb_4 = hls_to_rgb(*hsl_4)
        return [rgb, rgb_2, rgb_180, rgb_4]
    else:
        return [hsl, hsl_2, hsl_180, hsl_4]


def palette_grey(grey_fractions):
    greys = np.zeros((len(grey_fractions), 3))
    greys[:, 1] = grey_fractions
    return greys


def constrained_cmap(rgbs, hls_greys, hls_interpolation=False, n_steps=256):
    """
    Given a set of RGBs (likely a color palette), and a HSL grey palette that the set of RGBs will be constrained by.
    The goal is to have a palette of colors that allows for multiple color gradients, but are also colorblind friendly
    :param rgbs:
    :param hsl_greys:
    :return:
    """
    assert len(rgbs) == len(hls_greys)

    hls_map = np.array([rgb_to_hls(*rgb) for rgb in rgbs])
    hls_map[:, 1] = hls_greys[:, 1]

    if not hls_interpolation:
        rgbs_map = [hls_to_rgb(*hls) for hls in hls_map]
        ccmap = cgradient(rgbs_map)
    else:
        ccmap = interpolate_hls(hls_map)
        ccmap = ListedColormap(ccmap)

    # Section - if we do rgb_interpolation (which is nicer in many cases) the grey values may not be preserved: force again
    sampled_cs = ccmap(np.linspace(0, 1, n_steps, endpoint=False))[:, :-1]  # (n_steps, 3), exlude A in RGBA
    s_hls = np.array([rgb_to_hls(*rgb) for rgb in sampled_cs])

    greys = np.linspace(hls_greys[0, 1], hls_greys[-1, 1], n_steps, endpoint=True)
    s_hls[:, 1] = greys

    ccmap = np.array([hls_to_rgb(*hls) for hls in s_hls])
    ccmap = np.clip(ccmap, 0, 1, out=ccmap)  # if by floating point errors we get numbers <0 or >1, remove
    ccmap = ListedColormap(ccmap)
    ccmap.set_under('black')
    return ccmap




# Section - testing
# # rgb = [124, 124, 255]
# rgb = [24, 124, 124]
# rgb = normalize_rgb(*rgb)
# # rgb2 = palette_complimentary(rgb)
# rgbs = palette_tetrad(rgb)
# rgbs1 = palette_tetrad(rgb, angle=35)
# rgbs2 = palette_tetrad(rgb, angle=45)
# rgbs3 = palette_tetrad(rgb, angle=70)
#
#
# # plot_color_from_rgb([rgb, rgb2])
# # plot_color_from_rgb(rgbs)
# # plot_color_from_rgb(rgbs1)
# # plot_color_from_rgb(rgbs2)
# # plot_color_from_rgb(rgbs3)
#
# # hls_greys = palette_grey(np.arange(0.1, 0.9 + 0.15, 0.15))
# hls_greys = palette_grey(np.arange(0.2, 0.8 + 0.2, 0.2))
# print(hls_greys)
#
# rgb_greys = [hls_to_rgb(*hls) for hls in hls_greys]
# print(rgb_greys)
#
# img = np.arange(100*100).reshape((100, 100))
# img[:20, :20] = 0
# cmap_grey = cgradient(rgb_greys)
# cmap_grey.set_under('black')
#
# ccmap = constrained_cmap(rgbs, hls_greys)
# plt.imshow(img, cmap=ccmap, vmin=0.001)
# plt.show()
# plt.close()