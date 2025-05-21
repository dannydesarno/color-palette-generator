from rgb2hls import *
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


def cgradient(colors, n_steps=255*4, cmap_name='gcmap'):
    """
    Color map gradient that connects each color in the list
    :param colors: list or iterable of R G B values from 0 to 1. Example, [[0, 0, 1], [1, 1, 0], ...]
    :return:
    """
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, n_steps)
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


def constrained_cmap(rgbs, hls_greys):
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

    rgbs_map = [hls_to_rgb(*hls) for hls in hls_map]

    ccmap = cgradient(rgbs_map)
    ccmap.set_under('black')
    return ccmap





# rgb = [124, 124, 255]
rgb = [24, 124, 124]
rgb = normalize_rgb(*rgb)
# rgb2 = palette_complimentary(rgb)
rgbs = palette_tetrad(rgb)
rgbs1 = palette_tetrad(rgb, angle=35)
rgbs2 = palette_tetrad(rgb, angle=45)
rgbs3 = palette_tetrad(rgb, angle=70)


# plot_color_from_rgb([rgb, rgb2])
# plot_color_from_rgb(rgbs)
# plot_color_from_rgb(rgbs1)
# plot_color_from_rgb(rgbs2)
# plot_color_from_rgb(rgbs3)

# hls_greys = palette_grey(np.arange(0.1, 0.9 + 0.15, 0.15))
hls_greys = palette_grey(np.arange(0.2, 0.8 + 0.2, 0.2))
print(hls_greys)

rgb_greys = [hls_to_rgb(*hls) for hls in hls_greys]
print(rgb_greys)

img = np.arange(100*100).reshape((100, 100))
img[:20, :20] = 0
cmap_grey = cgradient(rgb_greys)
cmap_grey.set_under('black')

ccmap = constrained_cmap(rgbs, hls_greys)
plt.imshow(img, cmap=ccmap, vmin=0.001)
plt.show()
plt.close()