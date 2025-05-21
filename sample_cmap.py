from color_theory import *


# Section - testing hsl linear color mapping
# # Example: from desaturated red to saturated blue
# rgb1 = (0.6, 0.3, 0.3)
# rgb2 = (0.4, 0.25, 0.7)
# rgb3 = (0.2, 0.2, 1.0)
# colors = interpolate_hls([rgb1, rgb2, rgb3])

# # Visualize
# gradient = np.linspace(0, 1, 256).reshape(1, -1)
# plt.imshow(gradient, aspect='auto', cmap=cmap)
# plt.axis('off')
# plt.show()


# Section - this is a colormapping I am trying to make that avoids red and green somewhat manually


rgb_yellow = hex2rgb('#FFFF00')
rgb_purple = hex2rgb('#8377D1')
rgb_cmap = palette_tetrad(rgb_yellow, angle=-35)
print(rgb_cmap)

# Put this in manually, TODO right sorting algorithm
rgb_cmap = [(0.0, 0.0, 1.0), (0.0, 0.5833333333333333, 1.0), (1.0, 0.41666666666666663, 0.0), (1., 1., 0.),]
rgb_cmap2 = [rgb_purple, (0.0, 0.0, 1.0), (0.0, 0.5833333333333333, 1.0), (1.0, 0.41666666666666663, 0.0), (1., 1., 0.),]
# plot_color_from_rgb(rgb_cmap)

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
hls_greys = palette_grey(np.arange(0.5, 0.8 + 0.1, 0.1))
print(hls_greys)

rgb_greys = [hls_to_rgb(*hls) for hls in hls_greys]
print(rgb_greys)

img = np.arange(100*100).reshape((100, 100))
img[:20, :20] = 0
cmap_grey = cgradient(rgb_greys)
cmap_grey.set_under('black')

ccmap = constrained_cmap(rgb_cmap, hls_greys)


rgb_cmap = [(0.0, 0.0, 1.0), (0.0, 0.5833333333333333, 1.0), (1.0, 0.41666666666666663, 0.0), (1., 1., 0.),]
# colors = interpolate_hls(rgb_cmap)

hls_greys2 = palette_grey(np.arange(0.4, 0.8 + 0.1, 0.1))
ccmap2 = constrained_cmap(rgb_cmap2, hls_greys2, hls_interpolation=True)


ccmap3 = constrained_cmap(rgb_cmap2, hls_greys2, hls_interpolation=False)

fig, ax = plt.subplots(3)
for axs in ax:
    axs.axis('off')
ax[0].imshow(img, cmap=ccmap, vmin=0.001)
ax[1].imshow(img, cmap=ccmap2, vmin=0.001)
ax[2].imshow(img, cmap=ccmap3, vmin=0.001)
plt.show()
plt.close()


