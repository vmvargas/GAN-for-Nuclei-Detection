import matplotlib.pyplot as plt

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

image = data.astronaut()

image_rescaled = rescale(image, 1.0 / 4.0)
image_resized = resize(image, (510, 510))

fig, axes = plt.subplots(nrows=2, ncols=1)

ax = axes.ravel()

ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original image")

ax[1].imshow(image_rescaled)
ax[1].set_title("Rescaled image")


plt.tight_layout()
plt.show()
