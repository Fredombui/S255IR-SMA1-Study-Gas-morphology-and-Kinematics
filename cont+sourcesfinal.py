from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, AsinhStretch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle, ConnectionPatch
import numpy as np
import matplotlib as mpl

# === Matplotlib setup ===
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

# ----------------------------
# User inputs
fits_file = "ALMA01712814_continuum.fits"     # continuum FITS file

# Original zoom region was [100, 400, 100, 400]
# → Make it half more zoomed (smaller window around same center)
center_y = (100 + 400) / 2
center_x = (100 + 400) / 2
height = (400 - 100) / 1.5   # shrink by factor 1.5
width = (400 - 100) / 1.5

zoom1 = [
    int(center_y - height / 2),
    int(center_y + height / 2),
    int(center_x - width / 2),
    int(center_x + width / 2)
]

zoom2 = [220, 280, 220, 280]  # close-up (unchanged)
cmap = "magma"               # colormap for continuum

# Source coordinates (RA in deg, Dec in deg)
sources = [
    ("SMA1", 93.2250403, 17.9897532),
    ("SMA3", 93.2243558, 17.9899063),
    ("SMA2", 93.2240617, 17.9905971),
]

# ----------------------------
# Load FITS and WCS safely
h = fits.open(fits_file)[0]
hdr = h.header
data = np.squeeze(h.data)                     # remove degenerate dimensions
if hdr.get('NAXIS', 0) > 2:
    wcs = WCS(hdr, naxis=2)
else:
    wcs = WCS(hdr)

# ----------------------------
# Figure setup: now only 2 panels
fig, axs = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'projection': wcs})
norm = ImageNormalize(data, stretch=AsinhStretch(), vmin = 0.001, vmax=0.045)

# ----------------------------
# Function to add colorbar matching panel height
def add_colorbar_same_height(im, ax, label):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.55)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.yaxis.set_label_position('left')
    cb.ax.yaxis.set_ticks_position('left')
    cb.set_label(label, rotation=270, labelpad=-5)
    cb.minorticks_off()
    cb.ax.grid(False)
    return cb

# ----------------------------
# Panel 1 – zoom-in region
im1 = axs[0].imshow(data, origin='lower', cmap=cmap, norm=norm)
axs[0].set_xlim(zoom1[2], zoom1[3])
axs[0].set_ylim(zoom1[0], zoom1[1])
#axs[0].set_title("1.3 mm S255IR Continuum")
axs[0].set_xlabel("Right Ascension")
axs[0].set_ylabel("Declination")

# === Add source labels ===
for name, ra, dec in sources:
    x_pix, y_pix = wcs.wcs_world2pix(ra, dec, 0)
    axs[0].plot(x_pix, y_pix, marker='o', color='white', markersize=4, mec='black', mew=0.5)
    axs[0].text(
        x_pix + 8, y_pix + 8, name, color='white', fontsize=7, weight='bold',
        ha='right', va='top', path_effects=[]
    )

# Draw rectangle showing zoomed region for Panel 2
rect2 = Rectangle(
    (zoom2[2], zoom2[0]),
    zoom2[3] - zoom2[2],
    zoom2[1] - zoom2[0],
    linewidth=1.8,
    edgecolor='white',
    facecolor='none'
)
axs[0].add_patch(rect2)

# Connect Panel 1 → Panel 2 (bottom and top edges)
# Bottom edge of zoom box
fig.add_artist(ConnectionPatch(
    xyA=(zoom2[2], zoom2[0]), coordsA=axs[0].transData,
    xyB=(0, 0), coordsB=axs[1].transAxes,
    color='white', lw=1.2, linestyle=':'
))

fig.add_artist(ConnectionPatch(
    xyA=(zoom2[2], zoom2[0]), coordsA=axs[0].transData,
    xyB=(0, 0), coordsB=axs[1].transAxes,
    color='black', lw=0.7, linestyle=':'
))

# Top edge of zoom box
fig.add_artist(ConnectionPatch(
    xyA=(zoom2[2], zoom2[1]), coordsA=axs[0].transData,
    xyB=(0, 1), coordsB=axs[1].transAxes,
    color='white', lw=1.2, linestyle=':'
))

fig.add_artist(ConnectionPatch(
    xyA=(zoom2[2], zoom2[1]), coordsA=axs[0].transData,
    xyB=(0, 1), coordsB=axs[1].transAxes,
    color='black', lw=0.7, linestyle=':'
))
# ----------------------------
# Panel 2 – close-up region
im2 = axs[1].imshow(data, origin='lower', cmap=cmap, norm=norm)
axs[1].set_xlim(zoom2[2], zoom2[3])
axs[1].set_ylim(zoom2[0], zoom2[1])
#axs[1].set_title("(SMA1 Core)")
axs[1].set_xlabel("Right Ascension")
axs[1].coords[1].set_ticklabel_visible(False)

# ----- Add white contours on Panel 2 -----
contour_levels = [0.005, 0.010, 0.020, 0.040]   # MODIFY as needed
axs[1].contour(
    data,
    levels=contour_levels,
    colors='white',
    linewidths=0.8,
    origin='lower'
)

# ✅ Only ONE colorbar on this final (close-up) panel
#add_colorbar_same_height(im2, axs[1], "Intensity (mJy/beam)")
cbar = plt.colorbar(im2, ax=axs, orientation='vertical')
cbar.set_label("Intensity (mJy/beam)")


# ----------------------------
plt.tight_layout(pad=2.0, w_pad=1.0)
plt.show()