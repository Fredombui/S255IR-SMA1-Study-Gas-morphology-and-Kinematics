#!/usr/bin/env python3
"""
Plot radio jet contours from a FITS image or cube.
Automatically handles 2D, 3D, or 4D data (Stokes/frequency cubes).
"""

from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ImageNormalize, AsinhStretch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ------------------------------------------------------------
# File name
fits_file = "G192.6005_C_2018.fits"
# ------------------------------------------------------------

# ---- Load FITS data ----------------------------------------------------------
hdu = fits.open(fits_file)[0]
data = np.squeeze(hdu.data)
header = hdu.header
print(f"Original data shape: {data.shape}")

# ---- Prepare WCS and extract 2D slice ---------------------------------------
wcs_full = WCS(header)
ndim = wcs_full.pixel_n_dim

if ndim > 2:
    # Keep the last two axes (RA, Dec)
    spatial_axes = (ndim - 2, ndim - 1)
    # Build a list of slice indices: 0 for extra axes, full slices for RA/Dec
    slices = [0] * (ndim - 2) + [slice(None), slice(None)]
    # Apply slices only if data actually has those dimensions
    if data.ndim > 2:
        data_2d = np.squeeze(data[tuple(slices)])
    else:
        data_2d = data
    # Get the correct 2D WCS projection
    wcs_2d = wcs_full.slice(spatial_axes)
else:
    # Already 2D image
    data_2d = data
    wcs_2d = wcs_full

# ---- Estimate RMS and contour levels ----------------------------------------
rms = np.nanstd(data_2d[data_2d < np.percentile(data_2d, 80)])
levels = np.array([3, 5, 7, 10, 15]) * rms
print(f"Estimated RMS ≈ {rms:.3e}")

# ---- Plot -------------------------------------------------------------------
fig = plt.figure(figsize=(7, 6))
ax = plt.subplot(projection=wcs_2d)

norm = ImageNormalize(data_2d, stretch=AsinhStretch(),
                      vmin=-2*rms, vmax=10*rms)
ax.imshow(data_2d, origin="lower", cmap="gray", norm=norm)

ax.contour(
    data_2d,
    transform=ax.get_transform(wcs_2d),
    levels=levels,
    colors="lime",
    linewidths=1.0,
    alpha=0.9,
)

# ---- Add beam if available ---------------------------------------------------
if all(k in header for k in ("BMAJ", "BMIN", "BPA")):
    bmaj = header["BMAJ"] * 3600  # deg → arcsec
    bmin = header["BMIN"] * 3600
    bpa = header["BPA"]
    beam = Ellipse(
        (0.1, 0.1),
        width=bmin,
        height=bmaj,
        angle=bpa,
        transform=ax.transAxes,
        facecolor="none",
        edgecolor="black",
        lw=1.2,
    )
    ax.add_patch(beam)

# ---- Labels and display ------------------------------------------------------
ax.set_xlabel("Right Ascension (J2000)")
ax.set_ylabel("Declination (J2000)")
ax.set_title("G192.6005 — Radio Jet Contours")
ax.grid(False)

plt.tight_layout()
plt.show()
