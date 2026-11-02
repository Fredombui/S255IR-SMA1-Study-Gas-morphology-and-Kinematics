# ============================================================
# ALMA Moment Maps + Radio Jet Contours + Beam Indicator (Fixed)
# ============================================================

# Dependencies
# pip install spectral-cube astropy matplotlib reproject (if needed)

from spectral_cube import SpectralCube
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.patches import Ellipse
from astropy.coordinates import SkyCoord
import matplotlib as mpl

# === Matplotlib setup ===
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

# === 1) Load Spectral Cube ===
cube = SpectralCube.read('13CO_v0_line1_imreframed2.fits')
cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio',
                               rest_value=(220.398e9 * u.Hz))

# === 2) Estimate RMS from line-free channels ===
rms = 0.008393  # Jy/beam
rms_mjy = rms * 1000  # mJy/beam

# Define contour levels (for blue and red lobes)
levels_blue_jyb = (np.array([15, 20, 30, 50, 80, 120]) * rms_mjy * u.mJy/u.beam).to(u.Jy/u.beam).value
levels_red_jyb  = (np.array([20, 30, 50, 80, 120]) * rms_mjy * u.mJy/u.beam).to(u.Jy/u.beam).value

# === 3) Extract blue/red spectral slabs and moment 0 maps ===
blue_slab = cube.spectral_slab(-20.25*u.km/u.s, -0.3675*u.km/u.s)
red_slab  = cube.spectral_slab(11.58*u.km/u.s, 34.17*u.km/u.s)
blue_mom0 = blue_slab.moment0()
red_mom0  = red_slab.moment0()

# === 4) Load continuum map ===
cont_hdu = fits.open('ALMA01712814_continuum.fits')[0]
cont_data = cont_hdu.data.squeeze()
cont_wcs = WCS(cont_hdu.header).celestial  # RA/Dec only

# === 5) Prepare moment maps ===
blue_hdu = blue_mom0.hdu
red_hdu  = red_mom0.hdu
blue_img = blue_hdu.data.squeeze()
red_img  = red_hdu.data.squeeze()
blue_wcs = WCS(blue_hdu.header).celestial
red_wcs  = WCS(red_hdu.header).celestial

# === 6) Load radio jet contour map ===
jet_hdu = fits.open('G192.6005_C_2018.fits')[0]
jet_data = jet_hdu.data

# Handle dimensionality (reduce to 2D)
if jet_data.ndim == 4:
    jet_data = jet_data[0, 0, :, :]
elif jet_data.ndim == 3:
    jet_data = jet_data[0, :, :]
else:
    jet_data = jet_data.squeeze()

jet_wcs = WCS(jet_hdu.header).celestial

# Define contour levels for radio jet (scale appropriately)
rms_jet = 6e-6  # 6 µJy/beam
jet_levels = np.array([3, 5, 7, 11, 20, 50, 150]) * rms_jet

# === 7) Plot ===
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection=cont_wcs)

vmax_cont = np.percentile(cont_data[cont_data > 0], 99.5)
im = ax.imshow(cont_data, origin='lower', cmap='gray_r', vmin=0, vmax=vmax_cont)

cbar = fig.colorbar(im, ax=ax, pad=0.05, shrink=0.8)
cbar.set_label('Continuum Flux Density (Jy/beam)', fontsize=12)

# === Blue and red outflow lobes ===
ax.contour(blue_img, transform=ax.get_transform(blue_wcs),
           levels=levels_blue_jyb, linewidths=1.0, linestyles='solid', colors='blue', alpha=0.8)
ax.contour(red_img, transform=ax.get_transform(red_wcs),
           levels=levels_red_jyb, linewidths=1.0, linestyles='solid', colors='red', alpha=0.8)

# === Overlay Radio Jet Contours ===
ax.contour(jet_data, transform=ax.get_transform(jet_wcs),
           levels=jet_levels, colors='lime', linewidths=1.2, linestyles='solid', alpha=0.8)

# === 8) Add SMA sources ===
sma_coords = SkyCoord([
    '06:12:53.774810 +17:59:26.149517',
    '06:12:53.845391 +17:59:23.662697'
], unit=(u.hourangle, u.deg), frame='icrs')

ax.plot(sma_coords.ra, sma_coords.dec, transform=ax.get_transform('icrs'),
        marker='*', color='red', markersize=5, linestyle='none', label='SMA Sources')

# Label SMA sources
labels = ['SMA2', 'SMA3']
offsets = [(+0.00020, +0.00025), (-0.00025, -0.00020)]

for coord, label, (dra, ddec) in zip(sma_coords, labels, offsets):
    ax.text(coord.ra.deg + dra, coord.dec.deg + ddec, label,
            transform=ax.get_transform('icrs'),
            color='purple', fontsize=9,
            ha='left', va='bottom')

# === 9) Beam ellipse and label (auto-scaled, fixed to corner) ===
bmaj = cont_hdu.header['BMAJ'] * 3600  # [arcsec]
bmin = cont_hdu.header['BMIN'] * 3600  # [arcsec]
bpa  = cont_hdu.header['BPA']          # [deg]

# Pixel scale in arcsec/pixel
pixscale_x = abs(cont_wcs.wcs.cdelt[0]) * 3600
pixscale_y = abs(cont_wcs.wcs.cdelt[1]) * 3600

# Convert to relative axes fraction
beam_width_rel  = (bmin / pixscale_x) / cont_data.shape[1]
beam_height_rel = (bmaj / pixscale_y) / cont_data.shape[0]

# Beam position in axes coordinates
beam_x, beam_y = 0.12, 0.08

beam_ellipse = Ellipse(
    (beam_x, beam_y),
    width=beam_width_rel,
    height=beam_height_rel,
    angle=bpa,
    transform=ax.transAxes,
    facecolor='gray',
    edgecolor='black',
    alpha=0.8,
    zorder=5
)
ax.add_patch(beam_ellipse)

# Add "Beam" label
ax.text(
    beam_x, beam_y + 0.05, 'Beam',
    transform=ax.transAxes,
    ha='center', va='bottom',
    fontsize=10, color='black',
    zorder=6
)

# === 10) Axes labels ===
ax.set_xlabel('Right Ascension (J2000)')
ax.set_ylabel('Declination (J2000)')

# === 11) Add 13CO PV cut line ===
pv_center_13co = SkyCoord('06h12m54.00968s +17d59m23.111404s', frame='icrs')
pv_length_13co = 6.0 * u.arcsec
pv_pa_13co = -34.0 * u.deg

# Convert PA to radians
pa_rad_13co = pv_pa_13co.to(u.rad).value

# Half-length offsets
delta_ra_13co = (pv_length_13co/2 * np.sin(pa_rad_13co)).to(u.deg)
delta_dec_13co = (pv_length_13co/2 * np.cos(pa_rad_13co)).to(u.deg)

ra_start_13co  = pv_center_13co.ra - delta_ra_13co
ra_end_13co    = pv_center_13co.ra + delta_ra_13co
dec_start_13co = pv_center_13co.dec - delta_dec_13co
dec_end_13co   = pv_center_13co.dec + delta_dec_13co

ax.plot([ra_start_13co.deg, ra_end_13co.deg],
        [dec_start_13co.deg, dec_end_13co.deg],
        transform=ax.get_transform('icrs'),
        color='cyan', linestyle='solid', linewidth=2, label='PV Cut')

# Optional legend
ax.legend(loc='upper right', fontsize=10)


plt.tight_layout()
plt.show()
