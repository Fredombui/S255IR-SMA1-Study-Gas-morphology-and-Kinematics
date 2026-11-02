# ===============================================
# ALMA SiO Outflow + Continuum + Jet Plotting Script
# ===============================================
# Requires: spectral-cube, astropy, matplotlib, reproject

from spectral_cube import SpectralCube
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.patches import Ellipse
from reproject import reproject_interp
from astropy.coordinates import SkyCoord
import matplotlib as mpl

# === Matplotlib settings ===
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

# === 1) Load SiO cube ===
cube = SpectralCube.read('ALMA806_SiO_v0_line_imreframed.fits')
cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio',
                               rest_value=(217.10498e9 * u.Hz))

# === 2) RMS ===
rms = 0.00555  # Jy/beam
rms_mjy = rms * 1000

# === 3) Contour levels (blue/red lobes) ===
levels_blue_jyb = (np.array([30, 47, 54, 61]) * rms_mjy * u.mJy/u.beam).to(u.Jy/u.beam).value
levels_red_jyb  = (np.array([20.0, 28.3, 40.0, 56.6, 80.0, 113.1]) * rms_mjy * u.mJy/u.beam).to(u.Jy/u.beam).value

# === 4) Blue and red velocity ranges ===
blue_slab = cube.spectral_slab(-21.7*u.km/u.s, 2.58*u.km/u.s)
red_slab  = cube.spectral_slab(12.02*u.km/u.s, 29.55*u.km/u.s)

blue_mom0 = blue_slab.moment0()
red_mom0  = red_slab.moment0()

# === 5) Load continuum map ===
cont_hdu = fits.open('ALMA01712814_continuum.fits')[0]
cont_data = cont_hdu.data.squeeze()
cont_wcs = WCS(cont_hdu.header).celestial

# === 6) Prepare moment maps ===
blue_hdu = blue_mom0.hdu
red_hdu  = red_mom0.hdu
blue_img = blue_hdu.data.squeeze()
red_img  = red_hdu.data.squeeze()
blue_wcs = WCS(blue_hdu.header).celestial
red_wcs  = WCS(red_hdu.header).celestial

# === 7) Load radio jet map ===
jet_hdu = fits.open('G192.6005_C_2018.fits')[0]
jet_data = jet_hdu.data

# Handle dimensionality
if jet_data.ndim == 4:
    jet_data = jet_data[0, 0, :, :]
elif jet_data.ndim == 3:
    jet_data = jet_data[0, :, :]
else:
    jet_data = jet_data.squeeze()

jet_wcs = WCS(jet_hdu.header).celestial

# === 8) Reproject jet to continuum frame ===
jet_reproj, _ = reproject_interp((jet_data, jet_wcs), cont_wcs, cont_data.shape)

# Clean and mask
jet_reproj = np.nan_to_num(jet_reproj, nan=0.0)
jet_reproj[jet_reproj < 0] = 0

# === Jet contour levels ===
rms_jet = 6e-6  # 6 µJy/beam
jet_levels = np.array([3, 5, 7, 11, 20, 50, 150]) * rms_jet

# === 9) Plot ===
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection=cont_wcs)

# === Continuum background ===
vmax_cont = np.percentile(cont_data[cont_data > 0], 99.5)
im = ax.imshow(cont_data, origin='lower', cmap='gray_r', vmin=0, vmax=vmax_cont)

# === Colorbar ===
cbar = fig.colorbar(im, ax=ax, pad=0.05, shrink=0.8)
cbar.set_label('Continuum Flux Density (Jy/beam)', fontsize=12)

# === Blue and red contours ===
ax.contour(blue_img, transform=ax.get_transform(blue_wcs),
           levels=levels_blue_jyb, colors='blue', linewidths=1.0, alpha=0.8)
ax.contour(red_img, transform=ax.get_transform(red_wcs),
           levels=levels_red_jyb, colors='red', linewidths=1.0, alpha=0.8)

# === Radio jet contours (reprojected) ===
ax.contour(jet_reproj, transform=ax.get_transform(cont_wcs),
           levels=jet_levels, colors='lime', linewidths=1.2,
           linestyles='solid', alpha=0.9, zorder=10)

# === 10) SMA sources ===
sma_coords = SkyCoord([
    '06:12:53.774810 +17:59:26.149517',
    '06:12:53.845391 +17:59:23.662697'
], unit=(u.hourangle, u.deg), frame='icrs')

ax.plot(sma_coords.ra, sma_coords.dec, transform=ax.get_transform('icrs'),
        marker='*', color='red', markersize=5, linestyle='none', label='SMA SOURCES')

labels = ['SMA2', 'SMA3']
offsets = [(+0.00020, +0.00025), (-0.00025, -0.00020)]

for coord, label, (dra, ddec) in zip(sma_coords, labels, offsets):
    ax.text(coord.ra.deg + dra, coord.dec.deg + ddec, label,
            transform=ax.get_transform('icrs'),
            color='purple', fontsize=9, ha='left', va='bottom')

# === 11) Beam ellipse (anchored, auto-scaled) ===
bmaj = cont_hdu.header['BMAJ'] * 3600  # arcsec
bmin = cont_hdu.header['BMIN'] * 3600  # arcsec
bpa  = cont_hdu.header['BPA']

# Pixel scale (arcsec/pixel)
pixscale_x = abs(cont_wcs.wcs.cdelt[0]) * 3600
pixscale_y = abs(cont_wcs.wcs.cdelt[1]) * 3600

# Beam size in relative axes coordinates
beam_width_rel  = (bmin / pixscale_x) / cont_data.shape[1]
beam_height_rel = (bmaj / pixscale_y) / cont_data.shape[0]

# Fixed corner position
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

ax.text(
    beam_x, beam_y + 0.05,
    'Beam',
    transform=ax.transAxes,
    ha='center', va='bottom',
    fontsize=10, color='black',
    zorder=6
)

# === 12) Labels ===
ax.set_xlabel('Right Ascension (J2000)')
ax.set_ylabel('Declination (J2000)')

# === 12.1) Show PV cut line ===
pv_center = SkyCoord('06h12m53.99078485s +17d59m23.51105944s', frame='icrs')
pv_length = 6.0 * u.arcsec
pv_pa = -34.0 * u.deg  # position angle east of north

# Convert PA to radians
pa_rad = pv_pa.to(u.rad).value

# Half-length offsets
delta_ra = (pv_length/2 * np.sin(pa_rad)).to(u.deg)
delta_dec = (pv_length/2 * np.cos(pa_rad)).to(u.deg)

ra_start  = pv_center.ra - delta_ra
ra_end    = pv_center.ra + delta_ra
dec_start = pv_center.dec - delta_dec
dec_end   = pv_center.dec + delta_dec

# Plot PV cut line
ax.plot([ra_start.deg, ra_end.deg], [dec_start.deg, dec_end.deg],
        transform=ax.get_transform('icrs'),
        color='cyan', linestyle='solid', linewidth=2, label='PV Cut')

# Optional legend
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()


