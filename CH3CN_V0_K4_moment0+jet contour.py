# ============================================================
# ALMA Outflow + Continuum + Jet Plotting Script (with PV Cut Line)
# ============================================================
# Requires: spectral-cube, astropy, matplotlib, reproject

from spectral_cube import SpectralCube
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.patches import Ellipse
from astropy.coordinates import SkyCoord
import matplotlib as mpl

# === Matplotlib settings ===
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'

# === 1) Load CO cube ===
cube = SpectralCube.read('CH3CN_V0_K4_line_imreframed.fits')
cube = cube.with_spectral_unit(u.km/u.s, velocity_convention='radio',
                               rest_value=(220.68036e9 * u.Hz))

# === 2) Estimate RMS from line-free channels ===
rms = 0.00794  # Jy/beam
rms_mjy = rms * 1000  # mJy/beam

# === 3) Define contour levels (blue/red outflow) ===
levels_blue_jyb = (np.array([15.0, 23.8, 32.5, 41.3, 50]) * rms_mjy * u.mJy/u.beam).to(u.Jy/u.beam).value
levels_red_jyb  = (np.array([15.0, 23.8, 32.5, 41.3, 50]) * rms_mjy * u.mJy/u.beam).to(u.Jy/u.beam).value

# === 4) Create blue/red moment-0 maps ===
blue_slab = cube.spectral_slab(-5.00*u.km/u.s, 5.49*u.km/u.s)
red_slab  = cube.spectral_slab(9.0*u.km/u.s, 20.0*u.km/u.s)
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

# === 7) Load radio jet FITS ===
jet_hdu = fits.open('G192.6005_C_2018.fits')[0]
jet_data = np.squeeze(jet_hdu.data)
jet_wcs = WCS(jet_hdu.header).celestial

# === Jet contour levels ===
rms_jet = 7e-6  # 6 µJy/beam
jet_levels = np.array([3, 5, 7, 11, 20, 50, 150]) * rms_jet

# === 8) Plot ===
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection=cont_wcs)

# Continuum background
vmax_cont = np.percentile(cont_data[cont_data > 0], 99.5)
im = ax.imshow(cont_data, origin='lower', cmap='gray_r', vmin=0, vmax=vmax_cont)

# Color bar
cbar = fig.colorbar(im, ax=ax, pad=0.05, shrink=0.8)
cbar.set_label('Continuum Flux Density (Jy/beam)', fontsize=12)

# Blue and red contours
ax.contour(blue_img, transform=ax.get_transform(blue_wcs),
           levels=levels_blue_jyb, linewidths=1.0,
           linestyles='solid', colors='blue', alpha=0.8)
ax.contour(red_img, transform=ax.get_transform(red_wcs),
           levels=levels_red_jyb, linewidths=1.0,
           linestyles='solid', colors='red', alpha=0.8)

# Radio jet contours
ax.contour(jet_data, transform=ax.get_transform(jet_wcs),
           levels=jet_levels, colors='lime', linewidths=1.2,
           linestyles='solid', alpha=0.9, zorder=10)

# === 9) Add SMA sources ===
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
            color='purple', fontsize=9,
            ha='left', va='bottom')

# === 10) Beam ellipse (anchored, auto-scaled) ===
bmaj = cont_hdu.header['BMAJ'] * 3600  # arcsec
bmin = cont_hdu.header['BMIN'] * 3600  # arcsec
bpa  = cont_hdu.header['BPA']

pixscale_x = abs(cont_wcs.wcs.cdelt[0]) * 3600
pixscale_y = abs(cont_wcs.wcs.cdelt[1]) * 3600

beam_width_rel  = (bmin / pixscale_x) / cont_data.shape[1]
beam_height_rel = (bmaj / pixscale_y) / cont_data.shape[0]

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

# === 11) PV cut line ===
pv_center_SiO = SkyCoord('06h12m54.00968s +17d59m23.111404s', frame='icrs')
pv_length_SiO = 6.0 * u.arcsec
pv_pa_SiO = -34.0 * u.deg

# Convert PA to radians
pa_rad_SiO = pv_pa_SiO.to(u.rad).value

# Half-length offsets
delta_ra_SiO = (pv_length_SiO/2 * np.sin(pa_rad_SiO)).to(u.deg)
delta_dec_SiO = (pv_length_SiO/2 * np.cos(pa_rad_SiO)).to(u.deg)

ra_start_SiO  = pv_center_SiO.ra - delta_ra_SiO
ra_end_SiO    = pv_center_SiO.ra + delta_ra_SiO
dec_start_SiO = pv_center_SiO.dec - delta_dec_SiO
dec_end_SiO   = pv_center_SiO.dec + delta_dec_SiO

ax.plot([ra_start_SiO.deg, ra_end_SiO.deg],
        [dec_start_SiO.deg, dec_end_SiO.deg],
        transform=ax.get_transform('icrs'),
        color='cyan', linestyle='solid', linewidth=2, label='13CO PV Cut')

# === 12) Axes labels ===
ax.set_xlabel('Right Ascension (J2000)')
ax.set_ylabel('Declination (J2000)')

# Add legend for PV cut
ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.show()
