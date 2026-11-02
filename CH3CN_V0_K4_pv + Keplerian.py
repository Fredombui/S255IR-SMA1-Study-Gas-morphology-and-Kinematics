import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from spectral_cube import SpectralCube
from astropy import units as u
from astropy.coordinates import SkyCoord
from pvextractor import Path, extract_pv_slice
from astropy.constants import G

plt.style.use('default') 
mpl.rcParams['font.size'] = 10

# Parameters
cube_file = "CH3CN_V0_K4_line_imreframed.fits"
restfreq = 220.68036 * u.GHz
v_sys = 7.4                   
distance = 1.78 * u.kpc
inclination = np.deg2rad(75.4)
target_mass = 20.0 * u.Msun  # Only 16 Msun remains

# PV Path Parameters
center = SkyCoord("06h12m54.009680s", "+17d59m23.111404s", frame="icrs")
PA = -34.0 * u.deg
length = 6.0 * u.arcsec
width = 0.5 * u.arcsec

# Load and Prepare Cube
cube = SpectralCube.read(cube_file)
cube = cube.with_spectral_unit(
    u.km / u.s, 
    velocity_convention="radio", 
    rest_value=restfreq
)
vel_axis = cube.spectral_axis.value


# PV Extraction
def extract_pv(cube, center, PA, length, width):
    half_len = (length / 2).to(u.deg)
    ddec = half_len * np.cos(PA)
    dra = half_len * np.sin(PA) / np.cos(center.dec)
    
    start_pt = SkyCoord(ra=center.ra - dra, dec=center.dec - ddec)
    end_pt = SkyCoord(ra=center.ra + dra, dec=center.dec + ddec)
    
    path = Path(SkyCoord([start_pt, end_pt]), width=width)
    pv = extract_pv_slice(cube, path)
    
    offsets = np.linspace(-length.value/2, length.value/2, pv.data.shape[1])
    return pv, offsets

pv_slice, offsets = extract_pv(cube, center, PA, length, width)

# Recenter Offset Axis
mask = np.abs(vel_axis - v_sys) <= 2.0
intensity_at_vsys = np.nansum(pv_slice.data[mask, :], axis=0)
centroid = np.sum(offsets * intensity_at_vsys) / np.sum(intensity_at_vsys)
offsets_centered = offsets - centroid


fig, ax = plt.subplots(figsize=(10, 6))

im = ax.imshow(
    pv_slice.data, origin="lower", aspect="auto", cmap="viridis",
    extent=[offsets_centered[0], offsets_centered[-1], vel_axis[0], vel_axis[-1]],
    vmin=0.01, vmax=0.2
)

# Reference Lines
ax.axhline(v_sys, color="black", ls=":", lw=1, alpha=0.5)
ax.axvline(0, color="black", ls=":", lw=1, alpha=0.5)

# ----------------------------------------------------------

r_arcsec = np.linspace(0.0, length.value / 2, 200) * u.arcsec
r_au = (r_arcsec.value * distance.to(u.pc).value) * u.AU

with np.errstate(divide='ignore'):
    v_k = (np.sqrt(G * target_mass / r_au).to(u.km/u.s)).value * np.sin(inclination)

# Plot Redshifted side (Red)
ax.plot(r_arcsec.value, v_sys + v_k, color='red', lw=2.5, label='16 $M_{\odot}$ Model')

# Plot Blueshifted side (Blue)
ax.plot(-r_arcsec.value, v_sys - v_k, color='blue', lw=2.5)

# ----------------------------------------------------------
# Formatting
# ----------------------------------------------------------
#ax.set_title(f"PV Diagram: PA={PA.value}°, $M_*$={target_mass.value} $M_{{\odot}}$")
ax.set_xlabel("Offset [arcsec]")
ax.set_ylabel("Velocity [km/s]")

ax.set_xlim(offsets_centered[0], offsets_centered[-1])


ax.set_ylim(vel_axis[0], vel_axis[-1]) 

cb = plt.colorbar(im, pad=0.01)
cb.set_label("Intensity [Jy/beam]")
#ax.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.show()