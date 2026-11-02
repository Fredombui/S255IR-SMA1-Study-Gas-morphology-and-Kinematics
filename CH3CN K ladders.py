from spectral_cube import SpectralCube
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt

# 1. Load the full CH3CN data cube
cube = SpectralCube.read('ALMA01712809.fits', use_dask=True)

# 2. Define the exact coordinate of your continuum peak
ra = '06h12m54.02s' 
dec = '+17d59m23.1s'
peak_coord = SkyCoord(ra, dec, frame='icrs')

# 3. Find the nearest pixel to those coordinates
# This replaces the non-existent .extract_spectrum()
x_pix, y_pix = cube.wcs.celestial.world_to_pixel(peak_coord)
x_pix, y_pix = int(round(float(x_pix))), int(round(float(y_pix)))

# 4. Extract the 1D spectrum by slicing the cube at that pixel
# Cube order is [spectral, y, x]
raw_spectrum = cube[:, y_pix, x_pix]

# 5. Now convert the 1D spectrum units (Option 2 efficiency)
spectrum = raw_spectrum.to(u.Jy/u.beam)

# 6. Convert the x-axis to Frequency (GHz)
spectrum_freq = spectrum.with_spectral_unit(
    u.GHz, 
    velocity_convention='radio', 
    rest_value=220.679 * u.GHz
)

# 7. Quick plot
plt.figure(figsize=(10, 6))
plt.plot(spectrum_freq.spectral_axis, spectrum_freq.value, drawstyle='steps-mid')
plt.xlabel(f'Frequency ({spectrum_freq.spectral_axis.unit})')
plt.ylabel(f'Flux Density ({spectrum.unit})')
plt.title(f'CH3CN J=12-11 K-ladder at SMA1 Peak\nPixel: [{x_pix}, {y_pix}]')
plt.xlim(220.5, 220.8) 
plt.grid(True, alpha=0.3)
plt.show()