import numpy as np
import matplotlib.pyplot as plt
from spectral_cube import SpectralCube
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import curve_fit

# 1. DATA LOADING
# ------------------------------------------------------------------
cube = SpectralCube.read('ALMA01712809.fits', use_dask=True)
ra, dec = '06h12m54.02s', '+17d59m23.1s'
peak_coord = SkyCoord(ra, dec, frame='icrs')

# Extract 1D spectrum
x_pix, y_pix = cube.wcs.celestial.world_to_pixel(peak_coord)
raw_spectrum = cube[:, int(round(float(y_pix))), int(round(float(x_pix)))]

# Convert units and spectral axis
spectrum = raw_spectrum.to(u.Jy/u.beam)
spec_freq = spectrum.with_spectral_unit(u.GHz, velocity_convention='radio', 
                                        rest_value=220.679*u.GHz)

freq_axis = spec_freq.spectral_axis.value
data_values = spec_freq.value

# 2. LTE MODEL DEFINITION
# ------------------------------------------------------------------
def ch3cn_fit_engine(f, tex, width, v_lsr, amplitude):
    """LTE Model for CH3CN J=12-11 K-ladder."""
    # Rest frequencies for K=0 through K=6
    f_rest = np.array([220.747, 220.743, 220.730, 220.709, 220.679, 220.641, 220.594])
    weights = np.array([1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0]) # Ortho/Para weights
    
    model = np.zeros_like(f)
    for i, f0 in enumerate(f_rest):
        f_shift = f0 * (1 - v_lsr / 299792.458)
        sig = (width / 2.355) * (f_shift / 299792.458)
        
        # Boltzmann factor approximation for K-ladder levels
        energy_factor = np.exp(-(i**2 * 5.0) / tex)
        model += weights[i] * energy_factor * np.exp(-0.5 * ((f - f_shift) / sig)**2)
    
    return model * amplitude

# 3. TEMPERATURE ESTIMATION
# ------------------------------------------------------------------
# Initial guesses: [Tex, Width, V_lsr, Amp]
p0 = [150, 6.3, 6.7, 0.5] 

# Broadened bounds to accommodate resolved blue/redshifted components
# V_lsr range covers -5.0 to +20.0 km/s observed emission range
lower_bounds = [50, 1.0, -5.0, 0.01] 
upper_bounds = [500, 15.0, 20.0, 1.5]

popt, _ = curve_fit(ch3cn_fit_engine, freq_axis, data_values, p0=p0, bounds=(lower_bounds, upper_bounds))

# 4. RESULTS
print(f"--- S255IR-SMA1 Refined Analysis ---")
print(f"Excitation Temperature: {popt[0]:.1f} K")
print(f"Linewidth (FWHM): {popt[1]:.2f} km/s")
print(f"Velocity (V_lsr): {popt[2]:.2f} km/s")