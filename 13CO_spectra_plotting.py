import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
import astropy.units as u
from astropy.constants import c
from scipy.optimize import curve_fit

# === Style Settings ===
mpl.rcParams.update({
    "font.family": "serif",
    "xtick.direction": 'in',
    "ytick.direction": 'in',
    "xtick.top": True,
    "ytick.right": True,
    "savefig.dpi": 300
})

def gaussian(x, amp, cen, sigma):
    return amp * np.exp(-(x - cen)**2 / (2 * sigma**2))

# === Load and Process Data ===
spec_file = "13CO_spectra.fits"
hdu = fits.open(spec_file)[0]
spectrum = hdu.data.squeeze()
header = hdu.header

# Velocity Axis Calculation
naxis, crval, cdelt, crpix = header['NAXIS1'], header['CRVAL1'], header['CDELT1'], header['CRPIX1']
restfreq = header['RESTFRQ'] * u.Hz
freq_axis = (crval + (np.arange(naxis) - (crpix - 1)) * cdelt) * u.Hz
vel_axis = ((restfreq - freq_axis) / restfreq * c).to(u.km/u.s).value

# === Perform Gaussian Fit ===
# Guess: [Peak, Center, Width]
popt, _ = curve_fit(gaussian, vel_axis, spectrum, p0=[np.max(spectrum), 7.4, 1.0])
amp_fit, cen_fit, sigma_fit = popt
fwhm = 2.355 * sigma_fit

# Create a "super-smooth" axis for the Gaussian curve (1000 points)
vel_smooth_axis = np.linspace(vel_axis.min(), vel_axis.max(), 1000)
fit_curve = gaussian(vel_smooth_axis, *popt)

# === Plotting ===
plt.figure(figsize=(7, 5))

# 1. The DATA (as a smooth continuous line)
plt.plot(vel_axis, spectrum, color="black", lw=1.5, label="Observed")

# 2. The GAUSSIAN FIT (as a smooth colored line)
plt.plot(vel_smooth_axis, fit_curve, color="green", lw=2, label="Gaussian Fit")

# 3. Aesthetics
plt.axvline(7.4, color="blue", ls="--", alpha=0.5, label=r"$V_{sys}$ = 7.4 km/s")
plt.axhline(0, color="black", lw=0.8, alpha=0.3) # Zero baseline

plt.xlabel(r"LSRK Velocity [km s$^{-1}$]", fontsize=12)
plt.ylabel(r"Flux Density [Jy beam$^{-1}$]", fontsize=12)
plt.title(r" $^{13}$CO Spectral Profile", fontsize=13)

# Display Fit Results on Plot
stats_text = (f"Peak: {amp_fit:.2f} Jy\n"
              f"Center: {cen_fit:.2f} km/s\n"
              f"FWHM: {fwhm:.2f} km/s")
plt.text(0.05, 0.92, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

plt.legend(loc='upper right', frameon=False)
plt.xlim(vel_axis.min(), vel_axis.max())
plt.tight_layout()

plt.show()