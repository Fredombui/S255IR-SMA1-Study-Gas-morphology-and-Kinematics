import numpy as np
import astropy.units as u
from astropy.constants import m_p, M_sun
from spectral_cube import SpectralCube
from scipy.optimize import fsolve
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. DEFINE YOUR SOURCE PARAMETERS 
# ==========================================
T_ex = 75.5 * u.K           
R_iso = 84.4                
distance = 1.78 * u.kpc     
X_co = 1e-4                 
mu_H2 = 2.8                 

print("--- [STEP 1: Parameters] ---")
print(f"Excitation Temp: {T_ex}")
print(f"Isotopic Ratio:  {R_iso}")
print(f"Distance:        {distance}")

# ==========================================
# 2. FILE NAMES AND VELOCITIES
# ==========================================
file_12co = 'CO_v0_line_imreframed.fits' 
file_13co = '13CO_v0_line1_imreframed2.fits'

blue_wing_min, blue_wing_max = -15.0 * u.km/u.s, 3.0 * u.km/u.s
red_wing_min, red_wing_max = 12.0 * u.km/u.s, 35.0 * u.km/u.s

# ==========================================
# 3. LOAD AND PREPARE DATA
# ==========================================
print("\n--- [STEP 3: Loading Data] ---")
cube12 = SpectralCube.read(file_12co).with_spectral_unit(u.km/u.s, velocity_convention='radio').to(u.K)
cube13 = SpectralCube.read(file_13co).with_spectral_unit(u.km/u.s, velocity_convention='radio').to(u.K)

pix_scale_deg = np.abs(cube12.header['CDELT2']) * u.deg
print(f"Pixel Scale: {pix_scale_deg.to(u.arcsec):.4f}")

cube13 = cube13.spectral_interpolate(cube12.spectral_axis)

blue_12 = cube12.spectral_slab(blue_wing_min, blue_wing_max)
blue_13 = cube13.spectral_slab(blue_wing_min, blue_wing_max)
red_12 = cube12.spectral_slab(red_wing_min, red_wing_max)
red_13 = cube13.spectral_slab(red_wing_min, red_wing_max)

# ==========================================
# 4. OPTICAL DEPTH SOLVER (WITH SAFETY CAP)
# ==========================================
def solve_tau12(T12, T13, R):
    # If 13CO is non-detectable or negative, assume optically thin limit
    if T13 <= 0.1 or T12 <= 0: return 0.1 
    
    ratio_obs = T12 / T13
    
    # SAFETY: If ratio approaches the Isotopic limit, solver runs away.
    # We cap the ratio to avoid infinite Tau.
    if ratio_obs >= R * 0.98: return 50.0 
    
    def func(tau):
        return ((1 - np.exp(-tau)) / (1 - np.exp(-tau / R))) - ratio_obs
    
    tau_sol, info, ier, mesg = fsolve(func, 1.0, full_output=True)
    
    # Cap the result at 50 for physical stability in outflows
    return np.clip(tau_sol[0], 0.1, 50.0) if ier == 1 else 1.0

v_solve_tau = np.vectorize(solve_tau12)

print("\n--- [STEP 4: Solving Optical Depth] ---")
tau12_blue = v_solve_tau(blue_12.unmasked_data[:].value, blue_13.unmasked_data[:].value, R_iso)
tau12_red = v_solve_tau(red_12.unmasked_data[:].value, red_13.unmasked_data[:].value, R_iso)
print(f"Max Tau12 (Blue): {np.nanmax(tau12_blue):.2f}")
print(f"Max Tau12 (Red):  {np.nanmax(tau12_red):.2f}")

# ==========================================
# 5. COLUMN DENSITY AND MASS
# ==========================================
print("\n--- [STEP 5: Constants & Area] ---")
dv = np.abs(cube12.spectral_axis[1] - cube12.spectral_axis[0]).to(u.km/u.s).value
print(f"Channel Width (dv): {dv:.3f} km/s")

# CORRECTED C_LTE Constant for 12CO J=2-1 at Tex = 75.5 K 
# Standard formula: 5.61e12 * (Tex + 0.92) / (1 - exp(-11.06 / Tex))
C_LTE = 5.61e12 * (T_ex.value + 0.92) / (1 - np.exp(-11.06 / T_ex.value))
print(f"Corrected C_LTE Constant: {C_LTE:.2e} cm^-2 / (km/s)")

pixel_size_cm = (pix_scale_deg.to(u.rad).value * distance).to(u.cm)
A_pix = pixel_size_cm**2
print(f"Physical Pixel Size: {pixel_size_cm.value:.2e} cm")
print(f"Pixel Area (A_pix):   {A_pix.value:.2e} cm^2")

# Integrate tau over velocity
int_tau_blue = np.nansum(tau12_blue, axis=0) * dv
int_tau_red = np.nansum(tau12_red, axis=0) * dv

# Calculate Column Density Maps (Resulting unit: cm^-2)
N12_blue_map = (C_LTE * int_tau_blue) * (u.cm**-2)
N12_red_map  = (C_LTE * int_tau_red) * (u.cm**-2)

# Calculate mass per pixel (cm^-2 * cm^2 * kg = kg)
mass_map_blue = (N12_blue_map / X_co) * (mu_H2 * m_p * A_pix)
mass_map_red  = (N12_red_map / X_co) * (mu_H2 * m_p * A_pix)

M_out_blue = np.nansum(mass_map_blue).to(u.M_sun)
M_out_red  = np.nansum(mass_map_red).to(u.M_sun)

print("\n==========================================")
print("OUTFLOW MASS RESULTS")
print("==========================================")
print(f"Blueshifted Lobe Mass: {M_out_blue.value:.3f} M_sun")
print(f"Redshifted Lobe Mass:  {M_out_red.value:.3f} M_sun")
print(f"Total Outflow Mass:    {(M_out_blue.value + M_out_red.value):.3f} M_sun")
