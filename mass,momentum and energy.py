import numpy as np
import astropy.units as u
from astropy.constants import m_p, M_sun
from spectral_cube import SpectralCube
from scipy.optimize import fsolve
import warnings

# Suppress warnings from blank/NaN pixels in the ALMA cubes
warnings.filterwarnings('ignore')

# ==========================================
# 1. DEFINE SOURCE PARAMETERS
# ==========================================
T_ex = 75.5 * u.K           # Derived from 12CO peak
R_iso = 84.4                # [12CO]/[13CO] from Wilson & Rood (1994)
distance = 1.78 * u.kpc     # Heliocentric distance to S255IR
X_co = 1e-4                 # Canonical 12CO/H2 abundance
mu_H2 = 2.8                 # Mean molecular weight
v_sys = 7.4                 # Systemic velocity (km/s)

print("--- [STEP 1: Parameters] ---")
print(f"Excitation Temp: {T_ex}")
print(f"Isotopic Ratio:  {R_iso}")
print(f"Distance:        {distance}")

# ==========================================
# 2. FILE NAMES AND VELOCITIES
# ==========================================
# Update file names to match your local directory
file_12co = 'CO_v0_line_imreframed.fits' 
file_13co = '13CO_v0_line1_imreframed2.fits'

# Velocity boundaries defining the high-velocity wings
blue_wing_min, blue_wing_max = -15.0 * u.km/u.s, 3.0 * u.km/u.s
red_wing_min, red_wing_max = 12.0 * u.km/u.s, 35.0 * u.km/u.s

# ==========================================
# 3. LOAD AND PREPARE DATA
# ==========================================
print("\n--- [STEP 3: Loading & Preparing Data] ---")
cube12 = SpectralCube.read(file_12co).with_spectral_unit(u.km/u.s, velocity_convention='radio').to(u.K)
cube13 = SpectralCube.read(file_13co).with_spectral_unit(u.km/u.s, velocity_convention='radio').to(u.K)

# Extract pixel scale from header for area calculation
pix_scale_deg = np.abs(cube12.header['CDELT2']) * u.deg
print(f"Pixel Scale: {pix_scale_deg.to(u.arcsec):.4f}")

# Interpolate 13CO to match 12CO grid exactly
cube13 = cube13.spectral_interpolate(cube12.spectral_axis)

# Slice the cubes into outflow wing slabs
blue_12 = cube12.spectral_slab(blue_wing_min, blue_wing_max)
blue_13 = cube13.spectral_slab(blue_wing_min, blue_wing_max)
red_12 = cube12.spectral_slab(red_wing_min, red_wing_max)
red_13 = cube13.spectral_slab(red_wing_min, red_wing_max)

# ==========================================
# 4. OPTICAL DEPTH SOLVER
# ==========================================
def solve_tau12(T12, T13, R):
    """Numerical solver for Maud et al. (2015) Eq 3."""
    if T13 <= 0.1 or T12 <= 0: 
        return 0.1 # Optically thin fallback for noise
    
    ratio_obs = T12 / T13
    
    # SAFETY: If ratio is too high, the solver will fail/runaway.
    if ratio_obs >= R * 0.98: return 50.0 
    
    def func(tau):
        return ((1 - np.exp(-tau)) / (1 - np.exp(-tau / R))) - ratio_obs
    
    tau_sol, info, ier, mesg = fsolve(func, 1.0, full_output=True)
    
    # Cap the result at 50 for physical stability in high-velocity gas
    return np.clip(tau_sol[0], 0.1, 50.0) if ier == 1 else 1.0

v_solve_tau = np.vectorize(solve_tau12)

print("\n--- [STEP 4: Solving Optical Depth voxel-by-voxel] ---")
tau12_blue = v_solve_tau(blue_12.unmasked_data[:].value, blue_13.unmasked_data[:].value, R_iso)
tau12_red = v_solve_tau(red_12.unmasked_data[:].value, red_13.unmasked_data[:].value, R_iso)

print(f"Max Tau12 (Blue): {np.nanmax(tau12_blue):.2f}")
print(f"Max Tau12 (Red):  {np.nanmax(tau12_red):.2f}")

# ==========================================
# 5. MASS CALCULATION
# ==========================================
print("\n--- [STEP 5: Mass Integration] ---")
dv = np.abs(cube12.spectral_axis[1] - cube12.spectral_axis[0]).to(u.km/u.s).value
print(f"Channel Width (dv): {dv:.3f} km/s")

# Correct C_LTE Constant for 12CO J=2-1 at 75.5 K
C_LTE = 5.61e12 * (T_ex.value + 0.92) / (1 - np.exp(-11.06 / T_ex.value))
print(f"Corrected C_LTE Constant: {C_LTE:.2e} cm^-2 / (km/s)")

# Physical Area of a single pixel
pixel_size_cm = (pix_scale_deg.to(u.rad).value * distance).to(u.cm)
A_pix = pixel_size_cm**2
print(f"Pixel Area (A_pix): {A_pix.value:.2e} cm^2")

# Calculate mass per 3D voxel (in M_sun)
# Formula: tau * dv * C_LTE * (1/X_co) * mu_H2 * m_p * A_pix
mass_voxel_blue = (tau12_blue * dv * C_LTE * (u.cm**-2) / X_co * mu_H2 * m_p * A_pix).to(u.M_sun).value
mass_voxel_red  = (tau12_red * dv * C_LTE * (u.cm**-2) / X_co * mu_H2 * m_p * A_pix).to(u.M_sun).value

M_out_blue = np.nansum(mass_voxel_blue)
M_out_red  = np.nansum(mass_voxel_red)

# ==========================================
# 6. ENERGETICS (MOMENTUM & KINETIC ENERGY)
# ==========================================
print("\n--- [STEP 6: Calculating Energetics] ---")

v_axis_blue = blue_12.spectral_axis.to(u.km/u.s).value
v_axis_red = red_12.spectral_axis.to(u.km/u.s).value

# Relative velocities |v - v_sys| reshaped for 3D broadcasting
v_rel_blue = np.abs(v_axis_blue - v_sys)[:, np.newaxis, np.newaxis]
v_rel_red  = np.abs(v_axis_red - v_sys)[:, np.newaxis, np.newaxis]

# Momentum: Sum(Mass * v_rel)
P_blue = np.nansum(mass_voxel_blue * v_rel_blue)
P_red  = np.nansum(mass_voxel_red * v_rel_red)

# Kinetic Energy: 0.5 * Sum(Mass * v_rel^2)
erg_conv = (1 * u.M_sun * (1 * u.km/u.s)**2).to(u.erg).value
E_blue = 0.5 * np.nansum(mass_voxel_blue * (v_rel_blue**2)) * erg_conv
E_red  = 0.5 * np.nansum(mass_voxel_red * (v_rel_red**2)) * erg_conv

# ==========================================
# 7. FINAL SUMMARY
# ==========================================
print("\n" + "="*42)
print("FINAL RESULTS: S255IR-SMA1 OUTFLOW")
print("="*42)
print(f"Total Mass (M_out):      {(M_out_blue + M_out_red):.3f} M_sun")
print(f"Total Momentum (P_out):  {(P_blue + P_red):.2f} M_sun km/s")
print(f"Total Energy (E_k):      {(E_blue + E_red):.2e} ergs")
print("-" * 42)
print(f"Blue Lobe: {M_out_blue:.2f} M_sun | P={P_blue:.2f} | E={E_blue:.2e}")
print(f"Red Lobe:  {M_out_red:.2f} M_sun | P={P_red:.2f} | E={E_red:.2e}")
print("="*42)