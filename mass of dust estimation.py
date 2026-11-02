import astropy.units as u
from astropy.constants import c
from astropy.modeling.models import BlackBody

# 1. Define your known parameters
flux_density = 115 * u.mJy
dust_temp = 216 * u.K

# =====================================================================
# 2. SPECIFIC SOURCE VALUES (Updated)
# =====================================================================
distance = 1.78 * u.kpc               # Distance to source (Burns et al. 2016)

# Wavelength of 1.3mm converted to frequency
wavelength = 1.3 * u.mm
frequency = (c / wavelength).to(u.GHz) 

kappa_nu = 0.899 * (u.cm**2 / u.g)    # Dust opacity at 1.3mm (Ossenkopf & Henning 1994)
# =====================================================================

# 3. Calculate the Planck function B_nu(T_dust)
# The BlackBody model returns specific intensity in units containing sr^-1
bb = BlackBody(temperature=dust_temp)
B_nu = bb(frequency) 

# 4. Calculate Dust Mass: M_dust = (S_nu * d^2) / (kappa_nu * B_nu)
# We multiply B_nu by u.sr so the steradian unit cancels out properly 
# against the implicit solid angle derived from (S_nu / B_nu).
M_dust = (flux_density * distance**2) / (kappa_nu * B_nu * u.sr)

# 5. Convert to physical mass units
M_dust_sol = M_dust.to(u.M_sun)
M_dust_earth = M_dust.to(u.M_earth)

# Print the results
print(f"Flux Density: {flux_density}")
print(f"Dust Temp:    {dust_temp}")
print(f"Distance:     {distance}")
print(f"Frequency:    {frequency:.2f} (approx. {wavelength})")
print(f"Opacity:      {kappa_nu}")
print("-" * 35)
print(f"Calculated Dust Mass: {M_dust_sol:.4e} ")
print(f"                      ({M_dust_earth:.1f})")

# Calculate total mass (gas + dust), assuming a standard gas-to-dust ratio of 100
gas_to_dust_ratio = 100
M_total = M_dust_sol * gas_to_dust_ratio
print(f"Total Mass (Gas+Dust): {M_total:.4e}")