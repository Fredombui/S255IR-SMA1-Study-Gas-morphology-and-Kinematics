import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Define known fluxes (from literature)
# ------------------------------------------------------------
# Format: [pre_day, pre_flux, post_day, post_flux]
data = {
    "6 GHz":   [554, 16.8, 2590, 4.8],
    "10 GHz":  [554, 22.2, 2590, 6.9],
    "22 GHz":  [554, 25.0, 2590, 7.8],
    "92 GHz":  [1061, 46.0, 2166, 18.8]  # for 92 GHz, both earlier/later data available
}

# Define the interpolation day (mid-2022 ≈ 2166 days since 2016-07-10)
interp_day = 2166

# ------------------------------------------------------------
# Compute interpolated fluxes
# ------------------------------------------------------------
results = []
for band, (d1, f1, d2, f2) in data.items():
    # linear interpolation
    f_interp = f1 + (f2 - f1) * (interp_day - d1) / (d2 - d1)
    
    # assign error = 10% flux + interpolation uncertainty (~10%)
    flux_err = 0.15 * f_interp  # combined 10% + interpolation uncertainty
    
    results.append([band, d1, f1, d2, f2, interp_day, f_interp, flux_err])

# Convert to dataframe for table output
df = pd.DataFrame(results, columns=[
    "Band", "Pre-day", "Pre-flux (mJy)", "Post-day", "Post-flux (mJy)",
    "Interp-day", "Interp. flux (mJy)", "Error (mJy)"
])

# Print as nice text table
print(df.to_string(index=False, justify='center', formatters={
    "Pre-flux (mJy)": "{:.2f}".format,
    "Post-flux (mJy)": "{:.2f}".format,
    "Interp. flux (mJy)": "{:.2f}".format,
    "Error (mJy)": "{:.2f}".format
}))

# ------------------------------------------------------------
# Plot the interpolation
# ------------------------------------------------------------
plt.figure(figsize=(8,6))
colors = ['tab:blue','tab:orange','tab:green','tab:red']

for i, (band, (d1, f1, d2, f2)) in enumerate(data.items()):
    # plot measured points
    plt.plot([d1, d2], [f1, f2], 'o-', color=colors[i], label=band)
    
    # interpolated point
    f_interp = df.loc[i, "Interp. flux (mJy)"]
    flux_err = df.loc[i, "Error (mJy)"]
    plt.errorbar(interp_day, f_interp, yerr=flux_err, fmt='s', color=colors[i], markersize=6)
    plt.text(interp_day+20, f_interp, f"{f_interp:.1f}±{flux_err:.1f}", fontsize=8, color=colors[i])

plt.title("Interpolated Radio Fluxes for S255IR-NIRS3 (mid-2022)")
plt.xlabel("Days since 2016 July 10 (outburst start)")
plt.ylabel("Flux Density (mJy)")
plt.xlim(0, 2700)
plt.ylim(0, 60)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
