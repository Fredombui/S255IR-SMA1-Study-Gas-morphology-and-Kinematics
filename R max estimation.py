from astropy.coordinates import SkyCoord
import astropy.units as u

# 1. Input the coordinates (RA, Dec) as strings or degrees
# Center of S255IR-SMA1
center_coord = SkyCoord('06h12m54.01s', '+17d59m23.0s', frame='icrs') 

# Furthest tip of the 3-sigma contour for the Blue Lobe
blue_tip_coord = SkyCoord('06h12m54.1s', '+17d59m19s', frame='icrs')

# 2. Calculate the angular separation
separation = center_coord.separation(blue_tip_coord)

# 3. Convert to arcseconds
R_max_arcsec_blue = separation.arcsec
print(f"R_max (red): {R_max_arcsec_blue:.3f} arcsec")
