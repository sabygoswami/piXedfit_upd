# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 11:11:09 2022

@author: Sabyasachi
"""

#import numpy as np
#from astropy.io import fits
#import matplotlib.pyplot as plt
#from piXedfit.piXedfit_spectrophotometric import match_imgifs_spatial
#
#photo_fluxmap = "/home/saby/cubes/fluxmap_ASASSN14co.fits"              # photometric data cube
#ifs_data = "/home/saby/cubes/ASASSN-14co_ZAP50_svd100.fits"          # IFS data cube
#ifs_survey = "califa"                                    # IFS survey
#nproc = 6                                              # Number of processor
#name_out_fits = "specphoto_fluxmap_12073-12703.fits"         # Desired output name
#match_imgifs_spatial(photo_fluxmap, ifs_data, ifs_survey=ifs_survey, nproc=nproc, name_out_fits=name_out_fits)
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


cube = fits.open("D:/IFS/specphoto_fluxmap_12073-12703.fits")
cube.info()
print (cube[0].header)


header = cube[0].header

# get photometry and IFS regions
photo_region = cube['photo_region'].data
spec_region = cube['spec_region'].data 

# get unit of flux
unit_flux = float(header['unit'])         # in erg/s/cm2/A

# get maps of photometric fluxes
map_fluxes = cube['photo_flux'].data

# get photometric SEDs of individual pixels
# transpose from (band,y,x) to (y,x,band)
pix_photo_flux = np.transpose(cube['photo_flux'].data, axes=(1,2,0))*unit_flux
pix_photo_flux_err = np.transpose(cube['photo_fluxerr'].data, axes=(1,2,0))*unit_flux

# get spectra of individual pixels
# transpose from (wave,y,x) to (y,x,wave)
pix_spec_flux = np.transpose(cube['spec_flux'].data, axes=(1,2,0))*unit_flux
pix_spec_flux_err = np.transpose(cube['spec_fluxerr'].data, axes=(1,2,0))*unit_flux

# get wavelength of the spectra
spec_wave = cube['wave'].data

cube.close()


from astropy.visualization import make_lupton_rgb

g = map_fluxes[3]*10
r = map_fluxes[4]*10
i = map_fluxes[5]*10

rgb_default = make_lupton_rgb(i, r, g)

fig1 = plt.figure(figsize=(5,5))
f1 = plt.subplot()
plt.xlabel('[pixel]', fontsize=15)
plt.ylabel('[pixel]', fontsize=15)

plt.imshow(rgb_default, origin='lower', alpha=1.0)
plt.imshow(spec_region, origin='lower', cmap='Greys', alpha=0.2)

nbands = int(header['nfilters'])

filters = []
for bb in range(0,nbands):
    str_temp = 'fil%d' % bb
    filters.append(header[str_temp])

# get central wavelength of filters
from piXedfit.utils.filtering import cwave_filters
photo_wave = cwave_filters(filters)


from matplotlib.ticker import ScalarFormatter

nwaves = len(spec_wave)

for yy in range(45,55):
    for xx in range(45,55):
        photo_SED = pix_photo_flux[yy][xx]
        photo_SED_err = pix_photo_flux_err[yy][xx]
        spec_SED = pix_spec_flux[yy][xx]
        spec_SED_err = pix_spec_flux_err[yy][xx]

        fig1 = plt.figure(figsize=(14,7))
        f1 = plt.subplot()   
        plt.title("pixel: (x=%d, y=%d)" % (xx,yy), fontsize=20)
        f1.set_yscale('log')
        f1.set_xscale('log')
        plt.setp(f1.get_yticklabels(), fontsize=14)
        plt.setp(f1.get_xticklabels(), fontsize=14)
        plt.xlabel(r'Wavelength $[\AA]$', fontsize=21)
        plt.ylabel(r'$F_{\lambda}$ [erg $s^{-1}cm^{-2}\AA^{-1}$]', fontsize=21)
        xticks = [2000,3000,4000,6000,8000,10000,20000,30000,50000]
        plt.xticks(xticks)
        plt.ylim(1.0e-19,8e-16)
        for axis in [f1.xaxis]:
            axis.set_major_formatter(ScalarFormatter())

        # Optional: cut the spectra around the edges
        plt.plot(spec_wave[20:nwaves-20], spec_SED[20:nwaves-20], lw=1.0, color='red')
        plt.errorbar(photo_wave, photo_SED, yerr=photo_SED_err, markersize=10,
                                color='blue', fmt='s', lw=2)
        plt.savefig("D:/IFS/" + xx + yy + '.png')