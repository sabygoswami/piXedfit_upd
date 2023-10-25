# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 16:32:27 2022

@author: Sabyasachi
"""

from piXedfit.piXedfit_spectrophotometric import match_imgifs_spectral

specphoto_file = "/home/saby/piXedfit/specphoto_fluxmap_12073-12703.fits"
nproc = 2
name_out_fits = "corr_%s" % specphoto_file
match_imgifs_spectral(specphoto_file, nproc=nproc, name_out_fits=name_out_fits)