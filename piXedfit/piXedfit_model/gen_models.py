import numpy as np
import math
import sys, os
import random
#import fsps
import operator
from astropy.io import fits
from scipy.interpolate import interp1d

from ..utils.redshifting import cosmo_redshifting
from ..utils.filtering import filtering, cwave_filters
from ..utils.igm_absorption import igm_att_madau, igm_att_inoue
from .model_utils import *

# warning is not logged here. Perfect for clean unit test output
with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0

try: 
	global PIXEDFIT_HOME
	PIXEDFIT_HOME = os.environ['PIXEDFIT_HOME']
except:
	print ("PIXEDFIT_HOME should be included in your PATH!")


__all__ = ["generate_modelSED_propspecphoto", "generate_modelSED_spec", "generate_modelSED_photo", "generate_modelSED_specphoto", 
			"generate_modelSED_spec_decompose", "generate_modelSED_specphoto_decompose","save_models_photo", "save_models_rest_spec",
			"add_fagn_bol_samplers"]

def generate_modelSED_propspecphoto(sp=None,imf_type=1,duste_switch=1,add_neb_emission=1,dust_law=1,
	sfh_form=4,add_agn=0,filters=['galex_fuv','galex_nuv','sdss_u','sdss_g','sdss_r','sdss_i','sdss_z'],
	add_igm_absorption=0,igm_type=0,cosmo='flat_LCDM',H0=70.0,Om0=0.3,gas_logu=-2.0,params_val={'log_mass':0.0,
	'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,
	'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,'logzsol':0.0}):
	"""A function to generate model SED in which the output includes: properties, spectrum, and photometric fluxes

	:param sp:
		Initialization of FSPS, such as sp=fsps.StellarPopulation()

	:param imf_type:
		Choice for the IMF. Choices are: [0:Salpeter(1955), 1:Chabrier(2003), 2:Kroupa(2001)]

	:param duste_switch:
		Choice for turning on (1) or off (0) the dust emission modeling

	:param add_neb_emission:
		Choice for turning on (1) or off (0) the nebular emission modeling

	:param dust_law: (default: 1)
		Choice for the dust attenuation law. Options are: (a) 0 for Charlot & Fall (2000), (b) 1 for Calzetti et al. (2000).

	:param sfh_form: (default: 4)
		Choice for the parametric SFH model. Options are: (a) 0 for exponentially declining or tau model, (b) 1 for delayed tau model, (c) 2 for log normal model, (d) 3 for Gaussian form, (e) 4 for double power-law model.

	:param add_agn:
		Choice for turning on (1) or off (0) the AGN dusty torus modeling

	:param filters:
		List of photometric filters.

	:param add_igm_absorption:
		Switch for the IGM absorption.

	:param igm_type: (default: 0)
		Choice for the IGM absorption model. Options are: (a) 0 for Madau (1995), and (b) 1 for Inoue+(2014).

	:param cosmo (default: 'flat_LCDM'):
		Choices for the cosmological parameters. The choices are: ['flat_LCDM', 'WMAP5', 
		'WMAP7', 'WMAP9', 'Planck13', 'Planck15'], similar to the choices available in the 
		Astropy Cosmology package: https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies.
		If 'flat_LCDM' is chosen, the input H0 and Om0 should be provided.

	:param H0, Om0 (default: H0=70.0, Om0=0.3):
		Hubble constant and Omega matter at z=0.0

	:param gas_logu: (default: -2.0)
		Gas ionization parameter in logarithmic scale.

	:param sfh_t, sfh_sfr:
		arrays for arbitrary SFH. These parameters only relevant if sfh_form='arbitrary_sfh'. 

	:param param_val:
		A dictionary of parameters values.
	"""
	
	# input parameters:
	formed_mass = math.pow(10.0,params_val['log_mass'])
	age = math.pow(10.0,params_val['log_age'])
	tau = math.pow(10.0,params_val['log_tau'])
	t0 = math.pow(10.0,params_val['log_t0'])
	alpha = math.pow(10.0,params_val['log_alpha'])
	beta = math.pow(10.0,params_val['log_beta'])

	if sp == None:
		import fsps
		sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf_type)

	sp.params['imf_type'] = imf_type

	# dust emission:
	if duste_switch == 1:
		sp.params["add_dust_emission"] = True
		sp.params["duste_gamma"] = math.pow(10.0,params_val['log_gamma']) 
		sp.params["duste_umin"] = math.pow(10.0,params_val['log_umin'])
		sp.params["duste_qpah"] = math.pow(10.0,params_val['log_qpah'])
	elif duste_switch == 0:
		sp.params["add_dust_emission"] = False

	# nebular emission:
	if add_neb_emission == 1:
		sp.params["add_neb_emission"] = True
		sp.params['gas_logu'] = gas_logu
	elif add_neb_emission == 0:
		sp.params["add_neb_emission"] = False

	# AGN:
	if add_agn == 1:
		sp.params["fagn"] = math.pow(10.0,params_val['log_fagn'])
		sp.params["agn_tau"] = math.pow(10.0,params_val['log_tauagn'])
	elif add_agn == 0:
		sp.params["fagn"] = 0

	# other parameters:
	sp.params["logzsol"] = params_val['logzsol'] 
	sp.params['gas_logz'] = params_val['logzsol']
	sp.params['tage'] = age

	# generate the SED:
	SFR_fSM = -99.0			# for temporary
	if sfh_form==0 or sfh_form==1:
		if sfh_form==0:
			sp.params["sfh"] = 1
		elif sfh_form==1:
			sp.params["sfh"] = 4
		sp.params["const"] = 0
		sp.params["sf_start"] = 0
		sp.params["sf_trunc"] = 0
		sp.params["fburst"] = 0
		sp.params["tburst"] = 30.0
		sp.params["tau"] = tau
		#sp.params["tage"] = age

		# dust atenuation:
		if dust_law==0:
			sp.params["dust_type"] = 0  
			sp.params["dust_tesc"] = 7.0
			sp.params["dust_index"] = params_val['dust_index']
			dust1_index = -1.0
			sp.params["dust1_index"] = dust1_index
			sp.params["dust1"] = params_val['dust1']
			sp.params["dust2"] = params_val['dust2']
		elif dust_law==1:
			sp.params["dust_type"] = 2  
			sp.params["dust1"] = 0
			sp.params["dust2"] = params_val['dust2']

		wave, extnc_spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
		mass = sp.stellar_mass
		dust_mass0 = sp.dust_mass   # in solar mass/norm

		# total bolometric luminosity including AGN
		lbol_agn = calc_bollum_from_spec_rest(spec_wave=wave, spec_lum=extnc_spec)

		# bolometric luminosity excluding AGN
		sp.params["fagn"] = 0.0
		wave9, spec9 = sp.get_spectrum(peraa=True,tage=age) 			# spectrum in L_sun/AA
		lbol_noagn = calc_bollum_from_spec_rest(spec_wave=wave9,spec_lum=spec9)

		# get fraction of AGN luminosity from the total bolometric luminosity
		fagn_bol = (lbol_agn-lbol_noagn)/lbol_agn
		log_fagn_bol = np.log10(fagn_bol)

		# normalization:
		norm0 = formed_mass/mass
		extnc_spec = extnc_spec*norm0
		dust_mass = dust_mass0*norm0

		# calculate SFR:
		SFR_exp = 1.0/np.exp(age/tau)
		if sfh_form==0:
			SFR_fSM = formed_mass*SFR_exp/tau/(1.0-SFR_exp)/1e+9
		elif sfh_form==1:
			SFR_fSM = formed_mass*age*SFR_exp/((tau*tau)-((age*tau)+(tau*tau))*SFR_exp)/1e+9

	elif sfh_form==2 or sfh_form==3 or sfh_form==4:
		sp.params["sfh"] = 3
		# dust atenuation:
		if dust_law==0:
			sp.params["dust_type"] = 0  
			sp.params["dust_tesc"] = 7.0
			sp.params["dust_index"] = params_val['dust_index']
			dust1_index = -1.0
			sp.params["dust1_index"] = dust1_index
			sp.params["dust1"] = params_val['dust1']
			sp.params["dust2"] = params_val['dust2']
		elif dust_law==1:
			sp.params["dust_type"] = 2  
			sp.params["dust1"] = 0
			sp.params["dust2"] = params_val['dust2']

		SFR_fSM,mass,wave,extnc_spec,dust_mass = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)

		# total bolometric luminosity including AGN
		lbol_agn = calc_bollum_from_spec_rest(spec_wave=wave,spec_lum=extnc_spec)

		# bolometric luminosity excluding AGN
		sp.params["fagn"] = 0.0		
		SFR_fSM9,mass9,wave9,spec9,dust_mass9 = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)
		lbol_noagn = calc_bollum_from_spec_rest(spec_wave=wave9,spec_lum=spec9)

		# get fraction of AGN luminosity from the total bolometric luminosity
		fagn_bol = (lbol_agn-lbol_noagn)/lbol_agn
		log_fagn_bol = np.log10(fagn_bol)


	# redshifting
	#redsh_wave,redsh_spec0 = redshifting.cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)
	redsh_wave,redsh_spec0 = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)

	# IGM absorption:
	redsh_spec = redsh_spec0
	if add_igm_absorption == 1:
		if igm_type==0:
			trans = igm_att_madau(redsh_wave,params_val['z'])
			redsh_spec = redsh_spec0*trans
		elif igm_type==1:
			trans = igm_att_inoue(redsh_wave,params_val['z'])
			redsh_spec = redsh_spec0*trans

	# filtering:
	photo_SED_flux = filtering(redsh_wave,redsh_spec,filters)
	
	# get central wavelength of all filters:
	photo_cwave = cwave_filters(filters)

	# calculate mw-age:
	mw_age = calc_mw_age(sfh_form=sfh_form,tau=tau,t0=t0,alpha=alpha,beta=beta,
				age=age,formed_mass=formed_mass)

	# outputs:
	SED_prop = {}
	SED_prop['SM'] = formed_mass
	#SED_prop['survive_mass'] = mass
	SED_prop['SFR'] = SFR_fSM
	SED_prop['mw_age'] = mw_age
	SED_prop['dust_mass'] = dust_mass
	SED_prop['log_fagn_bol'] = log_fagn_bol

	spec_SED = {}
	spec_SED['wave'] = redsh_wave
	spec_SED['flux'] = redsh_spec

	photo_SED = {}
	photo_SED['wave'] = photo_cwave
	photo_SED['flux'] = photo_SED_flux

	return SED_prop, photo_SED, spec_SED



def generate_modelSED_spec(sp=None,imf_type=1,duste_switch=1,add_neb_emission=1,dust_law=1,sfh_form=4,
	add_agn=0,add_igm_absorption=0,igm_type=0,cosmo='flat_LCDM',H0=70.0,Om0=0.3,gas_logu=-2.0,params_val={'log_mass':0.0,
	'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,
	'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,'logzsol':0.0}):
	"""Function for generating a model spectrum given some parameters.

	:param sp: (optional, default: None)
		Initialization of FSPS, such as `sp=fsps.StellarPopulation()`. This is intended for rapid generation of model spectra from FSPS.
		However, this input is optional. If sp=None, FSPS will be called everytime this function is called.

	:param imf_type:
		Choice for the IMF. Choices are: 0 for Salpeter(1955), 1 for Chabrier(2003), and 2 for Kroupa(2001).

	:param duste_switch:
		Choice for switching on (value: 1) or off (value: 0) the dust emission modeling.

	:param add_neb_emission:
		Choice for switching on (value: 1) or off (value: 0) the nebular emission modeling.

	:param dust_law:
		Choice for the dust attenuation law. Options are: 0 for Charlot & Fall (2000) and 1 for Calzetti et al. (2000).

	:param sfh_form:
		Choice for the parametric SFH model. Options are: 0 for exponentially declining or tau model, 1 for delayed tau model, 2 for log normal model, 3 for Gaussian form, and 4 for double power-law model.

	:param add_agn:
		Choice for turning on (value: 1) or off (value: 0) the AGN dusty torus modeling.

	:param add_igm_absorption: 
		Choice for turning on (value: 1) or off (value: 0) the IGM absorption modeling.

	:param igm_type:
		Choice for the IGM absorption model. Options are: 0 for Madau (1995) and 1 for Inoue+(2014).

	:param cosmo: 
		Choices for the cosmology. Options are: (1)'flat_LCDM' or 0, (2)'WMAP5' or 1, (3)'WMAP7' or 2, (4)'WMAP9' or 3, (5)'Planck13' or 4, (6)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0:
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0:
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param gas_logu:
		Gas ionization parameter in logarithmic scale.

	:param param_val:
		Dictionary of the input values of the parameters. Should folllow the structure given in the default set. 
		Summary of the parameters are tabulated in Table 1 of `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210109717A/abstract>`_.

	:returns spec_SED:
		Array containing output model spectrum. It consists of spec_SED['wave'], which is the wavelengths grids, and spec_SED['flux'], which is the fluxes or the spectrum. 
	"""
	
	# input parameters:
	formed_mass = math.pow(10.0,params_val['log_mass'])
	age = math.pow(10.0,params_val['log_age'])
	tau = math.pow(10.0,params_val['log_tau'])
	t0 = math.pow(10.0,params_val['log_t0'])
	alpha = math.pow(10.0,params_val['log_alpha'])
	beta = math.pow(10.0,params_val['log_beta'])

	if sp == None:
		import fsps 
		sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf_type)

	sp.params['imf_type'] = imf_type

	# dust emission:
	if duste_switch == 1:
		sp.params["add_dust_emission"] = True
		sp.params["duste_gamma"] = math.pow(10.0,params_val['log_gamma']) 
		sp.params["duste_umin"] = math.pow(10.0,params_val['log_umin'])
		sp.params["duste_qpah"] = math.pow(10.0,params_val['log_qpah'])
	elif duste_switch == 0:
		sp.params["add_dust_emission"] = False

	# nebular emission:
	if add_neb_emission == 1:
		sp.params["add_neb_emission"] = True
		sp.params['gas_logu'] = gas_logu
	elif add_neb_emission == 0:
		sp.params["add_neb_emission"] = False

	# AGN:
	if add_agn == 1:
		sp.params["fagn"] = math.pow(10.0,params_val['log_fagn'])
		sp.params["agn_tau"] = math.pow(10.0,params_val['log_tauagn'])
	elif add_agn == 0:
		sp.params["fagn"] = 0

	# dust atenuation:
	if dust_law==0:
		sp.params["dust_type"] = 0  
		sp.params["dust_tesc"] = 7.0
		sp.params["dust_index"] = params_val['dust_index']
		dust1_index = -1.0
		sp.params["dust1_index"] = dust1_index
		sp.params["dust1"] = params_val['dust1']
		sp.params["dust2"] = params_val['dust2']
	elif dust_law==1:
		sp.params["dust_type"] = 2  
		sp.params["dust1"] = 0
		sp.params["dust2"] = params_val['dust2']

	# other parameters:
	sp.params["logzsol"] = params_val['logzsol'] 
	sp.params['gas_logz'] = params_val['logzsol']
	sp.params['tage'] = age

	# generate the SED:
	if sfh_form==0 or sfh_form==1:
		if sfh_form==0:
			sp.params["sfh"] = 1
		elif sfh_form==1:
			sp.params["sfh"] = 4
		sp.params["const"] = 0
		sp.params["sf_start"] = 0
		sp.params["sf_trunc"] = 0
		sp.params["fburst"] = 0
		sp.params["tburst"] = 30.0
		sp.params["tau"] = tau
		#sp.params["tage"] = age

		wave, extnc_spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
		# get model mass:
		mass = sp.stellar_mass
		# get dust mass: 
		#dust_mass0 = sp.dust_mass    # in solar mass/norm

		# normalization:
		norm0 = formed_mass/mass
		extnc_spec = extnc_spec*norm0

	elif sfh_form==2 or sfh_form==3 or sfh_form==4:
		sp.params["sfh"] = 3
		SFR_fSM,mass,wave,extnc_spec,dust_mass = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)
		
	# redshifting
	redsh_wave,redsh_spec0 = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)

	# IGM absorption:
	redsh_spec = redsh_spec0
	if add_igm_absorption == 1:
		if igm_type==0:
			trans = igm_att_madau(redsh_wave,params_val['z'])
			redsh_spec = redsh_spec0*trans
		elif igm_type==1:
			trans = igm_att_inoue(redsh_wave,params_val['z'])
			redsh_spec = redsh_spec0*trans

	spec_SED = {}
	spec_SED['wave'] = redsh_wave
	spec_SED['flux'] = redsh_spec

	return spec_SED


def generate_modelSED_photo(filters,sp=None,imf_type=1,duste_switch=0,add_neb_emission=1,dust_law=1,sfh_form=4,
	add_agn=0,add_igm_absorption=0,igm_type=0,cosmo='flat_LCDM',H0=70.0,Om0=0.3,gas_logu=-2.0,params_val={'log_mass':0.0,
	'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,
	'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,'logzsol':0.0}):
	"""Function for generating a model photometric SED given some parameters.

	:param filters: 
		List of photometric filters. The list of filters recognized by piXedfit can be accessed using :func:`piXedfit.utils.filtering.list_filters`. 
		Please see `this page <https://pixedfit.readthedocs.io/en/latest/manage_filters.html>`_ for information on managing filters that include listing available filters, adding, and removing filters. 

	:param sp: (optional, default: None)
		Initialization of FSPS, such as `sp=fsps.StellarPopulation()`. This is intended for rapid generation of model spectra from FSPS.
		However, this input is optional. If sp=None, FSPS will be called everytime this function is called.

	:param imf_type:
		Choice for the IMF. Choices are: 0 for Salpeter(1955), 1 for Chabrier(2003), and 2 for Kroupa(2001).

	:param duste_switch:
		Choice for switching on (value: 1) or off (value: 0) the dust emission modeling.

	:param add_neb_emission:
		Choice for switching on (value: 1) or off (value: 0) the nebular emission modeling.

	:param dust_law:
		Choice for the dust attenuation law. Options are: 0 for Charlot & Fall (2000) and 1 for Calzetti et al. (2000).

	:param sfh_form:
		Choice for the parametric SFH model. Options are: 0 for exponentially declining or tau model, 1 for delayed tau model, 2 for log normal model, 3 for Gaussian form, and 4 for double power-law model.

	:param add_agn:
		Choice for turning on (value: 1) or off (value: 0) the AGN dusty torus modeling.

	:param add_igm_absorption: 
		Choice for turning on (value: 1) or off (value: 0) the IGM absorption modeling.

	:param igm_type:
		Choice for the IGM absorption model. Options are: 0 for Madau (1995) and 1 for Inoue+(2014).

	:param cosmo: 
		Choices for the cosmology. Options are: (1)'flat_LCDM' or 0, (2)'WMAP5' or 1, (3)'WMAP7' or 2, (4)'WMAP9' or 3, (5)'Planck13' or 4, (6)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0:
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0:
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param gas_logu:
		Gas ionization parameter in logarithmic scale.

	:param param_val:
		Dictionary of the input values of the parameters. Should folllow the structure given in the default set. 
		Summary of the parameters are tabulated in Table 1 of `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210109717A/abstract>`_.

	:returns photo_SED:
		Output model photometric SED. It consists of photo_SED['wave'], which is the central wavelengths of the photometric filters, and photo_SED['flux'], which is the photometric fluxes. 
	"""

	# Input parameters
	formed_mass = math.pow(10.0,params_val['log_mass'])
	age = math.pow(10.0,params_val['log_age'])
	tau = math.pow(10.0,params_val['log_tau'])
	t0 = math.pow(10.0,params_val['log_t0'])
	alpha = math.pow(10.0,params_val['log_alpha'])
	beta = math.pow(10.0,params_val['log_beta'])

	# Initialize FSPS
	if sp == None:
		import fsps
		sp = fsps.StellarPopulation(zcontinuous=1, imf_type=imf_type)

	sp.params['imf_type'] = imf_type

	# dust emission:
	if duste_switch == 1:
		sp.params["add_dust_emission"] = True
		sp.params["duste_gamma"] = math.pow(10.0,params_val['log_gamma']) 
		sp.params["duste_umin"] = math.pow(10.0,params_val['log_umin'])
		sp.params["duste_qpah"] = math.pow(10.0,params_val['log_qpah'])
	elif duste_switch == 0:
		sp.params["add_dust_emission"] = False

	# nebular emission:
	if add_neb_emission == 1:
		sp.params["add_neb_emission"] = True
		sp.params['gas_logu'] = gas_logu
	elif add_neb_emission == 0:
		sp.params["add_neb_emission"] = False

	# AGN:
	if add_agn == 1:
		sp.params["fagn"] = math.pow(10.0,params_val['log_fagn'])
		sp.params["agn_tau"] = math.pow(10.0,params_val['log_tauagn'])
	elif add_agn == 0:
		sp.params["fagn"] = 0

	# dust atenuation:
	if dust_law==0:
		sp.params["dust_type"] = 0  
		sp.params["dust_tesc"] = 7.0
		sp.params["dust_index"] = params_val['dust_index']
		dust1_index = -1.0
		sp.params["dust1_index"] = dust1_index
		sp.params["dust1"] = params_val['dust1']
		sp.params["dust2"] = params_val['dust2']
	elif dust_law==1:
		sp.params["dust_type"] = 2  
		sp.params["dust1"] = 0
		sp.params["dust2"] = params_val['dust2']

	# other parameters:
	sp.params["logzsol"] = params_val['logzsol'] 
	sp.params['gas_logz'] = params_val['logzsol']
	sp.params['tage'] = age

	# generate the SED:
	if sfh_form==0 or sfh_form==1:
		if sfh_form==0:
			sp.params["sfh"] = 1
		elif sfh_form==1:
			sp.params["sfh"] = 4
		sp.params["const"] = 0
		sp.params["sf_start"] = 0
		sp.params["sf_trunc"] = 0
		sp.params["fburst"] = 0
		sp.params["tburst"] = 30.0
		sp.params["tau"] = tau
		#sp.params["tage"] = age

		wave, extnc_spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
		# get model mass:
		mass = sp.stellar_mass
		# get dust mass: 
		#dust_mass0 = sp.dust_mass   ## in solar mass/norm

		# normalization:
		norm0 = formed_mass/mass
		extnc_spec = extnc_spec*norm0

	elif sfh_form==2 or sfh_form==3 or sfh_form==4:
		sp.params["sfh"] = 3
		SFR_fSM,mass,wave,extnc_spec,dust_mass = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)
	else:
		print ("SFH choice is not recognized!")
		sys.exit()
		
	# redshifting
	#redsh_wave,redsh_spec0 = redshifting.cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)
	redsh_wave,redsh_spec0 = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)

	# IGM absorption:
	redsh_spec = redsh_spec0
	if add_igm_absorption == 1:
		if igm_type==0:
			trans = igm_att_madau(redsh_wave,params_val['z'])
			redsh_spec = redsh_spec0*trans
		elif igm_type==1:
			trans = igm_att_inoue(redsh_wave,params_val['z'])
			redsh_spec = redsh_spec0*trans

	# filtering:
	photo_SED_flux = filtering(redsh_wave,redsh_spec,filters)
	# get central wavelength of all filters:
	photo_cwave = cwave_filters(filters)

	photo_SED = {}
	photo_SED['wave'] = photo_cwave
	photo_SED['flux'] = photo_SED_flux

	return photo_SED


def generate_modelSED_specphoto(sp=None,imf_type=1,duste_switch=1,add_neb_emission=1,dust_law=1,sfh_form=4,
	add_agn=0,filters=['galex_fuv','galex_nuv','sdss_u','sdss_g','sdss_r','sdss_i','sdss_z'],add_igm_absorption=0,igm_type=0,
	cosmo='flat_LCDM',H0=70.0,Om0=0.3,gas_logu=-2.0,params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,
	'log_umin':0.0,'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,
	'log_tau':0.4,'logzsol':0.0}):
	"""A function to generate model spectrophotometric SED

	:param sp:
		Initialization of FSPS, such as sp=fsps.StellarPopulation()

	:param imf_type:
		Choice for the IMF. Choices are: [0:Salpeter(1955), 1:Chabrier(2003), 2:Kroupa(2001)]

	:param duste_switch:
		Choice for turning on (1) or off (0) the dust emission modeling

	:param add_neb_emission:
		Choice for turning on (1) or off (0) the nebular emission modeling

	:param dust_law: (default: 1)
		Choice for the dust attenuation law. Options are: (a) 0 for Charlot & Fall (2000), (b) 1 for Calzetti et al. (2000).

	:param sfh_form: (default: 4)
		Choice for the parametric SFH model. Options are: (a) 0 for exponentially declining or tau model, (b) 1 for delayed tau model, (c) 2 for log normal model, (d) 3 for Gaussian form, (e) 4 for double power-law model.

	:param add_agn:
		Choice for turning on (1) or off (0) the AGN dusty torus modeling

	:param filters:
		A list of photometric filters.

	:param add_igm_absorption:
		Switch for the IGM absorption.

	:param igm_type: (default: 0)
		Choice for the IGM absorption model. Options are: (a) 0 for Madau (1995), and (b) 1 for Inoue+(2014).

	:param cosmo (default: 'flat_LCDM'):
		Choices for the cosmological parameters. The choices are: ['flat_LCDM', 'WMAP5', 
		'WMAP7', 'WMAP9', 'Planck13', 'Planck15'], similar to the choices available in the 
		Astropy Cosmology package: https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies.
		If 'flat_LCDM' is chosen, the input H0 and Om0 should be provided.

	:param H0, Om0 (default: H0=70.0, Om0=0.3):
		Hubble constant and Omega matter at z=0.0. 

	:param gas_logu: (default: -2.0)
		Gas ionization parameter in logarithmic scale.

	:param param_val:
		A dictionary of parameters values.
	"""
	
	# some input parameters:
	formed_mass = math.pow(10.0,params_val['log_mass'])
	age = math.pow(10.0,params_val['log_age'])
	tau = math.pow(10.0,params_val['log_tau'])
	t0 = math.pow(10.0,params_val['log_t0'])
	alpha = math.pow(10.0,params_val['log_alpha'])
	beta = math.pow(10.0,params_val['log_beta'])

	sp.params['imf_type'] = imf_type

	# dust emission:
	if duste_switch == 1:
		sp.params["add_dust_emission"] = True
		sp.params["duste_gamma"] = math.pow(10.0,params_val['log_gamma']) 
		sp.params["duste_umin"] = math.pow(10.0,params_val['log_umin'])
		sp.params["duste_qpah"] = math.pow(10.0,params_val['log_qpah'])
	elif duste_switch == 0:
		sp.params["add_dust_emission"] = False

	# nebular emission:
	if add_neb_emission == 1:
		sp.params["add_neb_emission"] = True
		sp.params['gas_logu'] = gas_logu
	elif add_neb_emission == 0:
		sp.params["add_neb_emission"] = False

	# AGN:
	if add_agn == 1:
		sp.params["fagn"] = math.pow(10.0,params_val['log_fagn'])
		sp.params["agn_tau"] = math.pow(10.0,params_val['log_tauagn'])
	elif add_agn == 0:
		sp.params["fagn"] = 0

	# dust atenuation:
	if dust_law==0:
		sp.params["dust_type"] = 0  
		sp.params["dust_tesc"] = 7.0
		sp.params["dust_index"] = params_val['dust_index']
		dust1_index = -1.0
		sp.params["dust1_index"] = dust1_index
		sp.params["dust1"] = params_val['dust1']
		sp.params["dust2"] = params_val['dust2']
	elif dust_law==1:
		sp.params["dust_type"] = 2  
		sp.params["dust1"] = 0
		sp.params["dust2"] = params_val['dust2']

	# other parameters:
	sp.params["logzsol"] = params_val['logzsol'] 
	sp.params['gas_logz'] = params_val['logzsol']
	sp.params['tage'] = age

	if sfh_form==0 or sfh_form==1:
		if sfh_form==0:
			sp.params["sfh"] = 1
		elif sfh_form==1:
			sp.params["sfh"] = 4
		sp.params["const"] = 0
		sp.params["sf_start"] = 0
		sp.params["sf_trunc"] = 0
		sp.params["fburst"] = 0
		sp.params["tburst"] = 30.0
		sp.params["tau"] = tau
		#sp.params["tage"] = age
		
		wave, extnc_spec = sp.get_spectrum(peraa=True,tage=age) ## spectrum in L_sun/AA
		# get model mass:
		mass = sp.stellar_mass
		# get dust mass: 
		#dust_mass0 = sp.dust_mass   # in solar mass/norm

		# normalization:
		norm0 = formed_mass/mass
		extnc_spec = extnc_spec*norm0
	elif sfh_form==2 or sfh_form==3 or sfh_form==4:
		sp.params["sfh"] = 3
		SFR_fSM,mass,wave,extnc_spec,dust_mass = csp_spec_restframe_fit(sp=sp,sfh_form=sfh_form,formed_mass=formed_mass,
																age=age,tau=tau,t0=t0,alpha=alpha,beta=beta)
		
	# redshifting
	#redsh_wave,redsh_spec0 = redshifting.cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)
	redsh_wave,redsh_spec0 = cosmo_redshifting(cosmo=cosmo,H0=H0,Om0=Om0,z=params_val['z'],wave=wave,spec=extnc_spec)

	# IGM absorption:
	redsh_spec = redsh_spec0
	if add_igm_absorption == 1:
		if igm_type==0:
			trans = igm_att_madau(redsh_wave,params_val['z'])
			redsh_spec = redsh_spec0*trans
		elif igm_type==1:
			trans = igm_att_inoue(redsh_wave,params_val['z'])
			redsh_spec = redsh_spec0*trans

	# filtering:
	photo_SED_flux = filtering(redsh_wave,redsh_spec,filters)
	# get central wavelength of all filters:
	photo_cwave = cwave_filters(filters)

	spec_SED = {}
	spec_SED['wave'] = redsh_wave
	spec_SED['flux'] = redsh_spec

	photo_SED = {}
	photo_SED['wave'] = photo_cwave
	photo_SED['flux'] = photo_SED_flux

	return spec_SED,photo_SED

def generate_modelSED_spec_decompose(sp=None,imf=1, duste_switch=1,add_neb_emission=1,dust_law=1,add_agn=1,
	add_igm_absorption=0,igm_type=0,sfh_form=4,funit='erg/s/cm2/A',cosmo='flat_LCDM',H0=70.0,Om0=0.3,
	gas_logu=-2.0,params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,
	'log_gamma':-2.0,'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,
	'log_tau':0.4,'logzsol':0.0}):
	"""A function for generating model spectroscopic SED and decompose the SED into its components.

	:param funit:
		Flux unit. Options are: [0/'erg/s/cm2/A', 1/'erg/s/cm2', 2/'Jy']
	
	"""

	# allocate memories:
	spec_SED = {}
	spec_SED['wave'] = []
	spec_SED['flux_total'] = []
	spec_SED['flux_stellar'] = []
	spec_SED['flux_nebe'] = []
	spec_SED['flux_duste'] = []
	spec_SED['flux_agn'] = []

	# generate model spectrum: total
	spec_SED_tot = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=duste_switch,add_neb_emission=add_neb_emission,dust_law=dust_law,
					sfh_form=sfh_form,add_agn=add_agn,add_igm_absorption=add_igm_absorption,igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
	spec_SED['wave'] = spec_SED_tot['wave']
	spec_SED['flux_total'] = convert_unit_spec_from_ergscm2A(spec_SED_tot['wave'],spec_SED_tot['flux'],funit=funit)

	spec_SED_temp = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=0,add_neb_emission=0,dust_law=dust_law,sfh_form=sfh_form,add_agn=0,
					add_igm_absorption=add_igm_absorption,igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
	spec_flux_stellar = spec_SED_temp['flux']
	spec_SED['flux_stellar'] = convert_unit_spec_from_ergscm2A(spec_SED_temp['wave'],spec_flux_stellar,funit=funit)

	# get nebular emission:
	if add_neb_emission == 1:
		add_neb_emission_temp  = 1
		spec_SED_temp1 = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=0,add_neb_emission=add_neb_emission_temp,
							dust_law=dust_law,sfh_form=sfh_form,add_agn=0,add_igm_absorption=add_igm_absorption,igm_type=igm_type,
							cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
		add_neb_emission_temp  = 0
		spec_SED_temp2 = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=0,add_neb_emission=add_neb_emission_temp,
							dust_law=dust_law,sfh_form=sfh_form,add_agn=0,add_igm_absorption=add_igm_absorption,igm_type=igm_type,
							cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
		spec_flux_nebe = spec_SED_temp1['flux'] - spec_SED_temp2['flux']
		spec_SED['flux_nebe'] = convert_unit_spec_from_ergscm2A(spec_SED_temp['wave'],spec_flux_nebe,funit=funit)

	# get dust emission:
	if duste_switch == 1:
		duste_switch_temp = 0
		spec_SED_temp = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=duste_switch_temp,add_neb_emission=add_neb_emission,
							dust_law=dust_law,sfh_form=sfh_form,add_agn=add_agn,add_igm_absorption=add_igm_absorption,igm_type=igm_type,
							cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
		spec_flux_duste = spec_SED_tot['flux'] - spec_SED_temp['flux']
		spec_SED['flux_duste'] = convert_unit_spec_from_ergscm2A(spec_SED_temp['wave'],spec_flux_duste,funit=funit)
	# get AGN emission:
	if add_agn == 1:
		add_agn_temp = 0
		spec_SED_temp = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=duste_switch,add_neb_emission=add_neb_emission,
							dust_law=dust_law,sfh_form=sfh_form,add_agn=add_agn_temp,add_igm_absorption=add_igm_absorption,
							igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
		spec_flux_agn = spec_SED_tot['flux'] - spec_SED_temp['flux']
		spec_SED['flux_agn'] = convert_unit_spec_from_ergscm2A(spec_SED_temp['wave'],spec_flux_agn,funit=funit)


	return spec_SED


def generate_modelSED_specphoto_decompose(sp=None, imf=1, duste_switch=1,add_neb_emission=1,dust_law=1,
	add_agn=1,add_igm_absorption=0,igm_type=0,cosmo='flat_LCDM',H0=70.0,Om0=0.3,gas_logu=-2.0,sfh_form=4,
	funit='erg/s/cm2/A',filters=['galex_fuv','galex_nuv','sdss_u','sdss_g','sdss_r','sdss_i','sdss_z'],
	params_val={'log_mass':0.0,'z':0.001,'log_fagn':-3.0,'log_tauagn':1.0,'log_qpah':0.54,'log_umin':0.0,'log_gamma':-2.0,
	'dust1':0.5,'dust2':0.5,'dust_index':-0.7,'log_age':1.0,'log_alpha':0.1,'log_beta':0.1,'log_t0':0.4,'log_tau':0.4,
	'logzsol':0.0}):
	"""A function for generating model spectroscopic SED and decompose the SED into its components.
	
	:param funit:
		Flux unit. Options are: [0/'erg/s/cm2/A', 1/'erg/s/cm2', 2/'Jy']

	"""

	# allocate memories:
	spec_SED = {}
	spec_SED['wave'] = []
	spec_SED['flux_total'] = []
	spec_SED['flux_stellar'] = []
	spec_SED['flux_nebe'] = []
	spec_SED['flux_duste'] = []
	spec_SED['flux_agn'] = []

	photo_SED = {}
	photo_SED['wave'] = cwave_filters(filters)
	photo_SED['flux'] = []

	# generate model spectrum: total
	spec_SED_tot,photo_SED_tot = generate_modelSED_specphoto(sp=sp,imf_type=imf,duste_switch=duste_switch,add_neb_emission=add_neb_emission,
										dust_law=dust_law,sfh_form=sfh_form,add_agn=add_agn,filters=filters,add_igm_absorption=add_igm_absorption,
										igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
	spec_SED['wave'] = spec_SED_tot['wave']
	spec_SED['flux_total'] = convert_unit_spec_from_ergscm2A(spec_SED_tot['wave'],spec_SED_tot['flux'],funit=funit)

	photo_SED['flux'] = convert_unit_spec_from_ergscm2A(photo_SED_tot['wave'],photo_SED_tot['flux'],funit=funit)

	spec_SED_temp = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=0,add_neb_emission=0,dust_law=dust_law,sfh_form=sfh_form,
											add_agn=0,add_igm_absorption=add_igm_absorption,igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,
											params_val=params_val)
	spec_flux_stellar = spec_SED_temp['flux']
	spec_SED['flux_stellar'] = convert_unit_spec_from_ergscm2A(spec_SED_temp['wave'],spec_flux_stellar,funit=funit)


	# get nebular emission:
	if add_neb_emission == 1:
		spec_SED_temp1 = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=0,add_neb_emission=1,dust_law=dust_law,
							sfh_form=sfh_form,add_agn=add_agn,add_igm_absorption=add_igm_absorption,igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,
							params_val=params_val)

		spec_SED_temp2 = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=0,add_neb_emission=0,dust_law=dust_law,
							sfh_form=sfh_form,add_agn=add_agn,add_igm_absorption=add_igm_absorption,igm_type=igm_type,cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,
							params_val=params_val)

		spec_flux_nebe = spec_SED_temp1['flux'] - spec_SED_temp2['flux']
		spec_SED['flux_nebe'] = convert_unit_spec_from_ergscm2A(spec_SED_temp['wave'],spec_flux_nebe,funit=funit)

	# get dust emission:
	if duste_switch == 1:
		duste_switch_temp = 0
		spec_SED_temp = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=duste_switch_temp,add_neb_emission=add_neb_emission,
							dust_law=dust_law,sfh_form=sfh_form,add_agn=add_agn,add_igm_absorption=add_igm_absorption,igm_type=igm_type,
							cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
		spec_flux_duste = spec_SED_tot['flux'] - spec_SED_temp['flux']
		spec_SED['flux_duste'] = convert_unit_spec_from_ergscm2A(spec_SED_temp['wave'],spec_flux_duste,funit=funit)
	# get AGN emission:
	if add_agn == 1:
		add_agn_temp = 0
		spec_SED_temp = generate_modelSED_spec(sp=sp,imf_type=imf,duste_switch=duste_switch,add_neb_emission=add_neb_emission,
							dust_law=dust_law,sfh_form=sfh_form,add_agn=add_agn_temp,add_igm_absorption=add_igm_absorption,igm_type=igm_type,
							cosmo=cosmo,H0=H0,Om0=Om0,gas_logu=gas_logu,params_val=params_val)
		spec_flux_agn = spec_SED_tot['flux'] - spec_SED_temp['flux']
		spec_SED['flux_agn'] = convert_unit_spec_from_ergscm2A(spec_SED_temp['wave'],spec_flux_agn,funit=funit)

	return spec_SED,photo_SED


def save_models_photo(filters,gal_z,imf_type=1,sfh_form=4,dust_law=1,add_igm_absorption=0,igm_type=0,duste_switch=0,
	add_neb_emission=1,add_agn=0,gas_logu=-2.0,nmodels=100000,params_range ={'logzsol':[-2.0,0.2],'log_tau':[-1.0,1.5],
	'log_age':[-2.0,1.14],'log_alpha':[-2.0,2.0],'log_beta':[-2.0,2.0],'log_t0':[-1.0,1.14],'dust_index':[-2.2,0.4],'dust1':[0.0,4.0], 
	'dust2':[0.0,4.0],'log_gamma':[-4.0, 0.0],'log_umin':[-1.0,1.39],'log_qpah':[-3.0,1.0],'log_fagn':[-5.0,0.48],'log_tauagn':[0.7, 2.18]},
	nproc=10,cosmo=0,H0=70.0,Om0=0.3,name_out_fits=None):
	"""Function for generating a set of photometric model SEDs and store them into a FITS file.
	The values of the parameters are randomly generated and for each parameter, the random values are uniformly distributed.  

	:param filters: 
		List of photometric filters. The list of filters recognized by piXedfit can be accessed using :func:`piXedfit.utils.filtering.list_filters`. 
		Please see `this page <https://pixedfit.readthedocs.io/en/latest/manage_filters.html>`_ for information on managing filters that include listing available filters, adding, and removing filters. 

	:param gal_z:
		Galaxy's redshift.
	
	:param imf_type:
		Choice for the IMF. Choices are: 0 for Salpeter(1955), 1 for Chabrier(2003), and 2 for Kroupa(2001).

	:param sfh_form:
		Choice for the parametric SFH model. Options are: 0 for exponentially declining or tau model, 1 for delayed tau model, 2 for log normal model, 3 for Gaussian form, and 4 for double power-law model.

	:param dust_law:
		Choice for the dust attenuation law. Options are: 0 for Charlot & Fall (2000) and 1 for Calzetti et al. (2000).

	:param add_igm_absorption: 
		Choice for turning on (value: 1) or off (value: 0) the IGM absorption modeling.

	:param igm_type:
		Choice for the IGM absorption model. Options are: 0 for Madau (1995) and 1 for Inoue+(2014).

	:param duste_switch:
		Choice for switching on (value: 1) or off (value: 0) the dust emission modeling.

	:param add_neb_emission:
		Choice for switching on (value: 1) or off (value: 0) the nebular emission modeling.

	:param add_agn:
		Choice for turning on (value: 1) or off (value: 0) the AGN dusty torus modeling.

	:param gas_logu:
		Gas ionization parameter in logarithmic scale.

	:param nmodels:
		Number of model SEDs to be generated.

	:param params_range:
		Ranges of parameters in a dictionary format. Summary of the parameters are tabulated in Table 1 of `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210109717A/abstract>`_.

	:param nproc:
		Number of cores to be used in the calculations.

	:param cosmo: 
		Choices for the cosmology. Options are: (1)'flat_LCDM' or 0, (2)'WMAP5' or 1, (3)'WMAP7' or 2, (4)'WMAP9' or 3, (5)'Planck13' or 4, (6)'Planck15' or 5.
		These options are similar to the choices available in the `Astropy Cosmology <https://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies>`_ package.

	:param H0:
		The Hubble constant at z=0. Only relevant when cosmo='flat_LCDM' is chosen.

	:param Om0:
		The Omega matter at z=0.0. Only relevant when cosmo='flat_LCDM' is chosen.

	:returns name_out_fits:
		Desired name for the output FITS file. if None, a default name will be used.
	"""

	dir_file = PIXEDFIT_HOME+'/data/temp/'
	CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_model/'

	def_params_range ={'logzsol':[-2.0,0.2],'log_tau':[-1.0,1.5],'log_age':[-3.0,1.14],'log_alpha':[-2.0,2.0],
		'log_beta':[-2.0,2.0],'log_t0':[-1.0,1.14],'dust_index':[-2.2,0.4],'dust1':[0.0,4.0], 'dust2':[0.0,4.0],
		'log_gamma':[-4.0, 0.0],'log_umin':[-1.0,1.39],'log_qpah':[-3.0,1.0],'log_fagn':[-5.0,0.48],'log_tauagn':[0.7, 2.18]}

	# get keys in input params_range:
	keys = list(params_range.keys())

	# merge with the default one
	params_range1 = def_params_range
	for ii in range(0,len(keys)):
		params_range1[keys[ii]] = params_range[keys[ii]]

	# number of filters
	nbands = len(filters)

	# store in text file
	name_filters_list = "filters_list%d.dat" % (random.randint(0,10000))
	file_out = open(name_filters_list,"w")
	for ii in range(0,nbands):
		file_out.write("%s\n" % filters[int(ii)]) 
	file_out.close()

	# move the file to temp dir
	os.system('mv %s %s' % (name_filters_list,dir_file))

	# make configuration file
	name_config = "config_file%d.dat" % (random.randint(0,10000))
	file_out = open(name_config,"w")
	file_out.write("imf_type %d\n" % imf_type)
	file_out.write("add_neb_emission %d\n" % add_neb_emission)
	file_out.write("gas_logu %lf\n" % gas_logu)
	file_out.write("add_igm_absorption %d\n" % add_igm_absorption)
	file_out.write("sfh_form %d\n" % sfh_form)
	file_out.write("dust_law %d\n" % dust_law)
	file_out.write("igm_type %d\n" % igm_type)
	file_out.write("duste_switch %d\n" % duste_switch)
	file_out.write("add_agn %d\n" % add_agn)
	file_out.write("nmodels %d\n" % nmodels)
	file_out.write("pr_logzsol_min %lf\n" % params_range1['logzsol'][0])
	file_out.write("pr_logzsol_max %lf\n" % params_range1['logzsol'][1])
	file_out.write("pr_log_tau_min %lf\n" % params_range1['log_tau'][0])
	file_out.write("pr_log_tau_max %lf\n" % params_range1['log_tau'][1])
	file_out.write("pr_log_t0_min %lf\n" % params_range1['log_t0'][0])
	file_out.write("pr_log_t0_max %lf\n" % params_range1['log_t0'][1])
	file_out.write("pr_log_alpha_min %lf\n" % params_range1['log_alpha'][0])
	file_out.write("pr_log_alpha_max %lf\n" % params_range1['log_alpha'][1])
	file_out.write("pr_log_beta_min %lf\n" % params_range1['log_beta'][0])
	file_out.write("pr_log_beta_max %lf\n" % params_range1['log_beta'][1])
	file_out.write("pr_log_age_min %lf\n" % params_range1['log_age'][0])
	file_out.write("pr_log_age_max %lf\n" % params_range1['log_age'][1])
	file_out.write("pr_dust_index_min %lf\n" % params_range1['dust_index'][0])
	file_out.write("pr_dust_index_max %lf\n" % params_range1['dust_index'][1])
	file_out.write("pr_dust1_min %lf\n" % params_range1['dust1'][0])
	file_out.write("pr_dust1_max %lf\n" % params_range1['dust1'][1])
	file_out.write("pr_dust2_min %lf\n" % params_range1['dust2'][0])
	file_out.write("pr_dust2_max %lf\n" % params_range1['dust2'][1])
	file_out.write("pr_log_gamma_min %lf\n" % params_range1['log_gamma'][0])
	file_out.write("pr_log_gamma_max %lf\n" % params_range1['log_gamma'][1])
	file_out.write("pr_log_umin_min %lf\n" % params_range1['log_umin'][0])
	file_out.write("pr_log_umin_max %lf\n" % params_range1['log_umin'][1])
	file_out.write("pr_log_qpah_min %lf\n" % params_range1['log_qpah'][0])
	file_out.write("pr_log_qpah_max %lf\n" % params_range1['log_qpah'][1])
	file_out.write("pr_log_fagn_min %lf\n" % params_range1['log_fagn'][0])
	file_out.write("pr_log_fagn_max %lf\n" % params_range1['log_fagn'][1])
	file_out.write("pr_log_tauagn_min %lf\n" % params_range1['log_tauagn'][0])
	file_out.write("pr_log_tauagn_max %lf\n" % params_range1['log_tauagn'][1])

	# cosmology
	if cosmo=='flat_LCDM' or cosmo==0:
		cosmo1 = 0
	elif cosmo=='WMAP5' or cosmo==1:
		cosmo1 = 1
	elif cosmo=='WMAP7' or cosmo==2:
		cosmo1 = 2
	elif cosmo=='WMAP9' or cosmo==3:
		cosmo1 = 3
	elif cosmo=='Planck13' or cosmo==4:
		cosmo1 = 4
	elif cosmo=='Planck15' or cosmo==5:
		cosmo1 = 5
	#elif cosmo=='Planck18' or cosmo==6:
	#	cosmo1 = 6
	else:
		print ("Input cosmo is not recognized!")
		sys.exit()
	file_out.write("cosmo %d\n" % cosmo1)

	file_out.write("H0 %lf\n" % H0)
	file_out.write("Om0 %lf\n" % Om0)
	# output files name
	if name_out_fits == None:
		name_out_fits = "random_modelSEDs.fits"
	file_out.write("name_out_fits %s\n" % name_out_fits)
	# redshift
	file_out.write("gal_z %lf\n" % gal_z)  
	file_out.close()

	os.system('mv %s %s' % (name_config,dir_file))
	os.system("mpirun -n %d python %s./save_models_photo.py %s %s" % (nproc,CODE_dir,name_filters_list,name_config))
	os.system("rm %s%s" % (dir_file,name_config))
	os.system("rm %s%s" % (dir_file,name_filters_list))

	return name_out_fits


def save_models_rest_spec(imf_type=1,sfh_form=4,dust_law=1,duste_switch=0,add_neb_emission=1,add_agn=0,gas_logu=-2.0,
	nmodels=100000,params_range={'logzsol':[-2.0,0.2],'log_tau':[-1.0,1.5],'log_age':[-2.0,1.14],'log_alpha':[-2.0,2.0],
	'log_beta':[-2.0,2.0],'log_t0':[-1.0,1.14],'dust_index':[-2.2,0.4],'dust1':[0.0,4.0], 'dust2':[0.0,4.0],
	'log_gamma':[-4.0, 0.0],'log_umin':[-1.0,1.39],'log_qpah':[-3.0,1.0],'log_fagn':[-5.0,0.48],'log_tauagn':[0.7, 2.18]},
	nproc=10,name_out=None):
	"""Function for generating a set of model spectra at rest-frame. The values of the parameters are randomly generated and for each parameter, the random values are uniformly distributed.

	:param imf_type:
		Choice for the IMF. Choices are: 0 for Salpeter(1955), 1 for Chabrier(2003), and 2 for Kroupa(2001).

	:param sfh_form:
		Choice for the parametric SFH model. Options are: 0 for exponentially declining or tau model, 1 for delayed tau model, 2 for log normal model, 3 for Gaussian form, and 4 for double power-law model.

	:param dust_law:
		Choice for the dust attenuation law. Options are: 0 for Charlot & Fall (2000) and 1 for Calzetti et al. (2000).

	:param duste_switch:
		Choice for switching on (value: 1) or off (value: 0) the dust emission modeling.

	:param add_neb_emission:
		Choice for switching on (value: 1) or off (value: 0) the nebular emission modeling.

	:param add_agn:
		Choice for turning on (value: 1) or off (value: 0) the AGN dusty torus modeling.

	:param gas_logu:
		Gas ionization parameter in logarithmic scale.

	:param nmodels:
		Number of model SEDs to be generated.

 	:param params_range:
		Ranges of parameters in a dictionary format. Summary of the parameters are tabulated in Table 1 of `Abdurro'uf et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021arXiv210109717A/abstract>`_.

	:param nproc:
		Number of cores to be used in the calculations.

	:returns name_out:
		Desired name for the output HDF5 file.
	"""

	dir_file = PIXEDFIT_HOME+'/data/temp/'
	CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_model/'

	def_params_range ={'logzsol':[-2.0,0.2],'log_tau':[-1.0,1.5],'log_age':[-1.0,1.14],'log_alpha':[-2.0,2.0],
			'log_beta':[-2.0,2.0],'log_t0':[-1.0,1.14],'dust_index':[-2.2,0.4],'dust1':[0.0,4.0], 'dust2':[0.0,4.0],
			'log_gamma':[-4.0, 0.0],'log_umin':[-1.0,1.39],'log_qpah':[-3.0,1.0],'log_fagn':[-5.0,0.48],'log_tauagn':[0.7, 2.18]}

	# get keys in input params_range:
	keys = list(params_range.keys())

	# merge with the default one
	params_range1 = def_params_range
	for ii in range(0,len(keys)):
		params_range1[keys[ii]] = params_range[keys[ii]]

	# make configuration file
	name_config = "config_file%d.dat" % (random.randint(0,10000))
	file_out = open(name_config,"w")
	file_out.write("imf_type %d\n" % imf_type)
	file_out.write("add_neb_emission %d\n" % add_neb_emission)
	file_out.write("gas_logu %lf\n" % gas_logu)
	file_out.write("sfh_form %d\n" % sfh_form)
	file_out.write("dust_law %d\n" % dust_law)
	file_out.write("duste_switch %d\n" % duste_switch)
	file_out.write("add_agn %d\n" % add_agn)
	file_out.write("nmodels %d\n" % nmodels)
	file_out.write("pr_logzsol_min %lf\n" % params_range1['logzsol'][0])
	file_out.write("pr_logzsol_max %lf\n" % params_range1['logzsol'][1])
	file_out.write("pr_log_tau_min %lf\n" % params_range1['log_tau'][0])
	file_out.write("pr_log_tau_max %lf\n" % params_range1['log_tau'][1])
	file_out.write("pr_log_t0_min %lf\n" % params_range1['log_t0'][0])
	file_out.write("pr_log_t0_max %lf\n" % params_range1['log_t0'][1])
	file_out.write("pr_log_alpha_min %lf\n" % params_range1['log_alpha'][0])
	file_out.write("pr_log_alpha_max %lf\n" % params_range1['log_alpha'][1])
	file_out.write("pr_log_beta_min %lf\n" % params_range1['log_beta'][0])
	file_out.write("pr_log_beta_max %lf\n" % params_range1['log_beta'][1])
	file_out.write("pr_log_age_min %lf\n" % params_range1['log_age'][0])
	file_out.write("pr_log_age_max %lf\n" % params_range1['log_age'][1])
	file_out.write("pr_dust_index_min %lf\n" % params_range1['dust_index'][0])
	file_out.write("pr_dust_index_max %lf\n" % params_range1['dust_index'][1])
	file_out.write("pr_dust1_min %lf\n" % params_range1['dust1'][0])
	file_out.write("pr_dust1_max %lf\n" % params_range1['dust1'][1])
	file_out.write("pr_dust2_min %lf\n" % params_range1['dust2'][0])
	file_out.write("pr_dust2_max %lf\n" % params_range1['dust2'][1])
	file_out.write("pr_log_gamma_min %lf\n" % params_range1['log_gamma'][0])
	file_out.write("pr_log_gamma_max %lf\n" % params_range1['log_gamma'][1])
	file_out.write("pr_log_umin_min %lf\n" % params_range1['log_umin'][0])
	file_out.write("pr_log_umin_max %lf\n" % params_range1['log_umin'][1])
	file_out.write("pr_log_qpah_min %lf\n" % params_range1['log_qpah'][0])
	file_out.write("pr_log_qpah_max %lf\n" % params_range1['log_qpah'][1])
	file_out.write("pr_log_fagn_min %lf\n" % params_range1['log_fagn'][0])
	file_out.write("pr_log_fagn_max %lf\n" % params_range1['log_fagn'][1])
	file_out.write("pr_log_tauagn_min %lf\n" % params_range1['log_tauagn'][0])
	file_out.write("pr_log_tauagn_max %lf\n" % params_range1['log_tauagn'][1])

	# output files name:
	if name_out == None:
		name_out = "random_model_spectra.hdf5"
	file_out.write("name_out %s\n" % name_out)  
	file_out.close()

	os.system('mv %s %s' % (name_config,dir_file))
	os.system("mpirun -n %d python %s./save_models_rest_spec.py %s" % (nproc,CODE_dir,name_config))
	os.system("rm %s%s" % (dir_file,name_config))

	return name_out


def add_fagn_bol_samplers(name_sampler_fits=None, name_out_fits=None, nproc=10):
	"""Function to add f_agn_bol into the FITS file containing MCMC samplers. 
	This parameter means a fraction of the total bolometric luminosity that is come from AGN contribution.
	This is different to the f_agn (native of FSPS and one of free parameters in SED fitting) which means 
	the ratio between AGN bolometric luminosity and the stellar bolometric luminosity.

	:param name_sampler_fits: (default: None)
		Name of the input FITS file that contains the MCMC samplers.

	:param name_out_fits: (default: None)
		Desired name for the output FITS file. If None, the name will be 'add_fagnbol_[name_sampler_fits]'.

	:param nproc: (default: 10)
		Number of cores to be used.   
	"""
	if name_sampler_fits == None:
		print ("name_sampler_fits can't be None!...")
		sys.exit()
	else:
		if name_out_fits == None:
			name_out_fits = 'add_fagnbol_%s' % name_sampler_fits

	CODE_dir = PIXEDFIT_HOME+'/piXedfit/piXedfit_model/'
	os.system("mpirun -n %d python %s./add_fagnbol.py %s %s" % (nproc,CODE_dir,name_sampler_fits,name_out_fits))

	return name_out_fits




