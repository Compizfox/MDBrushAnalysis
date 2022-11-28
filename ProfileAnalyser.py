"""
Exports the ProfileAnalyser class.
"""
import os
import pickle
from enum import Enum

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from BrushDensityParser import BrushDensityParser


class ProfileAnalyser:
	"""
	Analyses density profiles to extract sorption behaviour.
	"""

	# Defaults:
	FILENAME_DENS_POLY   = 'PolyDens.dat'
	FILENAME_DENS_SOLV   = 'SolvDens.dat'
	INTERP_FACTOR        = 10
	TA_TRIM_FRACTION     = 0.95
	SG_WINDOW            = 9
	SG_ORDER             = 2
	POLY_END_TRIM        = 25
	VAPOUR_LOC_TRIM      = 5
	VAPOUR_LOC_THRESHOLD = 0.002

	LOCS: Enum = Enum('Locations', ['ab', 'ad'])
	
	READ_CACHE = True
	SAVE_CACHE = True
		
	def __init__(self, directory: str, filename_poly: str = FILENAME_DENS_POLY,
	             filename_solvent: str = FILENAME_DENS_SOLV, interp_factor: int = INTERP_FACTOR,
	             ta_trim_frac: float = TA_TRIM_FRACTION, sg_window: int = SG_WINDOW, sg_order: int = SG_ORDER,
	             pe_trim: int = POLY_END_TRIM, vl_trim: int = VAPOUR_LOC_TRIM,
	             vl_threshold: float = VAPOUR_LOC_THRESHOLD, read_cache = READ_CACHE, cache_profile = SAVE_CACHE) -> None:
		"""
		:param directory:        Path to the base directory containing the files.
		:param filename_poly:    Filename of the polymer density file.
		:param filename_solvent: Filename of the solvent density file.
		:param interp_factor:    Number of times to spatially interpolate density profiles before time averaging
		:param ta_trim_frac:     Fraction of temporal frames to discard at the beginning
		:param sg_window:        Window size of the Savitsky-Golay filter
		:param sg_order:         Order of the polynomial fitted by the Savitsky-Golay filter
		:param pe_trim:          Number of spatial chunks to discard from start for determining outer brush end
		:param vl_trim:          Number of spatial chunks to discard from end for determining vapour location
		:param vl_threshold:     Threshold in the gradient of the solvent density above which to consider the
		                         vapour phase ending (and the adsorption layer starting). Should be higher than the
		                         fluctuations in the vapour phase.
		:param read_cache:       Boolean, whether or not to read existing cached data
		:param cache_profile:    Boolean, whether or not to cache processed data as pickle
		"""
		self.sg_window = (sg_window * interp_factor) // 2 * 2 + 1  # Round to nearest odd integer
		self.sg_order = sg_order
		self.pe_trim = pe_trim * interp_factor
		self.vl_trim = vl_trim * interp_factor
		self.vl_threshold = vl_threshold / interp_factor

		self._process(directory, filename_poly, filename_solvent, interp_factor, ta_trim_frac, read_cache, cache_profile)

	def _process(self, directory: str, filename_poly: str, filename_solvent: str, interp_factor: int,
	             ta_trim_frac: float, read_cache, cache_profile) -> None:
		"""
		Parses data using BrushDensityParser, and trims, spatially interpolates and time-averages it.
		Implements simple file caching of the aggregated data using pickle.
		:param directory:        Path to the base directory containing the files.
		:param filename_poly:    Filename of the polymer density file.
		:param filename_solvent: Filename of the solvent density file.
		:param interp_factor:    Number of times to spatially interpolate density profiles before time averaging
		:param ta_trim_frac:     Fraction of temporal frames to discard at the beginning
		:param read_cache:       Boolean, whether or not to read existing cached data
		:param cache_profile:    Boolean, whether or not to cache processed data as pickle
		"""
		cachefile = directory + f'/pa_cache.pickle'
		if os.path.exists(cachefile) and read_cache:
			with open(cachefile, 'rb') as cachehandle:
				print("Using cached result from {}".format(cachefile))
				self.poly_ta, self.solv_ta = pickle.load(cachehandle)
		else:
			bdp = BrushDensityParser()

			dens_poly = bdp.load_density(directory + '/' + filename_poly)
			dens_solv = bdp.load_density(directory + '/' + filename_solvent)

			# Slice for trimming unequilibrated first temporal chunks from time average
			num_frames = len(dens_poly)
			s = np.s_[int(num_frames * ta_trim_frac):, :, :]

			# Interpolate in space
			dens_poly_f = interp1d(dens_poly[0, :, 1], dens_poly[s], axis=1, kind='cubic')
			dens_solv_f = interp1d(dens_solv[0, :, 1], dens_solv[s], axis=1, kind='cubic')
			x = np.linspace(dens_poly[0, 0, 1], dens_poly[0, -1, 1], int(dens_poly.shape[1] * interp_factor))

			# time-averaged profiles
			self.poly_ta: np.ndarray = np.mean(dens_poly_f(x), axis=0)
			self.solv_ta: np.ndarray = np.mean(dens_solv_f(x), axis=0)
			
			if cache_profile:
				with open(cachefile, 'wb') as cachehandle:
					pickle.dump((self.poly_ta, self.solv_ta), cachehandle)
			
	def _get_vapour_location(self) -> int:
		"""
		Find the point where the solvent adsorption layer stops and the vapour phase begins.
		:return: Index of the vapour location
		"""
		# Return index of first occurence of the (absolute value of the) solvent density gradient crossing a threshold
		# approaching from the right
		# (argmax is a non-intuitive, but the best way to do this)
		gradient = np.gradient(self.solv_ta[:-self.vl_trim, 2])
		i = np.argmax(np.abs(gradient[::-1]) > self.vl_threshold)
		# We inverted the profile to start the search from the right, so we have to subtract i from the max index
		return self.solv_ta[:-self.vl_trim].shape[0] - i - 1

	def _get_profile_slice(self, loc: LOCS) -> slice:
		"""
		Fabricate slice objects corresponding to the ab- and adsorption ranges, depending on the value of loc.
		:param loc: Location of the integral (ad- or ab-sorption), instance of the LOCS enum
		:return: slice
		"""
		if loc == self.LOCS.ab:
			return np.s_[:self.get_poly_inflection()]
		if loc == self.LOCS.ad:
			return np.s_[self.get_poly_inflection():self._get_vapour_location()]

		raise TypeError("loc must be an instance of LOCS")

	def get_poly_inflection(self) -> int:
		"""
		Find the inflection point of the polymer density profile by calculating the gradient using a Savitsky-Golay
		filter and getting the index of the minimum element in that array.
		:return: Index of the inflection point
		"""
		# Smooth using Savitzky–Golay
		poly_ta_smoothed = savgol_filter(self.poly_ta[:, 2], self.sg_window, self.sg_order, deriv=1)
		# Inflection point is minimum of first derivative
		return poly_ta_smoothed[self.pe_trim:].argmin() + self.pe_trim

	def get_poly_end(self) -> int:
		"""
		Find the absolute end of the brush
		:return: Index of the end of the brush
		"""
		# Smooth using Savitzky–Golay
		poly_ta_smoothed = savgol_filter(self.poly_ta[:, 2], self.sg_window, self.sg_order, deriv=2)
		# Maximum curvature point is maximum of second derivative
		return poly_ta_smoothed[self.pe_trim:].argmax() + self.pe_trim

	def get_solv_area(self, loc: LOCS) -> float:
		"""
		Calculate the integral of the solvent density profile in (absorption) or on (adsorption) the brush, depending on
		the value of loc.
		:param loc: Location of the integral (ad- or ab-sorption), instance of the LOCS enum
		:return: Float corresponding to the area
		"""
		profile_slice = self._get_profile_slice(loc)

		return np.trapz(self.solv_ta[profile_slice][:, 2], self.solv_ta[profile_slice][:, 1])

	def get_solv_fraction(self, loc: LOCS) -> float:
		"""
		Calculate the fraction of sorbed solvent particles in (absorption) or on (adsorption) the brush, depending on
		the value of loc.
		:param loc: Location of the integral (ad- or ab-sorption), instance of the LOCS enum
		:return: Float corresponding to the solvent fraction
		"""
		profile_slice = self._get_profile_slice(loc)

		solv_area = self.get_solv_area(loc)
		poly_area = np.trapz(self.poly_ta[profile_slice][:, 2], self.poly_ta[profile_slice][:, 1])

		return solv_area / (solv_area + poly_area)
