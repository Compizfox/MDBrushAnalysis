"""
Exports the ProfileAnalyser class.
"""

import numpy as np
from scipy.signal import savgol_filter

from BrushDensityParser import BrushDensityParser


class ProfileAnalyser:
	"""
	Analyses density profiles to extract sorption states.
	0: No sorption
	1: Adsorption, no absorption
	2: Adsorption,    absorption
	"""
	# Defaults:
	FILENAME_DENS_POLY        = 'PolyDens.dat'  # Filename of the file containing polymer density data
	FILENAME_DENS_SOLV        = 'SolvDens.dat'  # Filename of the file containing solvent density data
	TA_TRIM_FRACTION          = 0.5             # Fraction of temporal frames to discard at the beginning
	SG_WINDOW                 = 21              # Window size of the Savitzky-Golay filter
	SG_ORDER                  = 2               # Order of the polynomial fitted by the Savitsky-Golay filter
	T_CONFIDENCE              = 0.95            # Confidence level to compute in the confidence intervals
	POLY_END_TRIM             = 25              # Number of spatial chunks to discard for determining outer brush end
	VAPOUR_GRADIENT_THRESHOLD = 0.0015          # Threshold in the gradient of the solvent density above which to
	#                                             consider the vapour phase ending (and the adsorption layer starting).
	#                                             Should be higher than the fluctuations in the vapour phase.
	VAPOUR_TOP_TRIM           = 10              # Number of spatial chunks to discard at the top/end of the vapour
	#                                             region for determining vapour location

	def __init__(self, directory: str, filename_poly: str = FILENAME_DENS_POLY,
	             filename_solvent: str = FILENAME_DENS_SOLV, ta_trim_frac: int = TA_TRIM_FRACTION):
		"""
		:param directory: String containing the path to the base directory containing the files.
		:param filename_poly: String containing the filename of the polymer density file.
		:param filename_solvent: String containing the filename of the solvent density file.
		:param ta_trim: Number of temporal chunks (profiles) to discard at the beginning
		"""
		self.directory: str = directory
		bdp = BrushDensityParser()

		self.dens_poly = bdp.load_density(directory + '/' + filename_poly)
		self.dens_solv = bdp.load_density(directory + '/' + filename_solvent)

		# Slice for trimming unequilibrated first temporal chunks from time average
		num_frames = len(self.dens_poly)
		s = np.s_[int(num_frames*ta_trim_frac):, :, :]

		# time-averaged profiles
		self.poly_ta: np.ndarray = np.mean(self.dens_poly[s], axis=0)
		self.solv_ta: np.ndarray = np.mean(self.dens_solv[s], axis=0)

	def get_poly_inflection(self, window: int = SG_WINDOW, order: int = SG_ORDER) -> int:
		"""
		Finds the inflection point of the polymer density profile by calculating the gradient using a Savitsky-Golay
		filter and getting the index of the minimum element in that array.
		:param window: Window size of the Savitsky-Golay filter
		:param order: Order of the polynomial fitted by the Savitsky-Golay filter
		:return: Index of the inflection point
		"""
		# Smooth using Savitzky–Golay
		poly_ta_smoothed = savgol_filter(self.poly_ta[:, 2], window, order, deriv=1)
		# Inflection point is minimum of first derivative
		return poly_ta_smoothed.argmin()

	def get_poly_end(self, window: int = SG_WINDOW, order: int = SG_ORDER, trim: int = POLY_END_TRIM) -> int:
		"""
		Finds the absolute end of the brush
		:return: Index of the end of the brush
		"""
		# Smooth using Savitzky–Golay
		poly_ta_smoothed = savgol_filter(self.poly_ta[:, 2], window, order, deriv=2)
		# Maximum curvature point is maximum of second derivative
		return poly_ta_smoothed[trim:].argmax() + trim

	def _get_solv_area(self, profile_range: slice) -> float:
		"""
		Integrates the given slice of the solvent density profile
		:param profile_range: (Spatial) slice of the density profile
		:return: Float corresponding to the area
		"""
		return np.trapz(self.solv_ta[profile_range][:, 2], self.solv_ta[profile_range][:, 1])

	def _get_vapour_location(self, threshold: float = VAPOUR_GRADIENT_THRESHOLD, trim: int = VAPOUR_TOP_TRIM) -> int:
		"""
		Finds the point where the solvent adsorption layer stops and the vapour phase begins.
		:param float threshold: Threshold in the gradient of the solvent density above which to consider the vapour
		                        phase ending (and the adsorption layer starting). Should be higher than the
		                        fluctuations in the vapour phase.
		:return: Index of the vapour location
		"""
		# Return index of first occurence of the (absolute value of the) solvent density gradient crossing a threshold
		# approaching from the right
		# (argmax is a non-intuitive, but the best way to do this)
		gradient = np.gradient(self.solv_ta[:-trim, 2])
		i = np.argmax(np.abs(gradient[::-1]) > threshold)
		# We inverted the profile to start the search from the right, so we have to subtract i from the max index
		return self.solv_ta[:-trim].shape[0] - i - 1

	def _get_profile_range_in(self, window: int = SG_WINDOW, order: int = SG_ORDER) -> slice:
		return np.s_[:self.get_poly_inflection(window, order)]

	def _get_profile_range_out(self, window: int = SG_WINDOW, order: int = SG_ORDER,
	                          threshold: float = VAPOUR_GRADIENT_THRESHOLD) -> slice:
		return np.s_[self.get_poly_inflection(window, order):self._get_vapour_location(threshold)]

	def get_solv_area_in(self, window: int = SG_WINDOW, order: int = SG_ORDER) -> float:
		"""
		Calculate the integral of the solvent density profile inside the brush.
		:return: Float corresponding to the area
		"""
		return self._get_solv_area(self._get_profile_range_in(window, order))

	def get_solv_area_out(self, window: int = SG_WINDOW, order: int = SG_ORDER,
	                      threshold: float = VAPOUR_GRADIENT_THRESHOLD) -> float:
		"""
		Calculate the integral of the solvent density profile outside the brush.
		:return:Float corresponding to the area
		"""
		return self._get_solv_area(self._get_profile_range_out(window, order, threshold))

	def get_time_resolved_area(self, window: int = SG_WINDOW, order: int = SG_ORDER,
	                           threshold: float = VAPOUR_GRADIENT_THRESHOLD) -> np.ndarray:
		"""
		Calculate the integral of the solvent density profiles as a function of time.
		:return: 2d ndarray of shape (3, n) corresponding density inside the brush, outside the brush, and total (
		respectively) over time.
		"""
		slice_in = self._get_profile_range_in(window, order)
		slice_out = self._get_profile_range_out(window, order, threshold)

		area_in = np.trapz(self.dens_solv[:, slice_in, 2], self.dens_solv[:, slice_in, 1], axis=1)
		area_out = np.trapz(self.dens_solv[:, slice_out, 2], self.dens_solv[:, slice_out, 1], axis=1)
		area_total = area_in + area_out

		return np.stack([area_in, area_out, area_total])
