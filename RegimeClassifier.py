"""
Exports the RegimeClassifier class.
"""

from typing import Sequence, Tuple

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import sem, t

from BrushDensityParser import BrushDensityParser


class RegimeClassifier:
	"""
	Analyses density profiles and radial distribution functions to classify the system in one of three sorption states.
	0: No sorption
	1: Adsorption, no absorption
	2: Adsorption,    absorption
	"""
	# Defaults:
	FILENAME_DENS_POLY        = 'PolyDens.dat'  # Filename of the file containing polymer density data
	FILENAME_DENS_SOLV        = 'SolvDens.dat'  # Filename of the file containing solvent density data
	TA_TRIM                   = 20              # Number of temporal frames (profiles) to discard at the beginning
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
	             filename_solvent: str = FILENAME_DENS_SOLV, ta_trim: int = TA_TRIM):
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
		s = np.s_[ta_trim:, :, :]

		# time-averaged profiles
		self.poly_ta: np.ndarray = np.mean(self.dens_poly[s], axis=0)
		self.solv_ta: np.ndarray = np.mean(self.dens_solv[s], axis=0)

		# And confidence intervals
		self.poly_ci: np.ndarray = self._get_error(self.dens_poly[s][:, :, 2])
		self.solv_ci: np.ndarray = self._get_error(self.dens_solv[s][:, :, 2])

	@classmethod
	def _get_ci(cls, data: np.ndarray, confidence: float = T_CONFIDENCE) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Calculates confidence intervals using a Student T-distribution with given confidence from an array of data.
		:param data: Ndarray containing the sample data.
		:param confidence: Confidence level
		:return: Tuple of two ndarrays containing the (absolute) lower and upper confidence boundaries.
		"""
		# Number of samples
		N = data.shape[0]

		return t.interval(confidence, N - 1, loc=np.mean(data, axis=0), scale=sem(data, axis=0))

	@classmethod
	def _get_error(cls, data: np.ndarray, confidence: float = T_CONFIDENCE) -> np.ndarray:
		"""
		Calculates confidence intervals using _get_ci() and converts them to (symmetric) relative confidence levels.
		:param data: Ndarray containing the sample data.
		:param confidence: Confidence level
		:return: Ndarray containing confidence level(s)
		"""
		return cls._get_ci(data, confidence)[1] - np.mean(data, axis=0)

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

	def _get_area(self, profile_range: slice) -> Tuple[float, float]:
		"""
		Integrates the given slice of the solvent density profile
		:param profile_range: (Spatial) slice of the density profile
		:return: Tuple of (area, error) corresponding to the area and its respective confidence level
		"""
		area = np.trapz(self.solv_ta[profile_range][:, 2], self.solv_ta[profile_range][:, 1])
		# Propagate error to numerical integral: geometric mean of error in the profile
		error = np.nansum(self.solv_ci[profile_range] ** 2) ** (1/2)
		return area, error

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

	def get_solv_area_in(self, window: int = SG_WINDOW, order: int = SG_ORDER) -> Tuple[float, float]:
		"""
		Calculate the integral of the solvent density profile inside the brush.
		:return: Tuple of (area, error) corresponding to the area and its respective confidence level
		"""
		profile_range = np.s_[:self.get_poly_inflection(window, order)]
		return self._get_area(profile_range)

	def get_solv_area_out(self, window: int = SG_WINDOW, order: int = SG_ORDER,
	                      threshold: float = VAPOUR_GRADIENT_THRESHOLD) -> Tuple[float, float]:
		"""
		Calculate the integral of the solvent density profile outside the brush.
		:return: Tuple of (area, error) corresponding to the area and its respective confidence level
		"""
		profile_range = np.s_[self.get_poly_inflection(window, order):self._get_vapour_location(threshold)]
		return self._get_area(profile_range)

	def get_classification(self, in_threshold: int = 15, out_threshold: int = 4, window: int = SG_WINDOW,
	                       order: int = SG_ORDER) -> int:
		"""
		Get the regime classification the system is in, determined by the amount of integrated adsorbed and adsorbed
		solvent.
		:param in_threshold: Threshold for absorbed solvent above which system will be classified as 2
		:param out_threshold: Threshold for adsorbed solvent above which system will be classified as 1
		:param window: see get_poly_inflection()
		:param order: see get_poly_inflection()
		:return: Integer corresponding to classification (0, 1, or 2)
		"""
		if self.get_solv_area_in(window, order)[0] > in_threshold:
			# Sorbed solvent in brush > threshold, so we have absorption
			return 2
		if self.get_solv_area_out(window, order)[0] > out_threshold:
			# Sorbed solvent in brush > threshold, so we have adsorption
			return 1
		return 0

	def get_time_resolved_area(self) -> np.ndarray:
		"""
		Calculate the integral of the total solvent density profile as a function of time.
		:return: ndarray of total density over time.
		"""
		return np.trapz(self.dens_solv[:, :, 2], self.dens_solv[:, :, 1], axis=1)
