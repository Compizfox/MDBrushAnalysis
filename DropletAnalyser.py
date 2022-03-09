"""
Exports the DropletAnalyser class.
"""
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq
from scipy.signal import savgol_filter


class DropletAnalyser:
	"""
	Analyses 2D density profiles of a droplet simulation to extract the contact angle.
	"""

	# Defaults:
	BLUR_KERNEL_SIZE = (15, 15)
	CIRCLEFIT_BOTTOM_TRIM = 10

	def __init__(self, dens_solv: np.ndarray, dens_poly: np.ndarray, center_droplet: bool = True,
	             blur_kernel_size: Tuple[int, int] = BLUR_KERNEL_SIZE,
	             circlefit_bottom_trim: int = CIRCLEFIT_BOTTOM_TRIM) -> None:
		"""
		Process pixmaps and compute the contact angle.
		:param dens_solv: Pixmap (2D ndarray in column-major order (x, z)) corresponding to solvent density
		:param dens_poly: Pixmap (2D ndarray of column-major order (x, z)) corresponding to polymer density
		:param center_droplet: Whether to center the droplet in the box (over x) before analysis
		:param blur_kernel_size: Gaussian kernel size for blurring the solvent density pixmap. Must be positive and odd.
		:param circlefit_bottom_trim: Number of pixels in z (from the baseline) to exclude from the circle fit
		"""
		self.blur_kernel_size = blur_kernel_size
		self.circlefit_bottom_trim = circlefit_bottom_trim

		self.dens_poly = dens_poly
		if center_droplet:
			self.dens_solv = self._center_droplet(dens_solv)
		else:
			self.dens_solv = dens_solv

		# Normalize and blur solvent density pixmap (helps with everything in computer vision)
		img = cv2.normalize(src=self.dens_solv, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
		self.blur = cv2.GaussianBlur(img, self.blur_kernel_size, 0)
		# Transform pixmap to a bitmap containing edges
		self.edges = cv2.Canny(self.blur, 0, 100)
		# Detect baseline and fit circle
		self.bl = self._get_baseline(dens_poly, dens_solv)
		self.C, self.R = self._fit_circle(self.edges)

	def _get_baseline(self, dens_poly: np.ndarray, dens_solv: np.ndarray) -> int:
		"""
		Extract baseline (substrate z-height) from polymer density profile by finding the inflection point of the
		polymer density profile next to the droplet
		:param dens_poly: Pixmap corresponding to polymer density
		:return: Z-index of the baseline
		"""
		# Find x-indices where there is no droplet
		img = cv2.normalize(src=np.array(dens_solv), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
		blur = cv2.GaussianBlur(img, self.blur_kernel_size, 0)
		edges = cv2.Canny(blur, 0, 100)
		xs = np.where(np.amax(edges, axis=1) == 0)[0]
		
		# Obtain 1D z profile by averaging over x
		dens_poly_z = np.mean(np.array(dens_poly)[xs], axis=0)

		# Smooth and differentiate using Savitzky–Golay
		trim = 25
		dens_poly_z_smooth = savgol_filter(dens_poly_z, 9, 2, deriv=1)

		# Inflection point is minimum of first derivative
		return dens_poly_z_smooth[trim:].argmin() + trim

	def _fit_circle(self, edges: np.ndarray) -> Tuple[np.ndarray, float]:
		"""
		Fit a circle to droplet edges using least-squares optimisation and a implicitly-defined circle in calc_R().
		:param edges: Pixmap with droplet edges
		:return: Center and radius respectively of the fitted circle
		"""
		# We have a bitmap containing the droplet edges. We want to get the x and z indices of the edge points (pixels
		# that are 1) and put them in the 1D arrays xs and zs.
		xs, zs = np.nonzero(edges)

		# We don't want to include the bottom part of the droplet edge, so we define some threshold in z
		thresh = self.bl + self.circlefit_bottom_trim
		if not np.any(zs > thresh):
			# No points above the threshold: there is no droplet.
			raise TypeError('No droplet detected.')

		def calc_R(c: np.ndarray) -> np.ndarray:
			"""
			Given a center c, calculate the distance of every point to the center.
			Ignores points below a z-threshold `thresh`.
			:param c: Center coordinates
			:return: Array of distance of every point to the center
			"""
			return np.sqrt((xs[zs > thresh] - c[0])**2 + (zs[zs > thresh] - c[1])**2)

		def f(c: np.ndarray) -> np.ndarray:
			"""
			Calculate the distance between the data points and the mean circle centered at c=(xc, yc)
			:param c: Center coordinates
			:return: Array of distance of every point to the circle
			"""
			Ri = calc_R(c)
			return Ri - Ri.mean()

		# Use the point centered in x and at z=0 as the initial guess
		center_estimate = np.array([edges.shape[1]/2, 0])
		# Fit the circle
		c, _ = leastsq(f, center_estimate)
		r = calc_R(c).mean()

		self.xs = xs
		self.zs = zs

		return c, r

	def _get_deltas(self) -> Tuple[float, float]:
		"""
		Calculate and return two distances: the half distance between the contact points, and the z-distance between
		circle center and baseline.
		:return: dx, dz
		"""
		dz = self.C[1] - self.bl  # Z-distance between circle center and baseline
		dx = np.sqrt(self.R**2 - dz**2)  # X-distance between contact point and center projected on baseline = half
		#                                  distance between contact points
		return dx, dz

	@staticmethod
	def _center_droplet(dens_solv: np.ndarray) -> np.ndarray:
		"""
		Center the droplet in x by rolling the x axis.
		:param dens_solv: Pixmap corresponding to solvent density
		:return: Pixmap corresponding to solvent density, with the droplet centered
		"""
		# Obtain 1D x profile by averaging over z
		dens_solv_x = np.mean(dens_solv, axis=1)
		x_size = len(dens_solv_x)

		# Determine the center of mass of the droplet in the first half (in x) of the box
		half = int(x_size / 2)
		dens_solv_x_half = np.array(dens_solv_x)[:half]
		x_droplet_com_half = np.sum(dens_solv_x_half*np.arange(0, half))/np.sum(dens_solv_x_half)

		# If the droplet CoM is closer to the edge than to the center, roll the pixmap over by half it size
		if x_droplet_com_half < half / 2:
			dens_solv = np.roll(dens_solv, half, axis=0)
			dens_solv_x = np.mean(dens_solv, axis=1)

		# Determine center of mass of droplet which is the first moment of the profile divided by the total integral
		x_droplet_com = np.sum(dens_solv_x*np.arange(0, x_size))/np.sum(dens_solv_x)

		# Difference between axis center and droplet CoM
		shift_diff = x_size/2 - x_droplet_com

		# Roll the pixmap over the x-axis by the shift amount
		return np.roll(dens_solv, round(shift_diff), axis=0)

	def get_contact_angle(self) -> float:
		"""
		Calculate the contact angle.
		:return: Contact angle (in radians)
		"""
		dx, dz = self._get_deltas()
		return np.pi/2 + np.arctan(dz/dx)

	def plot(self, ax: plt.Axes, debug: bool = False):
		"""
		Plot the solvent density with the fitted circle and contact angles using Matplotlib.
		:param ax: Axes to draw in
		:param debug: Whether to plot debugging info (edges and trim line)
		"""
		ax.imshow(self.blur.T, origin='lower', aspect='equal')

		if debug:
			ax.plot(self.xs, self.zs, '.', color='tab:purple')
			ax.axhline(y=self.bl + self.circlefit_bottom_trim, color='black', alpha=0.5)

		circle = plt.Circle(self.C, self.R, color='r', fill=False)
		ax.add_artist(circle)
		ax.axhline(y=self.bl, color='black')

		dx, _ = self._get_deltas()
		x_con = [self.C[0] - dx, self.C[0] + dx]
		theta = self.get_contact_angle()
		l = dx/2

		ax.plot([x_con[0], x_con[0] - np.cos(np.pi - theta)*l],
		        [self.bl, self.bl + np.sin(np.pi - theta)*l], color='tab:orange')
		ax.plot([x_con[1], x_con[1] + np.cos(np.pi - theta)*l],
		        [self.bl, self.bl + np.sin(np.pi - theta)*l], color='tab:orange')

		ax.axis('equal')
		ax.set_title(f"Contact angle: {np.rad2deg(theta):.2f} deg")
