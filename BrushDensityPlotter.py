import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import savgol_filter

from BrushDensityParser import BrushDensityParser


class BrushDensityPlotter:
	def __init__(self, directory, filenamePoly='PolyDens.txt', filenameParticles='ParticleDens.txt'):
		self.dir = directory

		bdp = BrushDensityParser()

		self.densPoly = bdp.load_density(directory + filenamePoly)
		self.densPart = bdp.load_density(directory + filenameParticles)

	@staticmethod
	def smooth(densityArray, windowTime, windowSpace):
		# Smooth data using Savinsky-Golay filter

		if windowTime != 0:
			densityArray = savgol_filter(densityArray, windowTime, 2, axis=0)  # in time
		if windowSpace != 0:
			densityArray = savgol_filter(densityArray, windowSpace, 2, axis=1)  # in space

		return densityArray

	@staticmethod
	def normalise(densityArray):
		densityArray[:, :, 2] = densityArray[:, :, 2] / np.sum(densityArray[:, :, 2], axis=1)[:, None]
		return densityArray

	def animatedPlot(self, smooth=True, normalise=True, save=False, interval=0):
		# Build animated plot

		# Smooth data if wanted
		if smooth:
			densPoly_s = self.smooth(self.densPoly, 3, 0)
			densPart_s = self.smooth(self.densPart, 11, 11)
		else:
			densPoly_s = self.densPoly
			densPart_s = self.densPart

		# Normalise data
		if normalise:
			densPoly_s = self.normalise(densPoly_s)
			densPart_s = self.normalise(densPart_s)

		# Calculate total density profile
		densTotal = densPoly_s[:, :, 2] + densPart_s[:, :, 2]

		# Gather some sizes/maximums/sums in several dimensions
		maxT = densPoly_s.shape[0]
		maxX = densPoly_s[0, -1, 1]
		#maxD = max((densPoly_s[:, :, 2].max(), densPart_s[:, :, 2].max(), densTotal_s[:, :, 2].max()))
		maxD = densTotal.max()

		fig = plt.figure()

		def init():
			ax = plt.axes(xlim=(0, maxX), ylim=(0, maxD))

			l1, = ax.plot([], [], color='tab:blue')
			l2, = ax.plot([], [], color='tab:orange')
			l3, = ax.plot([], [], color='tab:green')
			plt.xlabel('Distance ($\sigma$)')
			plt.ylabel('Density ($\sigma^{-3}$)')
			plt.grid()
			l1.set_data([], [])
			l2.set_data([], [])
			l3.set_data([], [])

			return [l1, l2, l3]

		l1, l2, l3 = init()

		def animate(t):
			l1.set_data(densPoly_s[t, :, 1], densPoly_s[t, :, 2])
			l2.set_data(densPart_s[t, :, 1], densPart_s[t, :, 2])
			l3.set_data(densPart_s[t, :, 1], densTotal[t])
			return [l1, l2, l3]

		anim = animation.FuncAnimation(fig, animate, init_func=init, frames=maxT, interval=interval, blit=True)

		# Save plot
		if save:
			anim.save(self.dir + 'mpl_out.mp4', fps=30, bitrate=5000)
