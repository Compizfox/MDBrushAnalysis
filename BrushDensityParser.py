"""
Exports the AveChunkParser and BrushDensityParser classes.
"""

import re
from io import StringIO
from typing import Sequence, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class AveChunkParser:
	"""
	Parses data outputted by LAMMPS' ave/chunk or ave/time (in vector mode) fixes.
	"""
	@staticmethod
	def _load_file(filename: str, cols: Optional[Sequence]) -> np.ndarray:
		"""
		:param cols:
		:param filename:
		:return:
		"""
		# Use regex to strip the timestep lines
		with open(filename) as f:
			p = re.compile(r'\n\d.*')
			string = p.sub('', f.read())

		return pd.read_csv(StringIO(string), sep=' ', header=None, engine='c', comment='#', skipinitialspace=True,
		                   usecols=cols).to_numpy()

	@classmethod
	def get_reshaped_data(cls, filename: str, cols: Optional[Sequence]) -> np.ndarray:
		"""

		:param filename:
		:param cols:
		:return:
		"""
		data = cls._load_file(filename, cols)

		# The data array is a '2D flattened' representation of a 3D array
		# (the third dimension being the time). We need to first get the number of
		# chunks and then reshape the array using that number

		# Get the index where the second chunk is 1 (that is the number of chunks)
		num_chunks = np.nonzero(data[:, 0] == 1)[0][1]

		return data.reshape((-1, num_chunks, len(cols)))


class BrushDensityParser(AveChunkParser):
	"""
	Parses a LAMMPS ave/chunk output file.
	"""

	@classmethod
	def load_density(cls, filename: str) -> np.ndarray:
		"""
		:param filename: String containing the filename of the density file to parse.
		:return: A 3d ndarray with shape (a, b, 3), representing density profiles at different timesteps with a being
				 the number of temporal profiles and b being the number of spatial chunks in a profile. One row is a
				 tuple of (chunk #, spatial distance, density).
		"""
		return cls.get_reshaped_data(filename, (0, 1, 3))


class ProfilePlotter:
	"""
	Plots a profile (arbitrary quantity vs distance)
	"""
	def __init__(self, profile: np.ndarray, trim: int = None):
		self.profile = profile
		self.TRIM = (trim or 15)
		self.profile_ta = np.mean(self.profile[self.TRIM:, :, :], axis=0)

	def plot_ta(self, ax):
		"""
		Plot time-averaged profile.
		:param ax:
		:return:
		"""
		ax.plot(self.profile_ta[:, 1], self.profile_ta[:, 2])
		ax.set_xlabel('Distance ($\sigma$)')
		ax.set_ylim(0)
		ax.grid()

	def plot_animation(self, fig, quantity: str = 'Y'):
		"""

		:param fig:
		:param quantity:
		:return:
		"""
		# Gather some sizes/maximums/sums in several dimensions
		maxT = self.profile.shape[0]        # Maximum time
		maxX = self.profile[0, -1, 1]       # Maximum distance
		maxY = self.profile[:, :, 2].max()  # Maximum value

		def init():
			ax = plt.axes(xlim=(0, maxX), ylim=(0, maxY))

			l1, = ax.plot([], [])
			plt.xlabel('Distance ($\sigma$)')
			plt.ylabel(quantity)
			plt.grid()
			l1.set_data([], [])

			return [l1]

		[l1] = init()

		def animate(t):
			l1.set_data(self.profile[t, :, 1], self.profile[t, :, 2])
			return [l1]

		anim = animation.FuncAnimation(fig, animate, init_func=init, frames=maxT, interval=100, blit=True)

		# Save plot
		anim.save(directory + quantity + '.mp4', fps=30, bitrate=5000)
