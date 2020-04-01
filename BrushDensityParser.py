"""
Exports the AveChunkParser and BrushDensityParser classes.
"""

import re
from io import StringIO
from typing import Sequence, Optional

import numpy as np
import pandas as pd


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
