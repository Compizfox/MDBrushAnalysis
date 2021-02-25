"""
Exports the AveChunkParser and BrushDensityParser classes.
"""

import re
from io import StringIO
from typing import Dict, Optional, Sequence, Type

import numpy as np
import pandas as pd

from SmartOpen import SmartOpen


class AveChunkParser:
	"""
	Parses profiles (data outputted by LAMMPS' ave/chunk or ave/time (in vector mode) fixes).
	"""

	@staticmethod
	def _load_file(filename: str, dtype: Dict[str, Type[np.generic]], cols: Optional[Sequence[int]] = None) -> np.ndarray:
		"""
		Load a ave/chunk file.
		:param str filename:  Path to the file.
		:param dict dtype:    Dict of columns with data types.
		:param Sequence cols: Sequence of column indices (0-indexed) to use
		:return: 2D ndarray of shape (rows, cols)
		"""

		# Use regex to strip the timestep lines
		with SmartOpen(filename) as f:
			p = re.compile(r'\n\d.*')
			string = p.sub('', f.read())

		return pd.read_csv(StringIO(string), sep=' ', header=None, engine='c', comment='#', skipinitialspace=True,
		                   usecols=cols, dtype=dtype).to_numpy()

	@classmethod
	def get_reshaped_data(cls, filename: str, dtype: Dict[str, Type[np.generic]], cols: Optional[Sequence[int]] = None) -> np.ndarray:
		"""
		Load a ave/chunk file and reshape the stacked / long format data, creating a new temporal dimension.
		:param str filename:  Path to the file.
		:param dict dtype:    Dict of columns with data types.
		:param Sequence cols: Sequence of column indices (0-indexed) to use
		:return: 3D ndarray of shape (time, space, cols)
		"""
		data = cls._load_file(filename, dtype, cols)

		# The data array is a '2D flattened' representation of a 3D array
		# (the third dimension being the time). We need to first get the number of
		# chunks and then reshape the array using that number

		# Get the index where the second chunk is 1 (that is the number of chunks)
		num_chunks = np.nonzero(data[:, 0] == 1)[0][1]

		return data.reshape((-1, num_chunks, data.shape[1]))


class BrushDensityParser(AveChunkParser):
	"""
	Parses a LAMMPS ave/chunk output file.
	"""

	@classmethod
	def load_density(cls, filename: str) -> np.ndarray:
		"""
		Load a 1D density profile.
		:param str filename: Path to the density file to parse.
		:return: A 3d ndarray with shape (a, b, 3), representing density profiles at different timesteps with a being
				 the number of temporal frames and b being the number of spatial chunks in a profile. One row is a
				 tuple of (chunk #, spatial distance, density).
		"""
		return cls.get_reshaped_data(filename, {'chunk': np.uint, 'x': np.double, 'dens': np.double}, cols=(0, 1, 3))

	@classmethod
	def load_density_2d(cls, filename: str):
		"""
		Load a 2D density profile.
		:param str filename: Path to the density file to parse.
		:return: List of DataFrames, each DataFrame corresponding to a frame in time
		"""
		# Reshape time dimension of data.
		data = cls.get_reshaped_data(filename, {'chunk': np.uint, 'x': np.double, 'y': np.double, 'dens': np.double},
		                             cols=(0, 1, 2, 4))

		# The two spatial dimensions are now still stacked / in long format (one row per pixel). We first transform
		# the ndarray's first (time) dimension to a list of frames using list(). We can then convert each separate
		# frame to a DataFrame and use its pivot() function to reshape the long format data to a DataFrame indexed by
		# the 2 spatial axes.

		#                    1st column holds index
		return [pd.DataFrame(index=e[:, 0], data=e[:, 1:], columns=('x', 'y', 'dens'))
			        .pivot(index='x', columns='y', values='dens') for e in list(data)]
