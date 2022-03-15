"""
Exports the LAMMPSDataParser class and Dims enum.
"""

import re
from enum import Enum
from io import StringIO
from typing import List, Sequence, Tuple, TextIO

import numpy as np
import pandas as pd

from SmartOpen import SmartOpen


class Dims(Enum):
	x = 0
	y = 1
	z = 2


class AtomStyle:
	def __init__(self, columns):
		self.columns: dict = columns

	def get_names(self):
		return [*self.columns]

	def get_dtypes(self):
		return [*self.columns.items()]


atom_styles = {
	'atomic': AtomStyle(
		{'id': int, 'type': int, 'x': float, 'y': float, 'z': float, 'nx': int, 'ny': int, 'nz': int}),
	'bond': AtomStyle(
		{'id': int, 'mol': int, 'type': int, 'x': float, 'y': float, 'z': float, 'nx': int, 'ny': int, 'nz': int}),
	'molecular': AtomStyle(
		{'id': int, 'mol': int, 'type': int, 'x': float, 'y': float, 'z': float, 'nx': int, 'ny': int, 'nz': int}),
	'full': AtomStyle(
		{'id': int, 'mol': int, 'type': int, 'q': float, 'x': float, 'y': float, 'z': float, 'nx': int, 'ny': int,
		 'nz': int}),
}


class LAMMPSDataParser:
	"""
	Parses the position data of atoms from LAMMPS data files.
	See https://lammps.sandia.gov/doc/write_data.html
	"""

	def __init__(self, filename: str) -> None:
		"""
		Parses box dimensions and atom position data from LAMMPS data file and loads it into a Pandas dataframe.
		:param filename: Path to the LAMMPS data file.
		"""
		data_strings = []

		with SmartOpen(filename) as f:
			box_dims = self._extract_box_dimensions(f)
			self.atom_style = self._extract_atom_style(f)

			# Copy lines containing position data
			for line in f:
				if line.strip() == "Velocities":
					break
				elif line.strip() == "":
					continue
				data_strings.append(line)

		# Compute box sizes by taking difference of lo and hi values
		self.box_sizes: np.ndarray = box_dims.ptp(axis=0)

		# Put position data in Pandas dataframe
		self._data: pd.DataFrame = pd.read_csv(StringIO(''.join(data_strings)), sep=' ', header=None, index_col=0,
		                                       engine='c', names=atom_styles[self.atom_style].get_names(),
		                                       dtype=atom_styles[self.atom_style].get_dtypes())

		# Subtract the lower box coordinates from all atom coordinates, i.e. put the origin (0, 0, 0) at the
		# front-bottom-left corner so position coordinates go from 0 to box_size
		for dim in Dims:
			self._data[dim.name] -= box_dims[0, dim.value]

	@staticmethod
	def _extract_box_dimensions(f: TextIO) -> np.ndarray:
		"""
		Capture the box dimensions (2 values x 3 dimensions) using regex.
		:param f: File object
		:return: Array of shape (2, 3)
		"""
		box_dims = np.empty((2, 3))
		for dim in Dims:
			for line in f:
				p = re.compile(
					rf'([-+]?\d*\.?\d*[eE]?[+-]?\d*) ([-+]?\d*\.?\d*[eE]?[+-]?\d*) {dim.name}lo {dim.name}hi')
				match = p.search(line)
				if match:
					# Found line, stop search for this dimension
					box_dims[:, dim.value] = match.group(1, 2)
					break
		return box_dims

	@staticmethod
	def _extract_atom_style(f: TextIO) -> str:
		"""
		Capture the atom style using regex. This also sets the file pointer's position to the the line after the
		`Atom ...` line.
		:param f: File object
		:return: Atom style
		"""
		for line in f:
			p = re.compile(r'Atoms # (\w+)')
			match = p.search(line)
			if match:
				return match.group(1)
		raise TypeError('Atom style not found')

	def get_positions_by_type(self, atom_types: Sequence[int]) -> np.ndarray:
		"""
		Selects atoms by type and returns the positions.
		:param atom_types: List of integers corresponding to the atom types that should be included.
		:return: A 2D ndarray with shape (N, 3) where N is the number of atoms and the 3
		columns corresponds to the X, Y, and Z coordinates.
		"""
		return self._data.loc[self._data['type'].isin(atom_types), ['x', 'y', 'z']].to_numpy()

	def get_density_profile(self, atom_types: Sequence[int], dimensions: Sequence[str], resolution: float) \
			-> Tuple[np.ndarray, List[np.ndarray]]:
		"""
		Computes a N-dimensional density from the atom position data.
		:param atom_types: List of integers corresponding to the atom types that should be included in the total
		                   density.
		:param dimensions: List of strings corresponding to the dimensions over which the density should be profiled.
		                   The resulting density will be averaged over the remaining densities. For example,
		                   when dimensions = ['z'], a 1D density profile in Z will be obtained, averaged over X and Y.
		:param resolution: The linear resolution (in units of distance) of the density profile.
		:return (density, bin_locations): A N-dimensional ndarray containing the density and a list of N 1D ndarrays
		        describing the axes (spatial bin locations) for every dimension.
		"""
		# Get dimension ids from names
		dim_ids = [Dims[dim].value for dim in dimensions]
		# Calculate number of bins and spatial ranges for every dimension
		n_bins = [self.box_sizes[dim_num]//resolution for dim_num in dim_ids]
		ranges = [(0, self.box_sizes[dim_num]) for dim_num in dim_ids]

		# Get atom positions and convert to density by taking a histogram
		atom_positions = self.get_positions_by_type(atom_types)[:, dim_ids]
		(density, bins_edges) = np.histogramdd(atom_positions, bins=n_bins, range=ranges)

		# Convert the N+1 bin edges to N midpoint values
		bin_locations = [bin_edges[1:] - resolution/2 for bin_edges in bins_edges]

		# Calculate the (3D) bin volume and divide the density array by it to normalise
		mean_dim_ids = np.setdiff1d([dim.value for dim in Dims], dim_ids, assume_unique=True)
		bin_volume = resolution**len(dimensions)*np.prod([self.box_sizes[dim_num] for dim_num in mean_dim_ids])
		density /= bin_volume

		return density, bin_locations
