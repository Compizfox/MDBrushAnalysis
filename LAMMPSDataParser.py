"""
Exports the LAMMPSDataParser class and Dims enum.
"""

import re
from enum import Enum
from io import StringIO
from typing import List, Sequence, Tuple, TextIO

import numpy as np
import pandas as pd


class Dims(Enum):
	x = 0
	y = 1
	z = 2


class AtomType:
	def __init__(self, columns):
		self.columns: dict = columns

	def get_names(self):
		return [*self.columns]

	def get_dtypes(self):
		return [*self.columns.items()]


atom_types = {
	'atomic': AtomType(
		{'id': int, 'type': int, 'x': float, 'y': float, 'z': float, 'nx': int, 'ny': int, 'nz': int}),
	'bond': AtomType(
		{'id': int, 'mol': int, 'type': int, 'x': float, 'y': float, 'z': float, 'nx': int, 'ny': int, 'nz': int}),
	'molecular': AtomType(
		{'id': int, 'mol': int, 'type': int, 'x': float, 'y': float, 'z': float, 'nx': int, 'ny': int, 'nz': int}),
	'full': AtomType(
		{'id': int, 'mol': int, 'type': int, 'q': float, 'x': float, 'y': float, 'z': float, 'nx': int, 'ny': int,
		 'nz': int}),
}


class LAMMPSDataParser:
	"""
	Parses the position data of atoms from LAMMPS data files.
	See https://lammps.sandia.gov/doc/write_data.html
	"""

	def __init__(self, filename: str):
		"""
		Parses box dimensions and atom position data from LAMMPS data file and loads it into a Pandas dataframe.
		:param str filename: Path to the LAMMPS data file.
		"""
		data_string = ""
		box_dims: np.ndarray = np.empty((2, 3))
		with open(filename) as f:
			# Extract box dimensions
			for dim in Dims:
				for line in f:
					p = re.compile(
						rf'([-+]?\d*\.?\d*[eE][+-]\d*) ([-+]?\d*\.?\d*[eE][+-]\d*) {dim.name}lo {dim.name}hi')
					match = p.search(line)
					if match:
						# Found line, stop search for this dimension
						box_dims[:, dim.value] = match.group(1, 2)
						break

			# Extract atom type
			self.atom_type = self._extract_atom_type(f)

			# Copy lines between the lines in the file delimiting position data
			copy = False
			for line in f:
				# Beginning of position data
				if line.strip() == f"Atoms # {self.atom_type}":
					# Skip first line
					copy = True
					continue
				# End of position data
				elif line.strip() == "Velocities":
					break
				elif copy:
					if line.strip() == "":
						continue
					data_string += line

		# Compute box sizes by taking difference of lo and hi values
		self._box_sizes: np.ndarray = box_dims.ptp(axis=0)

		# Put position data in Pandas dataframe
		self._data: pd.DataFrame = pd.read_csv(StringIO(data_string), sep=' ', header=None, index_col=0, engine='c',
		                                       names=atom_types[self.atom_type].get_names(),
		                                       dtype=atom_types[self.atom_type].get_dtypes())

		# Subtract the lower box coordinates from all atom coordinates, i.e. put the origin (0, 0, 0) at the
		# front-bottom-left corner so position coordinates go from 0 to box_size
		for dim in Dims:
			self._data[dim.name] -= box_dims[0, dim.value]

	def _extract_atom_type(self, f: StringIO):
		for line in f:
			p = re.compile(r'Atoms # (\w+)')
			match = p.search(line)
			if match:
				f.seek(0)
				return match.group(1)

		raise TypeError('Atom type not found')

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
		n_bins = [self._box_sizes[dim_num]//resolution for dim_num in dim_ids]
		ranges = [(0, self._box_sizes[dim_num]) for dim_num in dim_ids]

		# Get atom positions and convert to density by taking a histogram
		atom_positions = self.get_positions_by_type(atom_types)[:, dim_ids]
		(density, bins_edges) = np.histogramdd(atom_positions, bins=n_bins, range=ranges)

		# Convert the N+1 bin edges to N midpoint values
		bin_locations = [bin_edges[1:] - resolution/2 for bin_edges in bins_edges]

		# Calculate the (3D) bin volume and divide the density array by it to normalise
		mean_dim_ids = np.setdiff1d([dim.value for dim in Dims], dim_ids, assume_unique=True)
		bin_volume = resolution**len(dimensions)*np.prod([self._box_sizes[dim_num] for dim_num in mean_dim_ids])
		density /= bin_volume

		return density, bin_locations
