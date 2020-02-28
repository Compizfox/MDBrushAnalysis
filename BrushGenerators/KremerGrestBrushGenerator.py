"""
Exports the KremerGrestBrushGenerator class
"""
from typing import Tuple, Optional
from enum import Enum

import numpy as np

from . import BrushGenerator


class KremerGrestBrushGenerator(BrushGenerator):
	"""
	Generate a LAMMPS data file containing a Kremer-Grest polymer brush grafted to a planar wall in a rectangular box.

	Kremer, K.; Grest, G. S. Dynamics of Entangled Linear Polymer Melts: A Molecular‐dynamics Simulation. J. Chem. Phys.
	1990, 92 (8), 5057–5086. https://doi.org/10.1063/1.458541.
	"""
	AtomTypes = Enum('AtomTypes', ['graft', 'bead', 'solvent'])
	BondTypes = Enum('BondTypes', ['fene'])

	masses = {
		AtomTypes.graft: 1,
		AtomTypes.bead:  1,
		AtomTypes.solvent: 1
	}

	def __init__(self, box_size: Tuple[float, float, float], rng_seed: Optional[int]):
		"""
		:param Tuple box_size:  3-tuple of floats describing the dimensions of the rectangular box.
		:param int   rng_seed:  Seed used to initialize the PRNG. May be None, in which case a random seed will be used.
		"""
		bead_size = 1       # (sigma)
		bottom_padding = 1  # (sigma)
		super().__init__(box_size, rng_seed, bead_size, bottom_padding)

	def _build_bead(self, mol_id: int, graft_coord: np.ndarray, bead_id: int) -> None:
		if bead_id == 0:
			atom_type = self.AtomTypes.graft.value
		else:
			atom_type = self.AtomTypes.bead.value

		self._atoms_list.append({'mol_id':    bead_id + 1,
		                         'atom_type': atom_type,
		                         'q':         0,
		                         'x':         graft_coord[0],
		                         'y':         graft_coord[1],
		                         'z':         float(bead_id * self.bead_size)
		                         })

		# Molecular topology
		if bead_id >= 1:
			atom_id = len(self._atoms_list)
			self._bonds_list.append({'bond_type': self.BondTypes.fene.value,
			                         'atom1':     atom_id - 1,
			                         'atom2':     atom_id
			                         })
