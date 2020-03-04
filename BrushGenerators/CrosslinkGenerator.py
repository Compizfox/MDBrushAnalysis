"""
Exports CrosslinkGenerator
Written by G.C. Ritsema van Eck, liberally borrowing and butchering work by L.B. Veldscholte (Compizfox)
"""
import random
from enum import Enum
from typing import Tuple, Optional

import numpy as np

from . import BrushGenerator


class CrosslinkGenerator(BrushGenerator):
	"""
	Generate a LAMMPS data file containing a Kremer-Grest polymer brush grafted to a planar wall in a rectangular box,
	adding single-particle side chains and/or backbone heteroparticles.
	Defaults are equivalent to KremerGrestBrushGenerator. Doesn't contain a lot of checks, so silly inputs will yield
	silly and unphysical data files. You have been warned.

	Kremer, K.; Grest, G. S. Dynamics of Entangled Linear Polymer Melts: A Molecular‐dynamics Simulation. J. Chem. Phys.
	1990, 92 (8), 5057–5086. https://doi.org/10.1063/1.458541.
	"""
	AtomTypes = Enum('AtomTypes', ['graft', 'bead', 'solvent', 'hetero', 'branch'])
	BondTypes = Enum('BondTypes', ['fene'])

	masses = {
		AtomTypes.graft:   1,
		AtomTypes.bead:    1,
		AtomTypes.solvent: 1,
		AtomTypes.hetero:  1,
		AtomTypes.branch:  1,
	}

	def __init__(self, box_size: Tuple[float, float, float], rng_seed: Optional[int], side_freq: float = 0.0,
	             het_freq: float = 0.0):
		"""
		:param Tuple box_size:  3-tuple of floats describing the dimensions of the rectangular box.
		:param int   rng_seed:  Seed used to initialize the PRNG. May be None, in which case a random seed will be used.
		:param float side_freq: Chance of a given backbone particle having a side group attached to it, mutually
		                        exclusive with being a heteroparticle.
		:param float het_freq:  Chance of a given backbone particle being a heteroparticle, mutually exclusive with
		                        having a side group.
		"""
		bead_size = 1       # (sigma)
		bottom_padding = 1  # (sigma)

		self.het_freq = het_freq
		self.het_chance = self.het_freq
		self.side_freq = side_freq
		self.side_chance = side_freq / (1 - self.het_chance)

		super().__init__(box_size, rng_seed, bead_size, bottom_padding)

	def _build_bead(self, mol_id: int, graft_coord: np.ndarray, bead_id: int) -> None:
		if bead_id == 0:
			atom_type = self.AtomTypes.graft.value
		elif random.random() < self.het_chance:
			atom_type = self.AtomTypes.hetero.value
		else:
			atom_type = self.AtomTypes.bead.value

		if atom_type == self.AtomTypes.bead.value and random.random() < self.side_chance:
			self._atoms_list.append({'mol_id':    mol_id,
			                         'atom_type': self.AtomTypes.branch.value,
			                         'q':         0,
			                         'x':         graft_coord[0],
			                         'y':         graft_coord[1] + 1,
			                         'z':         float(bead_id * self.bead_size)
			                         })
			self._atoms_list.append({'mol_id':    mol_id,
			                         'atom_type': atom_type,
			                         'q':         0,
			                         'x':         graft_coord[0],
			                         'y':         graft_coord[1],
			                         'z':         float(bead_id * self.bead_size)
			                         })

			atom_id = len(self._atoms_list)

			self._bonds_list.append({'bond_type': self.BondTypes.fene.value,
			                         'atom1':     atom_id - 1,
			                         'atom2':     atom_id
			                         })
			self._bonds_list.append({'bond_type': self.BondTypes.fene.value,
			                         'atom1':     atom_id - 2,
			                         'atom2':     atom_id
			                         })
		else:
			self._atoms_list.append({'mol_id':    mol_id,
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
