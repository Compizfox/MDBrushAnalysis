"""
Exports the KremerGrestBrushGenerator class
"""

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

	def _build_bead(self, mol_id: int, graft_coord: np.ndarray, bead_id: int) -> None:
		if bead_id == 0:
			atom_type = self.AtomTypes.graft.value
		else:
			atom_type = self.AtomTypes.bead.value

		self._atoms_list.append({'mol_id':    mol_id + 1,
		                         'atom_type': atom_type,
		                         'q':         0,
		                         'x':         graft_coord[0],
		                         'y':         graft_coord[1],
		                         'z':         float(bead_id) + 1
		                         })

		# Molecular topology
		if bead_id >= 1:
			atom_id = len(self._atoms_list)
			self._bonds_list.append({'bond_type': self.BondTypes.fene.value,
			                         'atom1':     atom_id - 1,
			                         'atom2':     atom_id
			                         })
