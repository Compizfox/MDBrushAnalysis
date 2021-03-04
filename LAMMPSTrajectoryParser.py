"""
Exports the LAMMPSTrajectoryParser class.
"""

import re
from io import StringIO

import pandas as pd

from SmartOpen import SmartOpen


class LAMMPSTrajectoryParser:
	"""
	Parses the time-resolved position data of atoms from LAMMPS dump files (trajectories).
	See https://lammps.sandia.gov/doc/dump.html
	"""
	def __init__(self, filename: str):
		"""
		Parses atom position data from LAMMPS dump file and loads it into a Pandas dataframe.
		:param str filename: Path to the LAMMPS dump file.
		"""
		position_data = {}
		self.columns = []

		with SmartOpen(filename) as f:
			copy = False
			t = 0
			for line in f:
				if line.strip() == 'ITEM: TIMESTEP':
					# Found frame header
					copy = False
					# Extract timestamp
					t = int(next(f).strip())
					position_data[t] = []
					continue

				prefix = 'ITEM: ATOMS'
				if line.startswith(prefix):
					if not self.columns:
						# Extract columns
						p = re.compile(r'(\w+)+\s?')
						self.columns = p.findall(line[len(prefix):])

					# Found position data header
					copy = True
					# Skip first line
					continue

				if copy:
					position_data[t].append(line)

		dfs = [pd.read_csv(StringIO(''.join(data_string)), sep=' ', header=None, index_col=0, engine='c',
		                   names=self.columns)
		       for data_string in position_data.values()]

		self._data = pd.concat(dfs, keys=position_data.keys(), names=['time', 'id'], axis=0)
