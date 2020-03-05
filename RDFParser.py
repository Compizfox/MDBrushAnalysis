"""

"""

import re
from io import StringIO

import numpy as np

class RDFParser:
	"""

	"""

	def parse_rdf(self, filename: str) -> np.ndarray:
		# Use regex to strip the timestep lines
		with open(filename) as f:
			p = re.compile(r'^\d+ \d+\n', re.MULTILINE)
			string = p.sub('', f.read())

		data = np.loadtxt(StringIO(string), usecols=(0, 1, 2))

		# The data array is a '2D flattened' representation of a 3D array
		# (the third dimension being the time). We need to first get the number of
		# bins and then reshape the array using that number

		# Get the index where the second bin is 1 (that is the number of bins)
		numBins = np.nonzero(data[:, 0] == 1)[0][1]

		reshaped = data.reshape((-1, numBins, 3))

		# We're only really interested in the 2 columns containing spatial coordinates and g(r)
		return reshaped[:, :, 1:3]

	def get_ta_rdf(self, filename: str):
		"""
		Get time-averaged RDF.
		:param filename:
		:return:
		"""
		return np.mean(self.parse_rdf(filename), axis=0)
