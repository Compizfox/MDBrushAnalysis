"""
Exports the PoissonDiskGenerator class.
"""

from itertools import product
from typing import Tuple, Optional

import numpy as np


class PoissonDiskGenerator:
	"""
	Generate a Poisson-disk point set using dart throwing accelerated by square cells.
	"""

	def __init__(self, seed: Optional[int]):
		"""
		:param int seed: Seed used to initialize the PRNG. May be None, in which case a random seed will be used.
		"""
		self.rng = np.random.RandomState(seed=seed)

		# Create a list of 20 neighbors (5x5 excluding corners and center)
		self.neighbor_matrix = []
		for i in [-2, -1, 0, 1, 2]:
			for j in [-2, -1, 0, 1, 2]:
				if not ((abs(i) == 2 and abs(j) == 2) or (i == 0 and j == 0)):
					self.neighbor_matrix.append((i, j))

	def generate(self, n: int, bead_size: float, size: Tuple[float, float], max_iter: int = 1000) -> np.ndarray:
		"""
		:param int   n:         Number of points.
		:param float bead_size: Minimum distance between points.
		:param Tuple size:      2-Tuple corresponding to the (2d) domain size.
		:param int   max_iter:  Iteration limit.
		:return: Ndarray of shape (n, 2) with all point coordinates.
		"""

		def check_overlap(cell: np.ndarray, point: np.ndarray) -> bool:
			"""
			Check for overlap in the 8 cells neighboring the current cell.
			:param np.ndarray cell:  Ndarray of shape (2) containing cell indices.
			:param np.ndarray point: Ndarray of shape (2) containing the point coordinates.
			:return bool:            Return True if point overlaps, False if not.
			"""
			for i, j in self.neighbor_matrix:
				x = cell[0] + i
				y = cell[1] + j
				if 0 <= x <= (x_max - 1) and 0 <= y <= (x_max - 1):
					coord = grid_lookup[x][y]
					if coord is not None:
						dist = (point - coord) ** 2
						if dist[0] + dist[1] < bead_size:
							return True

			return False

		def check_oob(point) -> bool:
			"""
			Check if the point is out-of-bounds, which can happen because the domain the algorithm runs on is
			slightly larger than the requested domain because of rasterising errors.
			:param  np.ndarray point: Ndarray of shape (2) containing the point coordinates.
			:return bool:             Return True if point is out-of-bounds, False if not.
			"""
			return (point > np.array(size)).all()

		# Create a grid of square background cells. The cells should be as large as possible but while still being fully
		# covered by the size of a point within it.
		cell_size = bead_size / np.sqrt(2)

		# Round desired domain size up to nearest multiple of cell size
		x_max = int(np.ceil(size[0] / cell_size))
		y_max = int(np.ceil(size[1] / cell_size))
		# List of indices of cells. Coordinate of bottom left corner = (i*cell_size, j*cell_size)
		active_cells = np.array(list(product(range(x_max), range(y_max))))

		# x*y empty list
		grid_lookup = [[None] * y_max for i in range(x_max)]

		coordinates = []
		for count in range(0, max_iter):
			# Choose random active cell
			if len(active_cells) == 0:
				break
			cell_id = self.rng.randint(low=0, high=len(active_cells))

			# Throw a dart
			rnd = self.rng.random_sample(2)
			x = active_cells[cell_id, 0] + rnd[0]
			y = active_cells[cell_id, 1] + rnd[1]
			point = np.array([x, y]) * cell_size

			# Check if point overlaps in neighbouring cells
			if not check_overlap(active_cells[cell_id, :], point) and not check_oob(point):
				coordinates.append(point)
				grid_lookup[active_cells[cell_id, 0]][active_cells[cell_id, 1]] = point
				active_cells = np.delete(active_cells, cell_id, axis=0)

			if len(coordinates) >= n:
				break

		return np.array(coordinates)
