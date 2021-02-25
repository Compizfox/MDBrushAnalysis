"""
Exports the SmartOpen class.
"""

from typing import TextIO


class SmartOpen:
	"""
	Open a file with the right data compression module (gzip, bzip2, lzma, or plain) based on the file extension.
	Is a context manager.
	"""
	def __init__(self, filename: str, mode: str = 'rt') -> None:
		"""
		:param filename: Path to the file.
		:param mode:     Mode string ('rt' by default)
		"""
		if filename.endswith('.gz'):
			import gzip
			o = gzip.open
		elif filename.endswith('.bz2'):
			import bz2
			o = bz2.open
		elif filename.endswith('.xz'):
			import lzma
			o = lzma.open
		else:
			o = open

		self.file_obj = o(filename, mode)

	def __enter__(self) -> TextIO:
		return self.file_obj

	def __exit__(self, exc_type, exc_value, traceback):
		self.file_obj.close()
