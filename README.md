# MDBrushAnalysis
![GitHub license](https://img.shields.io/github/license/Compizfox/MDBrushAnalysis)
![Python version](https://img.shields.io/badge/Python-%3E3.6-orange)
![GitHub last commit](https://img.shields.io/github/last-commit/Compizfox/MDBrushAnalysis)
[![DOI](https://zenodo.org/badge/216565077.svg)](https://zenodo.org/badge/latestdoi/216565077)

This repository contains a library of Python classes for parsing LAMMPS density profiles and data files and specifically
for post-processing of MD simulations of polymer brushes in equilibrium with solvent vapours.

## Installation
Simply clone the repository using Git:

```console
foo@bar:~$ git clone https://github.com/Compizfox/MDBrushAnalysis.git
```

or download a release and extract it. The Git approach has the advantage that updating is as easy as `git pull`.

### Dependencies
MDBrushAnalysis requires at least Python 3.6. The only external libraries required are Numpy and Pandas, which are
installable using your OS's package manager or using Pip:

```console
foo@bar:~$ pip install numpy
foo@bar:~$ pip install pandas
```

## Overview
### BrushDensityParser
`AveChunkParser` defines a generic class for parsing data output by LAMMPS' ave/chunk fix. The method
`get_reshaped_data()` provides a way to reshape the long format temporal frames to a new dimension. `BrushDensityParser`
is a subclass of the former, specifically for density profiles.

### LAMMPSDataParser
`LAMMPSDataParser` defines a class that can parse LAMMPS data files. Using the method `get_positions_by_type()`,
particles of a certain type can be selected and their coordinates returned. `get_density_profile()` computes
N-dimensional density profiles by binning.

### ProfileAnalyser
`ProfileAnalyser` defines a class for the post-processing of density profiles output by MD simulations of polymer
brushes in equilibrium with solvent vapours. Density profiles of polymer and solvent are loaded using
`BrushDensityParser` (described above). Unequilibrated temporal frames are trimmed from the beginning before temporal
averaging and spatial interpolation. Temporally-averaged data is cached in files using pickle.

The class contains various methods for determining spatial limits of interest in the system and integrating density
profiles to compute the amount and fraction of sorbed solvent.

## License
This project is free software licensed under the GPL. See [LICENSE](LICENSE) for details.

