# hinode

[![tests](https://github.com/sun-data/hinode/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/hinode/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sun-data/hinode/graph/badge.svg)](https://codecov.io/gh/sun-data/hinode)
[![Black](https://github.com/sun-data/hinode/actions/workflows/black.yml/badge.svg)](https://github.com/sun-data/hinode/actions/workflows/black.yml)
[![Ruff](https://github.com/sun-data/hinode/actions/workflows/ruff.yml/badge.svg)](https://github.com/sun-data/hinode/actions/workflows/ruff.yml)
[![typing](https://github.com/sun-data/hinode/actions/workflows/typing.yml/badge.svg)](https://github.com/sun-data/hinode/actions/workflows/typing.yml)
[![Documentation Status](https://readthedocs.org/projects/hinode/badge/?version=latest)](https://hinode.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/hinode.svg)](https://badge.fury.io/py/hinode)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23249959.svg)](https://doi.org/10.5281/zenodo.23249959)

A Python library to download and analyze observations from the
JAXA/NASA [Hinode](https://hinode.nao.ac.jp/en/) satellite.

So far it supports the Level 1 images of the X-Ray Telescope (XRT),
which it finds in the archive of the Solar Data Analysis Center,
sorts by filter, and downloads.
The images are not represented as instances of
[`sunpy.map.Map`](https://docs.sunpy.org/en/stable/generated/api/sunpy.map.Map.html),
but as instances of [`named_arrays.FunctionArray`](https://named-arrays.readthedocs.io/en/latest/_autosummary/named_arrays.FunctionArray.html),
which carry the time, exposure time and sky position of every pixel,
along with a mask of the pixels affected by saturation.
It also computes what a differential emission measure needs:
the temperature response of each filter, using [xrtpy](https://xrtpy.readthedocs.io),
the visible light leaking into each pixel, and the uncertainty of each pixel.

## Installation

This package is published on PyPI and can be installed using pip
```bash
pip install hinode
```

## Example

Load the XRT images taken through the Al_poly filter between two times
```python
import hinode

images = hinode.xrt.open(
    time_start="2019-09-30T18:06:11",
    time_stop="2019-09-30T18:11:01",
    filter="Al_poly",
)
```

## Citation

If you use hinode in your research, please cite it.
The citation metadata is kept in [`CITATION.cff`](https://github.com/sun-data/hinode/blob/main/CITATION.cff),
which the "Cite this repository" button on GitHub can export as BibTeX or APA.

Every release of hinode is archived on Zenodo with its own DOI.
The concept DOI, [10.5281/zenodo.23249959](https://doi.org/10.5281/zenodo.23249959),
always resolves to the latest version,
and the Zenodo page lists the DOI of every version.
Please include the version of hinode that you used,
which is given by `importlib.metadata.version("hinode")`.
The BibTeX entry below uses the concept DOI.
To cite a specific version instead,
replace `doi` with the DOI of that version.

```bibtex
@software{hinode,
  author = {Smart, Roy T.},
  title = {hinode},
  version = {X.Y.Z},
  doi = {10.5281/zenodo.23249959},
  url = {https://github.com/sun-data/hinode},
}
```
