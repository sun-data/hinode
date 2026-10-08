Introduction
============

`Hinode <https://hinode.nao.ac.jp/en/>`_ is a JAXA solar observatory,
built with NASA, the UK Space Agency, and ESA,
which has been observing the Sun since 2006 :cite:p:`Kosugi2007`.

This Python package downloads Hinode observations and represents them
using :mod:`named_arrays`,
a named tensor implementation with :class:`astropy.units.Quantity` support.
So far it supports the Level 1 images of the
X-Ray Telescope (XRT) :cite:p:`Golub2007`, in :mod:`hinode.xrt`.

Installation
============

This package is published to PyPI and can be installed using pip.

.. code-block::

    pip install hinode

API Reference
=============

.. autosummary::
    :toctree: _autosummary
    :template: module_custom.rst
    :recursive:

    hinode


Examples
========

Load the XRT images taken through the Al_poly filter while the
EUV Snapshot Imaging Spectrograph (ESIS) was observing the Sun on
2019 September 30, and display the last one.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import astropy.units as u
    import astropy.visualization
    import named_arrays as na
    import hinode

    # Download the images and load them into memory
    images = hinode.xrt.open(
        time_start="2019-09-30T18:06:11",
        time_stop="2019-09-30T18:11:01",
        filter="Al_poly",
    )

    # Select the last image
    image = images[{images.axis_time: ~0}]

    # Display the image
    unit = image.inputs.position.x.unit
    with astropy.visualization.quantity_support():
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        na.plt.pcolormesh(
            image.inputs.position.x,
            image.inputs.position.y,
            C=image.outputs,
            ax=ax,
            cmap="gray",
            norm="asinh",
            vmin=0 * u.DN / u.s,
            vmax=100 * u.DN / u.s,
        )
        ax.set_aspect("equal")
        ax.set_title(image.inputs.time.ndarray)
        ax.set_xlabel(f"helioprojective $x$ ({unit:latex_inline})")
        ax.set_ylabel(f"helioprojective $y$ ({unit:latex_inline})")


References
==========

.. bibliography::

|


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
