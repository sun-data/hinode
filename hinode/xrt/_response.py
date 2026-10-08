from typing import Sequence
import typing
import numpy as np
import astropy.units as u
import astropy.time
import named_arrays as na

__all__ = [
    "temperature_response",
]


def _name_xrtpy(filter: str) -> str:
    """
    The name :mod:`xrtpy` gives to the filters an image was taken through,
    named as in :func:`hinode.xrt.urls`.

    Parameters
    ----------
    filter
        The filters, as in the ``EC_FW1_`` and ``EC_FW2_`` header keywords,
        joined by a slash, such as ``"Al_poly"`` or ``"Al_poly/Ti_poly"``.
    """
    names = {"Gband": "G-band"}
    return "/".join(names.get(f, f.replace("_", "-")) for f in filter.split("/"))


def temperature_response(
    filter: str | Sequence[str] | na.AbstractScalarArray,
    time: str | astropy.time.Time,
    abundance: str = "coronal",
    photons: bool = False,
    axis_filter: str = "filter",
    axis_temperature: str = "temperature",
) -> na.FunctionArray[na.ScalarArray, na.ScalarArray]:
    r"""
    The temperature response of XRT through the given filters on a given
    date, as :mod:`xrtpy` computes it.

    This is the response of :class:`xrtpy.response.TemperatureResponseFundamental`:
    the CHIANTI spectrum distributed with :mod:`xrtpy` folded with the
    effective area of each filter :cite:p:`Narukage2011`,
    including the contamination of the CCD and of the filters on that date.

    Parameters
    ----------
    filter
        The filters to return the response of,
        named as in :func:`hinode.xrt.urls`,
        such as ``"Al_poly"`` or ``"Al_poly/Ti_poly"``.
        If a string or a sequence of them, the filters are put along
        `axis_filter`.
    time
        The date of the observation,
        which sets the thickness of the contamination.
    abundance
        The abundances of the CHIANTI spectrum,
        ``"coronal"``, ``"hybrid"``, or ``"photospheric"``.
    photons
        If :obj:`True`, the response is the number of photons the CCD absorbs,
        rather than the signal in DN, so that the ratio of the two responses is
        the average signal of one photon from plasma at each temperature,
        which sets the photon noise of :meth:`hinode.xrt.Filtergram.uncertainty`.
    axis_filter
        The name of the axis along the filters, if `filter` does not
        already have one.
    axis_temperature
        The name of the axis along the temperatures of the result.

    Returns
    -------
    A function of temperature whose outputs are the response of each filter
    to unit emission measure, in
    :math:`\text{DN}\,\text{cm}^5\,\text{s}^{-1}\,\text{pix}^{-1}`,
    or in :math:`\text{ph}\,\text{cm}^5\,\text{s}^{-1}\,\text{pix}^{-1}`
    if `photons` is :obj:`True`.

    Notes
    -----
    The spectra distributed with :mod:`xrtpy` 0.5 were computed with CHIANTI
    10.0 at a constant electron density of :math:`10^9\,\text{cm}^{-3}`,
    at log temperatures from 5 to 8 in steps of 0.05.
    The temperature response of the AIA channels in :mod:`sdo` uses an
    older version of CHIANTI at a constant pressure,
    so a DEM found from both instruments is usually allowed a factor between
    the two responses.

    The response in photons is the sum which
    :meth:`xrtpy.response.TemperatureResponseFundamental.integration` takes
    over wavelength, without the energy of each photon, the energy needed to
    free an electron, and the gain of the CCD.

    Examples
    --------
    The response of the Al_poly filter on the date of the flight of the EUV
    Snapshot Imaging Spectrograph (ESIS),
    and the average signal of one photon at each temperature.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import hinode

        response = hinode.xrt.temperature_response("Al_poly", "2019-09-30")
        photons = hinode.xrt.temperature_response(
            filter="Al_poly",
            time="2019-09-30",
            photons=True,
        )

        index = dict(filter=0)

        with astropy.visualization.quantity_support():
            fig, axs = plt.subplots(
                nrows=2,
                sharex=True,
                constrained_layout=True,
            )
            na.plt.plot(
                response.inputs,
                response.outputs[index],
                ax=axs[0],
                axis="temperature",
            )
            na.plt.plot(
                response.inputs,
                (response.outputs / photons.outputs)[index],
                ax=axs[1],
                axis="temperature",
            )
            axs[0].set_xscale("log")
            axs[0].set_yscale("log")
            axs[0].set_ylabel(f"response ({response.outputs.unit:latex_inline})")
            axs[1].set_ylabel("signal of one photon (DN / ph)")
            axs[1].set_xlabel(f"temperature ({response.inputs.unit:latex_inline})")
    """
    # Imported here, since it takes several seconds to import
    import xrtpy.response

    if isinstance(filter, str):
        filter = [filter]
    if not isinstance(filter, na.AbstractArray):
        filter = na.ScalarArray(np.array(filter, dtype=str), axes=axis_filter)
    filter = typing.cast(na.ScalarArray, filter.explicit)
    if not filter.shape:
        filter = filter.add_axes(axis_filter)
    if filter.ndim != 1:
        raise ValueError(f"`filter` must be 0D or 1D, got {filter.shape=}")
    (axis_filter,) = filter.shape

    date = astropy.time.Time(time)
    if date.shape:
        raise ValueError(f"`time` must be a single time, got {date.shape=}")

    channels = [
        xrtpy.response.TemperatureResponseFundamental(
            filter_name=_name_xrtpy(str(name)),
            observation_date=date,
            abundance_model=abundance,
        )
        for name in np.ravel(np.asarray(filter.ndarray))
    ]

    responses = []
    for channel in channels:
        if photons:
            response = (
                channel.spectra()
                * channel.effective_area()
                * channel.solid_angle_per_pixel
                * np.gradient(channel.wavelength)
            ).sum(axis=1)
        else:
            response = channel.temperature_response()
        responses.append(na.ScalarArray(response, axes=axis_temperature))

    # Every filter has the temperatures of the same spectra
    temperature = u.Quantity(channels[0].CHIANTI_temperature)

    return na.FunctionArray(
        inputs=na.ScalarArray(temperature, axes=axis_temperature),
        outputs=typing.cast(na.ScalarArray, na.stack(responses, axis=axis_filter)),
    )
