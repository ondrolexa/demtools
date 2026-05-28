"""FFT-based gradient and upward-continuation operators.

Adapted from Fatiando a Terra.
"""

import numpy as np


def _nextpow2(i):
    """Return the smallest power of two >= ``i``.

    Args:
        i (int): Input value.

    Returns:
        int: Smallest power of two >= ``i``.
    """
    buf = np.ceil(np.log(i) / np.log(2))
    return int(2**buf)


def _pad_data(data, shape):
    """Pad *data* symmetrically to the next power of two.

    Uses edge-value replication for padding.

    Args:
        data (numpy.ndarray): 2-D array to pad.
        shape (tuple): Original ``(ny, nx)`` shape (unused, kept for signature compat).

    Returns:
        tuple: ``(padded, padx, pady)`` where *padded* is the zero-padded array
        and ``(padx, pady)`` are the number of pixels added on each side.
    """
    n = _nextpow2(np.max(shape))
    nx, ny = shape
    padx = (n - nx) // 2
    pady = (n - ny) // 2
    padded = np.pad(data, ((padx, padx), (pady, pady)), mode="edge")
    return padded, padx, pady


def _fftfreqs(dx, dy, padshape):
    """Return 2-D wave-number arrays for the x and y directions.

    Args:
        dx (float): Pixel size in the x-direction.
        dy (float): Pixel size in the y-direction.
        padshape (tuple): ``(height, width)`` of the padded array.

    Returns:
        tuple: ``(kx, ky)`` 2-D arrays of angular wave numbers
        :math:`(\\text{rad}/m)`.
    """
    fx = 2 * np.pi * np.fft.fftfreq(padshape[0], dx)
    fy = 2 * np.pi * np.fft.fftfreq(padshape[1], dy)
    return np.meshgrid(fy, fx)[::-1]


def derivx(data, dx):
    """Compute the first horizontal derivative in the x-direction.

    Uses a central finite-difference scheme:
    :math:`\\frac{\\partial f}{\\partial x} \\approx
    \\frac{f(x+\\Delta x) - f(x-\\Delta x)}{2\\Delta x}`.

    Edge rows are mirrored from the adjacent interior row.

    Args:
        data (numpy.ndarray): 2-D array of values.
        dx (float): Pixel spacing in the x-direction.

    Returns:
        numpy.ndarray: Horizontal derivative (same shape as *data*).
    """
    deriv = np.empty_like(data)
    deriv[1:-1, :] = (data[2:, :] - data[:-2, :]) / (2 * dx)
    deriv[0, :] = deriv[1, :]
    deriv[-1, :] = deriv[-2, :]
    return deriv


def derivy(data, dy):
    """Compute the first horizontal derivative in the y-direction.

    Uses a central finite-difference scheme:
    :math:`\\frac{\\partial f}{\\partial y} \\approx
    \\frac{f(y+\\Delta y) - f(y-\\Delta y)}{2\\Delta y}`.

    Edge columns are mirrored from the adjacent interior column.

    Args:
        data (numpy.ndarray): 2-D array of values.
        dy (float): Pixel spacing in the y-direction.

    Returns:
        numpy.ndarray: Horizontal derivative (same shape as *data*).
    """
    deriv = np.empty_like(data)
    deriv[:, 1:-1] = (data[:, 2:] - data[:, :-2]) / (2 * dy)
    deriv[:, 0] = deriv[:, 1]
    deriv[:, -1] = deriv[:, -2]
    return deriv


def derivz(data, dx, dy):
    """Compute the first vertical derivative via FFT.

    The vertical derivative is obtained from the potential-field relation
    :math:`\\frac{\\partial f}{\\partial z} =
    \\mathcal{F}^{-1}\\left[|\\mathbf{k}| \\,
    \\mathcal{F}(f)\\right]`.

    The array is padded to the next power of two before the FFT to avoid
    wraparound artefacts.

    Args:
        data (numpy.ndarray): 2-D array of values.
        dx (float): Pixel size in the x-direction.
        dy (float): Pixel size in the y-direction.

    Returns:
        numpy.ndarray: Vertical derivative (same shape as *data*).
    """
    nx, ny = data.shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, data.shape)
    kx, ky = _fftfreqs(dx, dy, padded.shape)
    deriv_ft = np.fft.fft2(padded) * np.sqrt(kx**2 + ky**2)
    deriv = np.real(np.fft.ifft2(deriv_ft))
    # Remove padding from derivative
    return deriv[padx : padx + nx, pady : pady + ny]


def upcontinue(data, dx, dy, height):
    """Upward continue a potential-field grid.

    The upward-continued field at height *h* is:
    :math:`f_h = \\mathcal{F}^{-1}\\left[e^{-h|\\mathbf{k}|}
    \\,\\mathcal{F}(f)\\right]`.

    The array is padded to the next power of two before the FFT to avoid
    wraparound artefacts.

    Args:
        data (numpy.ndarray): 2-D array of values.
        dx (float): Pixel size in the x-direction.
        dy (float): Pixel size in the y-direction.
        height (float): Height (in the same units as *dx* / *dy*) to
            continue the field to.

    Returns:
        numpy.ndarray: Upward-continued field (same shape as *data*).
    """
    nx, ny = data.shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, data.shape)
    kx, ky = _fftfreqs(dx, dy, padded.shape)
    kz = np.sqrt(kx**2 + ky**2)
    upcont_ft = np.fft.fft2(padded) * np.exp(-height * kz)
    cont = np.real(np.fft.ifft2(upcont_ft))
    # Remove padding
    return cont[padx : padx + nx, pady : pady + ny]
