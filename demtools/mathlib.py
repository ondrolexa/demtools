# codes adopted from Fatiando a Terra
import numpy as np


def _pad_data(data, shape):
    n = _nextpow2(np.max(shape))
    nx, ny = shape
    padx = (n - nx) // 2
    pady = (n - ny) // 2
    padded = np.pad(data, ((padx, padx), (pady, pady)), mode="edge")
    return padded, padx, pady


def _nextpow2(i):
    buf = np.ceil(np.log(i) / np.log(2))
    return int(2**buf)


def _fftfreqs(dx, dy, shape, padshape):
    """
    Get two 2D-arrays with the wave numbers in the x and y directions.
    """
    nx, ny = shape
    fx = 2 * np.pi * np.fft.fftfreq(padshape[0], dx)
    fy = 2 * np.pi * np.fft.fftfreq(padshape[1], dy)
    return np.meshgrid(fy, fx)[::-1]


def derivx(data, dx):
    deriv = np.empty_like(data)
    deriv[1:-1, :] = (data[2:, :] - data[:-2, :]) / (2 * dx)
    deriv[0, :] = deriv[1, :]
    deriv[-1, :] = deriv[-2, :]
    return deriv


def derivy(data, dy):
    deriv = np.empty_like(data)
    deriv[:, 1:-1] = (data[:, 2:] - data[:, :-2]) / (2 * dy)
    deriv[:, 0] = deriv[:, 1]
    deriv[:, -1] = deriv[:, -2]
    return deriv


def derivz(data, dx, dy):
    nx, ny = data.shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, data.shape)
    kx, ky = _fftfreqs(dx, dy, data.shape, padded.shape)
    deriv_ft = np.fft.fft2(padded) * np.sqrt(kx**2 + ky**2)
    deriv = np.real(np.fft.ifft2(deriv_ft))
    # Remove padding from derivative
    return deriv[padx : padx + nx, pady : pady + ny]


def upcontinue(data, dx, dy, height):
    nx, ny = data.shape
    # Pad the array with the edge values to avoid instability
    padded, padx, pady = _pad_data(data, data.shape)
    kx, ky = _fftfreqs(dx, dy, data.shape, padded.shape)
    kz = np.sqrt(kx**2 + ky**2)
    upcont_ft = np.fft.fft2(padded) * np.exp(-height * kz)
    cont = np.real(np.fft.ifft2(upcont_ft))
    # Remove padding
    return cont[padx : padx + nx, pady : pady + ny]
