import numpy as np
import pytest

from demtools.mathlib import (
    _fftfreqs,
    _nextpow2,
    _pad_data,
    derivx,
    derivy,
    derivz,
    upcontinue,
)


class TestNextPow2:
    def test_nextpow2(self):
        assert _nextpow2(1) == 1
        assert _nextpow2(2) == 2
        assert _nextpow2(3) == 4
        assert _nextpow2(4) == 4
        assert _nextpow2(5) == 8
        assert _nextpow2(8) == 8
        assert _nextpow2(9) == 16


class TestPadData:
    def test_pad_data(self):
        data = np.ones((3, 3))
        padded, padx, pady = _pad_data(data, (3, 3))
        assert padded.shape[0] >= 3
        assert padded.shape[1] >= 3
        assert padx >= 0
        assert pady >= 0

    def test_pad_data_edge_values(self):
        data = np.ones((3, 3))
        padded, _, _ = _pad_data(data, (3, 3))
        assert np.all(padded == 1.0)

    def test_pad_symmetric(self):
        data = np.ones((5, 7))
        padded, padx, pady = _pad_data(data, (5, 7))
        assert padded.shape[0] == 5 + 2 * padx
        assert padded.shape[1] == 7 + 2 * pady


class TestFftFreqs:
    def test_fftfreqs_shapes(self):
        kx, ky = _fftfreqs(1.0, 1.0, (4, 4))
        assert kx.shape == (4, 4)
        assert ky.shape == (4, 4)


class TestDerivX:
    def test_derivx_constant(self):
        data = np.ones((5, 5))
        result = derivx(data, 1.0)
        assert np.allclose(result, 0.0)

    def test_derivx_linear_ramp(self):
        data = np.arange(25.0).reshape(5, 5)
        result = derivx(data, 1.0)
        expected = (data[2:, :] - data[:-2, :]) / 2.0
        assert np.allclose(result[1:-1, :], expected)
        assert np.allclose(result[0, :], result[1, :])
        assert np.allclose(result[-1, :], result[-2, :])

    def test_derivx_zero_dx(self):
        data = np.ones((3, 3))
        with np.errstate(divide="ignore", invalid="ignore"):
            result = derivx(data, 0)
        assert np.all(np.isinf(result[1:-1, :]) | np.isnan(result[1:-1, :]))


class TestDerivY:
    def test_derivy_constant(self):
        data = np.ones((5, 5))
        result = derivy(data, 1.0)
        assert np.allclose(result, 0.0)

    def test_derivy_linear_ramp(self):
        data = np.arange(25.0).reshape(5, 5)
        result = derivy(data, 1.0)
        expected = (data[:, 2:] - data[:, :-2]) / 2.0
        assert np.allclose(result[:, 1:-1], expected)

    def test_derivy_edge_mirror(self):
        data = np.arange(25.0).reshape(5, 5)
        result = derivy(data, 1.0)
        assert np.allclose(result[:, 0], result[:, 1])
        assert np.allclose(result[:, -1], result[:, -2])


class TestDerivZ:
    def test_derivz_constant(self):
        data = np.ones((5, 5))
        result = derivz(data, 1.0, 1.0)
        assert result.shape == (5, 5)

    def test_derivz_nonzero(self):
        data = np.arange(25.0).reshape(5, 5)
        result = derivz(data, 1.0, 1.0)
        assert result.shape == (5, 5)


class TestUpcontinue:
    def test_upcontinue_zero_height(self):
        data = np.ones((5, 5))
        result = upcontinue(data, 1.0, 1.0, 0)
        assert result.shape == (5, 5)
        assert np.allclose(result, data)

    def test_upcontinue_nonzero_height(self):
        data = np.arange(25.0).reshape(5, 5)
        result = upcontinue(data, 1.0, 1.0, 10)
        assert result.shape == (5, 5)
