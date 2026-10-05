import numpy as np
import pytest

from flybrain.spectral.bands import equal_count_bands


def test_bands_partition_contiguously_with_remainder_last():
    bands = equal_count_bands(10, 4)
    assert [b.tolist() for b in bands] == [[0, 1], [2, 3], [4, 5], [6, 7, 8, 9]]


def test_bands_cover_every_mode_once():
    bands = equal_count_bands(4999, 4)
    np.testing.assert_array_equal(np.concatenate(bands), np.arange(4999))


def test_too_few_modes_raises():
    with pytest.raises(ValueError):
        equal_count_bands(3, 4)
