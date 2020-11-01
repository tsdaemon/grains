import numpy as np
import pytest

from grains.area import get_contours_area


@pytest.fixture
def contours():
    return [
        np.array([[[0, 0]], [[0, 1]], [[1, 1]]]),
        np.array([[[0, 0]], [[0, 1]], [[1, 1]], [[1, 0]]]),
    ]


def test_get_contours_area(contours):
    df = get_contours_area(contours)
    assert (df['areas'] == [1, 0.5]).all()
    assert (df.index == [0, 1]).all()
