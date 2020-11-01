import cv2 as cv
import pytest

from grains.find_contours import find_image_contours


@pytest.fixture
def test_image(test_image_path):
    img = cv.imread(test_image_path)
    assert img is not None, f'Image is not found at {test_image_path}'
    return img


def test_find_image_contours(test_image):
    contours = find_image_contours(test_image)
    assert contours
