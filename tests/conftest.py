import os

import pytest


@pytest.fixture
def test_dir():
    return os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def test_image_path(test_dir):
    return os.path.join(test_dir, os.path.join('testdata', 'testimage1.jpg'))
