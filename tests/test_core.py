import os
import tempfile

import cv2 as cv
import pandas as pd

from grains.core import extract_contour_areas_and_output


def test_extract_contour_areas_and_output(test_image_path):
    temp_dir = tempfile.gettempdir()
    out_img_path = os.path.join(temp_dir, 'out.jpg')
    out_csv_path = os.path.join(temp_dir, 'out.csv')

    extract_contour_areas_and_output(test_image_path, out_img_path, out_csv_path)

    out_img = cv.imread(out_img_path)
    assert out_img.shape == (200, 200, 3)

    out_df = pd.read_csv(out_csv_path)
    assert len(out_df) == 50
    assert (out_df.columns == ['index', 'areas']).all()
