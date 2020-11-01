import cv2 as cv

from grains.area import get_contours_area
from grains.find_contours import find_image_contours

TOP_GRAINS = 100


def extract_contour_areas_and_output(img_path, out_img_path, out_csv_path):
    """
    Reads image, extracts it contours, calculates areas, output it as CSV

    Parameters
    ----------
    img_path : str
        Path to an input image
    out_img_path : str
        Path to an output image with contours and they indices drawn
    out_csv_path : str
        Path to an output CSV file
    """
    img = cv.imread(img_path)
    contours = find_image_contours(img)
    df = get_contours_area(contours)

    # Draw out image for debug
    out_img = _prepare_out_image(df, img)
    cv.imwrite(out_img_path, out_img)

    # Store data as csv
    df = _prepare_out_csv(df)
    df.to_csv(out_csv_path, index=False)


def _prepare_out_image(contours_df, img):
    out_img = img.copy()
    out_img = cv.drawContours(out_img, contours_df['contours'].values, -1, (0, 0, 255), 3)
    for i in range(min(TOP_GRAINS, len(contours_df))):
        contour = contours_df.loc[i]['contours']
        M = cv.moments(contour)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        cv.putText(out_img, str(i), (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2)

    return out_img


def _prepare_out_csv(df):
    df.reset_index(inplace=True)
    del df['contours']
    return df
