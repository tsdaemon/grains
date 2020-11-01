import pandas as pd
import cv2 as cv


def get_contours_area(contours):
    """
    Calculates area for each image contour and returns contours along with areas as DataFrame

    Parameters
    ----------
    contours : List of np.array[n, 1, 2]
        Contours on image

    Returns
    -------
    pd.DataFrame
        DataFrame with contours and areas
    """
    contours_df = pd.DataFrame(contours, columns=['contours'])
    contours_df['areas'] = contours_df['contours'].apply(cv.contourArea)
    contours_df = contours_df.sort_values(by='areas', ascending=False).reset_index(drop=True)
    return contours_df
