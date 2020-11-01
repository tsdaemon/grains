from functools import reduce

import cv2 as cv

BLUR_CORE = 8
CLAHE_CLIP_LIMIT = 25.0
CLAHE_CORE = 64
MIN_CONTOUR_LENGTH = 10
LIGHT_LIMIT = 2500


def find_image_contours(img):
    """
    Find grains contours on an image

    Parameters
    ----------
    img : np.array[x, y, 3]
        RGB grains image

    Returns
    -------
    List of np.array[n, 1, 2]
        List of contours found. Each contour is a set of xy points
   """
    gray_img = _gray(img)
    find_contours_img = reduce(
        lambda x, y: y(x), [_blur, _clahe, _triangle_threshold], gray_img
    )
    find_contours_img_crop = _crop_to_max_point(gray_img, find_contours_img)
    contours = _find_contours(find_contours_img_crop)
    return contours


def _gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def _blur(img):
    return cv.blur(img, (BLUR_CORE, BLUR_CORE))


def _clahe(img):
    clahe = cv.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_CORE, CLAHE_CORE))
    return clahe.apply(img)


def _triangle_threshold(img):
    return cv.threshold(img, 0, 255, cv.THRESH_TRIANGLE+cv.THRESH_BINARY)[1]


def _crop_to_max_point(original_img, img_to_crop):
    _, _, _, max_loc = cv.minMaxLoc(original_img)
    xs = int(max(max_loc[1] - LIGHT_LIMIT, 0))
    xe = int(min(max_loc[1] + LIGHT_LIMIT, original_img.shape[0]))
    ys = int(max(max_loc[0] - LIGHT_LIMIT, 0))
    ye = int(min(max_loc[0] + LIGHT_LIMIT, original_img.shape[1]))
    img_crop = img_to_crop[xs:xe, ys:ye]
    return img_crop


def _find_contours(img):
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # clean noise
    contours = [c for c in contours if len(c) >= MIN_CONTOUR_LENGTH]
    return contours
