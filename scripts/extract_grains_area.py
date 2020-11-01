import argparse

from grains.core import extract_contour_areas_and_output

parser = argparse.ArgumentParser(description='Estimate area of grains on a picture.')

parser.add_argument('--img', type=str, help='Path to an image with grains.', required=True)
parser.add_argument('--out-img', type=str, help='Path for output image with grains contours.', required=True)
parser.add_argument(
    '--out-csv',
    type=str,
    help='Path for output CSV dataset with grains areas and debug information.',
    required=True
)

if __name__ == '__main__':
    args = parser.parse_args()
    extract_contour_areas_and_output(
        args.img,
        args.out_img,
        args.out_csv,
    )
