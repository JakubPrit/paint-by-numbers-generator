import cv2 as cv
import numpy as np
import typing as tp
from enum import Enum
from argparse import ArgumentParser
from sklearn.cluster import KMeans # type: ignore


###################################################################
#                  TYPE DEFINITIONS AND CONSTANTS                 #
###################################################################

Img = np.ndarray
Pixel = Color = tp.Tuple[int, int, int]
Pixels = np.ndarray


###################################################################
#                         COLOR CLUSTERING                        #
###################################################################

def cluster_colors(image: Img, n_colors: int, seed: int = 0) -> np.ndarray:
    """ Cluster the colors of an image using KMeans.

        Args:
            image (Img): Image to cluster.
            n_colors (int): Number of colors to cluster the image into.
            seed (int, optional): Random seed for the KMeans algorithm. Defaults to 0.

        Returns:
            np.ndarray: Array of RGB colors.
    """

    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=seed).fit(pixels)
    colors = kmeans.cluster_centers_
    return colors


###################################################################
#                     IMAGE LOADING AND SAVING                    #
###################################################################

def load_image(path: str) -> np.ndarray:
    """Load an image from a file."""

    return cv.imread(path)


def save_image(path: str, image: np.ndarray) -> None:
    """Save an image to a file."""

    cv.imwrite(path, image)


###################################################################
#                ARGUMENT PARSING AND MAIN FUNCTION               #
###################################################################


def _arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('input',
                        type=str, nargs='+',
                        help='Paths of input images.'
    )
    parser.add_argument('-o', '--output',
                        type=str, required=True,
                        help='Path of output image. \
                              The file extension is ignored and set to .png'
    )
    parser.add_argument(
        '-k', '--color-palette-size',
        type=int, required=False,
        default=10,
        help='Number of colors used in the output. Defaults to 10.'
    )
    parser.add_argument('-l', '--outline',
                        action="store_true",
                        required=False, default=False,
                        help='Have outlines in the output image.'
    )
    parser.add_argument('-f', '--fill',
                        action="store_true",
                        required=False, default=False,
                        help='Fill the output image with the colors.'
    )
    parser.add_argument('-n', '--numbers',
                        action="store_true",
                        required=False, default=False,
                        help='Have numbers in the output image.'
    )
    return parser


def main() -> None:
    args = _arg_parser().parse_args()
    print(args) # debug
    for path in args.input:
        image = load_image(path)
        colors = cluster_colors(image, args.color_palette_size)
        print(colors) # debug


if __name__ == "__main__":
    main()