import cv2 as cv
import numpy as np
import typing as tp
import numpy.typing as npt
from enum import Enum
from argparse import ArgumentParser
from sklearn.cluster import KMeans # type: ignore


###################################################################
#                  TYPE DEFINITIONS AND CONSTANTS                 #
###################################################################

Img = npt.NDArray[np.uint8]
Color = tp.Tuple[int, int, int]
Colors = npt.NDArray[np.uint8]
Shape = tp.Tuple[int, ...]


###################################################################
#                         COLOR CLUSTERING                        #
###################################################################

def cluster(image: Img, n_colors: int, seed: int = 0) -> tp.Tuple[Colors, Img]:
    """ Cluster the colors of an image and return the colors and an image with the colors
        replaced by their assigned cluster center (color).

        Args:
            image: The image to cluster.
            n_colors: The number of colors to cluster the image into.
            seed: The seed for the random number generator.

        Returns:
            colors: The cluster centers (colors).
            clustered_image: The image with the colors replaced
                             by their assigned cluster center.
    """

    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=seed).fit(pixels)
    colors = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.labels_.astype(np.uint8)
    return colors, labels


def get_colored_image(shape: Shape, colors: Colors, labels: Img) -> Img:
    """ Get an image where each pixel is colored with the color of its cluster center.

        Args:
            shape (Shape): The shape of the image.
            colors (Colors): The cluster centers (colors).
            labels (Img): The cluster labels of each pixel.

        Returns:
            colored_image (Img): The image with the pixels colored
                           with the color of their cluster center.
    """

    return colors[labels].reshape(shape)


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
        colors, labels = cluster(image, args.color_palette_size)
        colored_image = get_colored_image(image.shape, colors, labels)
        cv.imshow('image', colored_image); cv.waitKey(0); cv.destroyAllWindows() # debug
        save_image(args.output, colored_image)


if __name__ == "__main__":
    main()