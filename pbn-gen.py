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
#              COLOR CLUSTERING AND IMAGE PROCESSING              #
###################################################################

def cluster(image: Img, n_colors: int, seed: int = 0) -> tp.Tuple[Colors, Img]:
    """ Cluster the colors of an image and return the colors and an image with the colors
        replaced by their assigned cluster center (color).

        Args:
            image (Img): The image to cluster.
            n_colors (int): The number of colors to cluster the image into.
            seed: The seed for the random number generator.

        Returns:
            Colors: The cluster centers (colors).
            Img: The image with the colors replaced by their assigned cluster center.
    """

    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=seed, n_init='auto').fit(pixels)
    colors = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.labels_.astype(np.uint8)
    return colors, labels


def get_smooth_image(shape: Shape, colors: Colors, labels: Img, blur_size: int = 3) -> Img:
    """ Get an image where each pixel is colored with the color of its cluster center.
        The image is smoothed using a median filter.

        Args:
            shape (Shape): The shape of the image.
            colors (Colors): The cluster centers (colors).
            labels (Img): The cluster labels of each pixel.
            blur_size (int): The size of the neighborhood for smoothing.

        Returns:
            Img: The image with the pixels colored with the color of their cluster center
                 after smoothing with a median blur filter.
    """

    img: Img = colors[labels].reshape(shape)
    return cv.medianBlur(img, blur_size)


def blur_image(image: Img) -> Img:
    """ Apply a bilateral filter to an image.

        Args:
            image (Img): The image to blur.

        Returns:
            Img: The blurred image.
    """

    return cv.bilateralFilter(image, 5, 200, 50)



###################################################################
#                     IMAGE LOADING AND SAVING                    #
###################################################################

def load_image(path: str, max_size: tp.Optional[tp.Tuple[int, int]] = None) -> np.ndarray:
    """ Load an image from a file. If max_size is specified, the image is resized to fit within
        the size while maintaining the aspect ratio.

        Args:
            path (str): The path of the image file.
            max_size (Optional[Tuple[int, int]]): The maximum size of the resized image
                as (width, height).

        Returns:
            Img: The loaded image.
    """

    img: Img = cv.imread(path)
    if max_size is not None:
        # Resize the image to fit within the max size while maintaining the aspect ratio
        height, width = img.shape[:2]
        target_width, target_height = max_size
        shrink_ratio = min(target_width / width, target_height / height)
        if shrink_ratio < 1:
            img = cv.resize(img, (0, 0), fx=shrink_ratio, fy=shrink_ratio,
                            interpolation=cv.INTER_AREA)

    return img


def save_image(path: str, image: np.ndarray) -> None:
    """ Save an image to a png file. """

    if not path.endswith('.png'):
        path += '.jpg'

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
    parser.add_argument('-r', '--resize',
                        type=int, nargs=2, required=False, default=None,
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Resize the input image to fit within the specified size \
                              The aspect ratio is maintained.'
    )
    parser.add_argument('-o', '--output',
                        type=str, required=True,
                        help='Path of output image. \
                              The file extension is always set to .png'
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
    for path in args.input:
        image = load_image(path, args.resize)
        image = blur_image(image)
        cv.imshow('image', image); cv.waitKey(0); cv.destroyAllWindows() # debug
        colors, labels = cluster(image, args.color_palette_size)
        colored_image = get_smooth_image(image.shape, colors, labels)
        cv.imshow('image', colored_image); cv.waitKey(0); cv.destroyAllWindows() # debug
        save_image(args.output, colored_image)


if __name__ == "__main__":
    main()