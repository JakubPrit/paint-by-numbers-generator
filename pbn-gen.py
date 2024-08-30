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
Mask = npt.NDArray[np.bool_]
Color = tp.Tuple[int, int, int]
Colors = npt.NDArray[np.uint8]
Shape = tp.Tuple[int, ...]
Contours = tp.List[np.ndarray]
class ColorMode(Enum):
    BGR = 'BGR'
    HSL = 'HSL'
    HSV = 'HSV'
    LAB = 'LAB'
    GRAYSCALE = 'GRAYSCALE'


OUTLINE_COLOR = (0, 0, 0)

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

    pixels = image.reshape(-1, image.shape[-1] if len(image.shape) > 2 else 1)
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
#                  OUTLINES AND NUMBERS (LABELS)                  #
###################################################################

def get_contours(image: Img) -> Contours:
    """ Get the contours (edges / outlines) of an image.

        Args:
            image (Img): The image to get the contours of.

        Returns:
            Contours: The contours of the image.
    """

    edges = cv.Canny(image, 0, 0)
    edges = edges.astype(bool)
    contours, _ = cv.findContours(edges.astype(np.uint8), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    dbg_img = np.zeros_like(edges, dtype=np.uint8) # debug
    cv.drawContours(dbg_img, contours, -1, 255, 1) # debug
    debug_show_image(dbg_img, ColorMode.GRAYSCALE) # debug
    save_image('contours.png', dbg_img, ColorMode.GRAYSCALE) # debug
    return contours


def get_outlines_mask(contours: Contours, shape: Shape) -> Mask:
    """ Get a mask of the outlines of an image.

        Args:
            contours (Contours): The contours of the image.
            shape (Shape): The shape of the image.

        Returns:
            Mask: The mask of the outlines.
    """

    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv.drawContours(mask, contours, -1, 1, 1)
    return mask.astype(bool)


def get_numbers(contours: Contours) -> Mask:
    """ Get the numbers (labels) for the outlines of an image.

        Args:
            contours (Contours): The contours of the image

        Returns:
            Mask: The mask with the numbers (labels).
    """

    # todo
    raise NotImplementedError


###################################################################
#                     IMAGE LOADING AND SAVING                    #
###################################################################

def cvt_from_bgr(bgr_img: Img, mode: ColorMode) -> Img:
    """ Convert an image from BGR to another color mode.

        Args:
            bgr_img (Img): The BGR image to convert.
            mode (ColorMode): The color mode to convert to.
        
        Returns:
            Img: The converted image.
    """

    if mode == ColorMode.BGR: return bgr_img
    elif mode == ColorMode.HSL: return cv.cvtColor(bgr_img, cv.COLOR_BGR2HLS)
    elif mode == ColorMode.HSV: return cv.cvtColor(bgr_img, cv.COLOR_BGR2HSV)
    elif mode == ColorMode.LAB: return cv.cvtColor(bgr_img, cv.COLOR_BGR2LAB)
    elif mode == ColorMode.GRAYSCALE: return cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)


def cvt_to_bgr(img: Img, mode: ColorMode) -> Img:
    """ Convert an image to BGR from another color mode.

        Args:
            img (Img): The image to convert.
            mode (ColorMode): The color mode to convert from.
        
        Returns:
            Img: The converted image in BGR.
    """

    if mode == ColorMode.BGR: return img
    elif mode == ColorMode.HSL: return cv.cvtColor(img, cv.COLOR_HLS2BGR)
    elif mode == ColorMode.HSV: return cv.cvtColor(img, cv.COLOR_HSV2BGR)
    elif mode == ColorMode.LAB: return cv.cvtColor(img, cv.COLOR_LAB2BGR)
    elif mode == ColorMode.GRAYSCALE: return cv.cvtColor(img, cv.COLOR_GRAY2BGR)


def load_image(path: str, color_mode: ColorMode, max_size: tp.Optional[tp.Tuple[int, int]] = None,
               ) -> np.ndarray:
    """ Load an image from a file. If max_size is specified, the image is resized to fit within
        the size while maintaining the aspect ratio. The image is converted to the specified
        color mode.

        Args:
            path (str): The path of the image file.
            color_mode (ColorMode): The color mode to convert the image to.
            max_size (Optional[Tuple[int, int]]): The maximum size of the resized image
                as (width, height).

        Returns:
            Img: The loaded image in the specified color mode.
    """

    if not cv.haveImageReader(path):
        raise ValueError(f"Unsupported image format for file {path}.")
    try:
        img: Img = cv.imread(path, cv.IMREAD_COLOR)
    except Exception as e:
        raise ValueError(f"Error loading image from {path}: {e}")
    if max_size is not None:
        # Resize the image to fit within the max size while maintaining the aspect ratio
        height, width = img.shape[:2]
        target_width, target_height = max_size
        shrink_ratio = min(target_width / width, target_height / height)
        if shrink_ratio < 1:
            img = cv.resize(img, (0, 0), fx=shrink_ratio, fy=shrink_ratio,
                            interpolation=cv.INTER_AREA)

    return cvt_from_bgr(img, color_mode)


def save_image(path: str, image: np.ndarray, image_color_mode: ColorMode) -> None:
    """ Save an image to a file. The image is converted to BGR before saving.

        Args:
            path (str): The path of the image file. If no extension is provided,
                or the extension is not supported by OpenCV, '.png' is appended.
            image (Img): The image to save.
            image_color_mode (ColorMode): The color mode of the input image.
    """

    if not cv.haveImageWriter(path):
        path += ('.png')
    image = cvt_to_bgr(image, image_color_mode)
    try:
        cv.imwrite(path, image)
    except Exception as e:
        print(f"Error saving image to {path}: {e}")


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
    parser.add_argument('-c', '--color-mode',
                        type=lambda x: x.upper(),
                        required=False, default=ColorMode.LAB.value,
                        choices=[mode.value for mode in ColorMode],
                        help='Color mode of the input image. Defaults to LAB.'
    )
    parser.add_argument('-o', '--output',
                        type=str, required=True,
                        help='Path of output image.'
    )
    parser.add_argument('-k', '--color-palette-size',
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


# debug function
def debug_show_image(image: Img, colormode: ColorMode) -> None:
    cv.imshow('image', cvt_to_bgr(image, colormode)); cv.waitKey(0); cv.destroyWindow('image')


def main() -> None:
    args = _arg_parser().parse_args()
    for path in args.input:
        color_mode = ColorMode[args.color_mode]
        image = load_image(path, color_mode, args.resize)
        image = blur_image(image)
        debug_show_image(image, color_mode)
        colors, labels = cluster(image, args.color_palette_size)
        colored_image = get_smooth_image(image.shape, colors, labels)
        debug_show_image(colored_image, color_mode)
        contours = get_contours(colored_image)
        outlines = get_outlines_mask(contours, colored_image.shape)
        # numbers = get_numbers(contours)
        bgr_image = cvt_to_bgr(colored_image, color_mode)
        if args.outline:
            bgr_image[outlines] = OUTLINE_COLOR
        save_image(args.output, bgr_image, ColorMode.BGR)


if __name__ == "__main__":
    main()