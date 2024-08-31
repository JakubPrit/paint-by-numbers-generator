import cv2 as cv
import numpy as np
import typing as tp
import numpy.typing as npt
from enum import Enum
from argparse import ArgumentParser
from sklearn.cluster import KMeans # type: ignore
from random import randint


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


def get_smooth_image(shape: Shape, colors: Colors, labels: Img) -> Img:
    """ Get an image where each pixel is colored with the color of its cluster center.
        The image is smoothed using a median filter.

        Args:
            shape (Shape): The shape of the image.
            colors (Colors): The cluster centers (colors).
            labels (Img): The cluster labels of each pixel.

        Returns:
            Img: The image with the pixels colored with the color of their cluster center
                 after smoothing with a median blur filter.
    """

    BLUR_SIZE = 3
    BLUR_ITERS = 3

    img: Img = colors[labels].reshape(shape)
    for _ in range(BLUR_ITERS):
        img = cv.medianBlur(img, BLUR_SIZE)
    return img


def blur_image(image: Img) -> Img:
    """ Apply a bilateral filter to an image.

        Args:
            image (Img): The image to blur.

        Returns:
            Img: The blurred image.
    """

    return cv.bilateralFilter(image, 7, 150, 50)


def remove_small_components(image: Img, colors: Colors,
                            min_size: int, max_components: int) -> Img:
    """ Remove small connected components from an image. By connected components,
        connected regions of the same color are meant.

        Args:
            image (Img): The image to remove small components from.
            colors (Colors): The colors in the image.
            min_size (int): The minimum size of the connected components to keep.
            max_components (int): The maximum number of connected components to keep.

        Returns:
            Img: The image with small connected components removed.
    """

    masks = [np.all(image == color, axis=-1) for color in colors]
    good_contours = []
    max_size = image.shape[0] * image.shape[1] # debug
    dbg_img = cvt_from_bgr(cvt_to_bgr(image, ColorMode.LAB), ColorMode.HSV) # debug
    for mask in masks:
        contours, hierarchy = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE,
                                              cv.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0]
        root_contours_idx = [i for i in range(len(contours)) if hierarchy[i][3] < 0]
        children: tp.List[tp.List[int]] = [[] for _ in range(len(contours))]
        for i in range(len(contours)):
            if hierarchy[i][3] >= 0:
                children[hierarchy[i][3]].append(i)
        depths = [0] * len(contours)
        stack = root_contours_idx.copy()
        while stack:
            i = stack.pop()
            for j in children[i]:
                depths[j] = depths[i] + 1
                stack.append(j)
        outer_contours_idx = [i for i in range(len(contours)) if depths[i] % 2 == 0]
        inner_contours_idx = [i for i in range(len(contours)) if depths[i] % 2 == 1]

        areas = list(map(cv.contourArea, contours))
        for i in inner_contours_idx:
            areas[hierarchy[i][3]] -= areas[i]
        good_contours.extend(outer_contours_idx)
        for i in outer_contours_idx: # debug
            contour, area = contours[i], areas[i] # debug
            hue = int(np.log(area + 1) / np.log(max_size + 1) * 180) # debug
            cv.drawContours(dbg_img, [contour], -1, (hue, 255, 255), 1) # debug
        debug_show_image(mask.astype(np.uint8)*255, ColorMode.GRAYSCALE) # debug
    debug_show_image(dbg_img, ColorMode.HSV) # debug
    print(len(good_contours)) # debug
    good_contours.sort(key=lambda i: areas[i], reverse=True)
    if len(good_contours) > max_components:
        good_contours = good_contours[:max_components]

    kept_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    dbg_contours = list(map(lambda i: contours[i], good_contours)) # debug
    cv.drawContours(kept_mask, dbg_contours, -1, 1, -1)
    dbg_img = image.copy()
    dbg_img[np.logical_not(kept_mask.astype(bool))] = 0
    debug_show_image(dbg_img, ColorMode.LAB) # debug

    raise NotImplementedError


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
    debug_show_image(edges, ColorMode.GRAYSCALE) # debug
    save_image('edges.png', edges, ColorMode.GRAYSCALE) # debug
    edges = edges.astype(bool)
    contours, _ = cv.findContours(edges.astype(np.uint8), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    dbg_img = np.zeros_like(edges, dtype=np.uint8) # debug
    cv.drawContours(dbg_img, contours, -1, 255, 1) # debug
    debug_show_image(dbg_img, ColorMode.GRAYSCALE) # debug
    save_image('contours.png', dbg_img, ColorMode.GRAYSCALE) # debug
    cv.drawContours(dbg_img, contours, -1, 255, -1) # debug
    debug_show_image(dbg_img, ColorMode.GRAYSCALE) # debug
    save_image('contours_fill.png', dbg_img, ColorMode.GRAYSCALE) # debug
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


def get_numbers(mask: Mask) -> Mask:
    """ Get the numbers (labels) for the outlines of an image.

        Args:
            mask (Mask): The mask of the outlines.

        Returns:
            Mask: The mask with the numbers (labels).
    """

    bordered = cv.copyMakeBorder(mask.astype(np.uint8), 1, 1, 1, 1, cv.BORDER_CONSTANT, value=255)
    bordered = cv.dilate(bordered, np.ones((3, 3), np.uint8), iterations=1)
    bordered = np.logical_not(bordered).astype(np.uint8)*255
    debug_show_image(bordered, ColorMode.GRAYSCALE)
    n_components, components = cv.connectedComponents(bordered, connectivity=4)

    hue = np.uint8(180 * components / n_components)
    full255 = np.full_like(hue, 255, np.uint8)
    dbg_img = cv.merge([hue, full255, full255])
    dbg_img[hue==0] = (0, 0, 0)
    debug_show_image(dbg_img, ColorMode.HSV)
    save_image('connected.png', dbg_img, ColorMode.HSV)

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
    parser.add_argument('-o', '--output',
                        type=str, required=True,
                        help='Path of output image.'
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
    parser.add_argument('-k', '--color-palette-size',
                        type=int, required=False,
                        default=10,
                        help='Number of colors used in the output. Defaults to 10.'
    )
    parser.add_argument('-m', '--min-cell-size',
                        type=int, required=False,
                        default=400,
                        help='Minimum size of a cell in the output image in pixels \
                        Defaults to 400.'
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
    parser.add_argument('-s', '--seed',
                        type=int, required=False,
                        default=None,
                        help='Seed for the color clustering algorithm. If not provided, \
                        a random seed is used.'
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
        seed = args.seed if args.seed is not None else randint(0, 1000)
        colors, labels = cluster(image, args.color_palette_size, seed)
        colored_image = get_smooth_image(image.shape, colors, labels)
        debug_show_image(colored_image, color_mode)
        remove_small_components(colored_image, colors, 1000, 100)
        contours = get_contours(colored_image)
        outlines = get_outlines_mask(contours, colored_image.shape)
        # numbers = get_numbers(outlines)
        bgr_image = cvt_to_bgr(colored_image, color_mode)
        if args.outline:
            bgr_image[outlines] = OUTLINE_COLOR
        save_image(args.output, bgr_image, ColorMode.BGR)


if __name__ == "__main__":
    main()