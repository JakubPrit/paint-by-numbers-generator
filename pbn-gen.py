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

    return cv.bilateralFilter(image, 7, 170, 70)


def get_connected_components(mask: Mask
                             ) -> tp.Tuple[Contours, tp.List[int], tp.List[tp.List[int]], tp.Any]:
    """ Get the connected components of a binary mask.

        Args:
            mask (Mask): The binary mask to get the connected components of.

        Returns:
            tp.List[Contours, tp.List[int], tp.List[tp.List[int]]]: The contours of the connected
                components, the depths of the contours, and list of children for each contour.
    """

    # Find the contours of the connected components in the mask
    contours: Contours
    contours, hierarchy = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE, # type: ignore
                                            cv.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]

    # Find the children and depths of the contours
    children: tp.List[tp.List[int]] = [[] for _ in range(len(contours))]
    for i in range(len(contours)):
        if hierarchy[i][3] >= 0:
            children[hierarchy[i][3]].append(i)
    depths = [0] * len(contours)
    stack = [i for i in range(len(contours)) if hierarchy[i][3] < 0]
    while stack:
        i = stack.pop()
        for j in children[i]:
            depths[j] = depths[i] + 1
            stack.append(j)

    return contours, depths, children, hierarchy


def mask_good_components(image: Img, colors: Colors, color_mode: ColorMode,
                         min_size: int, max_components: int) -> Mask:
    """ Mask out small connected components from an image. By connected components,
        connected regions of the same color are meant. Also removes thin border components
        (components that are almost completely filled by other components).

        Args:
            image (Img): The image to remove small components from.
            colors (Colors): The colors in the image.
            color_mode (ColorMode): The color mode of the image.
            min_size (int): The minimum size of the connected components to keep.
            max_components (int): The maximum number of connected components to keep.

        Returns:
            Mask: The mask of the kept connected components.
    """

    # Convert the image to a list of masks, one for each color
    if color_mode == ColorMode.GRAYSCALE:
        masks = [image == color for color in colors]
    else:
        masks = [np.all(image == color, axis=-1) for color in colors]

    good_contours = []
    all_contours, all_depths, all_children, all_areas = [], [], [], []

    for i_mask in range(len(masks)):
        mask = masks[i_mask]

        contours, depths, children, hierarchy = get_connected_components(mask)

        # Separate the outer and inner contours (inner contours are contours of holes)
        outer_contours_idx = [i for i in range(len(contours)) if depths[i] % 2 == 0]
        inner_contours_idx = [i for i in range(len(contours)) if depths[i] % 2 == 1]

        # Compute the areas of the components
        contour_areas = list(map(cv.contourArea, contours))
        component_areas = contour_areas.copy()
        for i in inner_contours_idx:
            component_areas[hierarchy[i][3]] -= component_areas[i]

        # Keep the outer components with an area greater than the minimum size and are not thin
        THIN_THRESHOLD = -1.0
        good_contours.extend([(i_mask, i) for i in outer_contours_idx
                              if component_areas[i] >= min_size
                              and component_areas[i] >= THIN_THRESHOLD * contour_areas[i]])

        all_contours.append(contours)
        all_depths.append(depths)
        all_children.append(children)
        all_areas.append(component_areas)
        del mask, contours, hierarchy, children, depths, outer_contours_idx, inner_contours_idx, \
            contour_areas, component_areas # just to be sure that I don't use them by mistake

    # Remove the smallest components if there are too many components
    good_contours.sort(key=lambda x: all_areas[x[0]][x[1]], reverse=True)
    if max_components > 0 and len(good_contours) > max_components:
        good_contours = good_contours[:max_components]

    all_good_contours_idx: tp.List[tp.List[int]] = [list() for _ in range(len(masks))]
    for i_mask, i in good_contours:
        all_good_contours_idx[i_mask].append(i)
    del good_contours # just to be sure that I don't use it by mistake

    # Create a mask of the kept components
    kept_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for i_mask in range(len(masks)):
        kept_mask_this_color = np.zeros_like(kept_mask, dtype=np.uint8)
        kept_contours = sorted(all_good_contours_idx[i_mask], key=lambda x: all_depths[i_mask][x])
        for i in kept_contours:
            cv.drawContours(kept_mask_this_color, [all_contours[i_mask][i]], 0, 1, -1)
            for j in all_children[i_mask][i]:
                cv.drawContours(kept_mask_this_color, [all_contours[i_mask][j]], 0, 0, -1)
        kept_mask |= kept_mask_this_color
        del kept_mask_this_color, kept_contours # just to be sure that I don't use them by mistake

    dbg_img = image.copy() # debug
    dbg_img[np.logical_not(kept_mask.astype(bool))] = 0 # debug
    debug_show_image(dbg_img, color_mode) # debug

    return kept_mask.astype(bool)


def fill_unmasked(image: Img, mask: Mask, colors: Colors, color_mode: ColorMode) -> Img:
    """ Fill the non-masked regions of an image with the colors of the neighboring regions.

        Args:
            image (Img): The image to fill.
            mask (Mask): The mask of the regions not to fill.
            colors (Colors): The colors of the masked regions.
            color_mode (ColorMode): The color mode of the image.

        Returns:
            Img: The image with the unmasked regions filled.
    """

    if color_mode == ColorMode.GRAYSCALE:
        inverse_color_masks = [image != color for color in colors]
    else:
        inverse_color_masks = [np.any(image != color, axis=-1).astype(np.uint8)
                               for color in colors]
    inverse_mask = np.logical_not(mask)
    for i in range(len(inverse_color_masks)):
        inverse_color_masks[i] |= inverse_mask
    color_distances = [cv.distanceTransform(inv_color_mask, cv.DIST_L2, 3)
                       for inv_color_mask in inverse_color_masks]

    # closest color = color with the smallest color distance for each pixel
    closest_colors = colors[np.argmin(color_distances, axis=0)]
    result = image.copy()
    result[inverse_mask] = closest_colors[inverse_mask]
    debug_show_image(result, color_mode) # debug
    return result


def largest_inscribed_circles(image: Img, colors: Colors, color_mode: ColorMode) -> Img:
    """ Get the largest inscribed circles in the connected components of an image.

        Args:
            image (Img): The image to get the inscribed circles of.
            colors (Colors): The colors in the image.
            color_mode (ColorMode): The color mode of the image.

        Returns:
            Img: The image with the largest inscribed circles.
    """

    # TODO

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

    # TODO

    # // bordered = cv.copyMakeBorder(mask.astype(np.uint8), 1, 1, 1, 1, cv.BORDER_CONSTANT, value=255)
    # // bordered = cv.dilate(bordered, np.ones((3, 3), np.uint8), iterations=1)
    # // bordered = np.logical_not(bordered).astype(np.uint8)*255
    # // debug_show_image(bordered, ColorMode.GRAYSCALE)
    # // n_components, components = cv.connectedComponents(bordered, connectivity=4)

    # // hue = np.uint8(180 * components / n_components)
    # // full255 = np.full_like(hue, 255, np.uint8)
    # // dbg_img = cv.merge([hue, full255, full255])
    # // dbg_img[hue==0] = (0, 0, 0)
    # // debug_show_image(dbg_img, ColorMode.HSV)
    # // save_image('connected.png', dbg_img, ColorMode.HSV)

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
                        default=200,
                        help='Minimum size of a cell in the output image in pixels \
                        Defaults to 200. If set to 0, all cells are naturally kept.'
    )
    parser.add_argument('-M', '--max-cells',
                        type=int, required=False,
                        default=250,
                        help='Maximum number of cells in the output image. Defaults to 250. \
                        If set to 0 or a negative number, no limit is imposed.'
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
def debug_show_image(image: Img, color_mode: ColorMode) -> None:
    cv.imshow('image', cvt_to_bgr(image, color_mode)); cv.waitKey(0); cv.destroyWindow('image')


def main() -> None:
    args = _arg_parser().parse_args()

    color_mode = ColorMode[args.color_mode]

    for path in args.input:
        seed = args.seed if args.seed is not None else randint(0, 1000)

        image = load_image(path, color_mode, args.resize)
        image = blur_image(image)
        debug_show_image(image, color_mode) # debug

        colors, labels = cluster(image, args.color_palette_size, seed)
        colored_image = get_smooth_image(image.shape, colors, labels)
        debug_show_image(colored_image, color_mode) # debug

        good_mask = mask_good_components(colored_image, colors, color_mode,
                                         args.min_cell_size, args.max_cells)
        colored_image = fill_unmasked(colored_image, good_mask, colors, color_mode)

        contours = get_contours(colored_image)
        outlines = get_outlines_mask(contours, colored_image.shape)
        # numbers = get_numbers(outlines)

        bgr_image = cvt_to_bgr(colored_image, color_mode)
        if args.outline:
            bgr_image[outlines] = OUTLINE_COLOR
        save_image(args.output, bgr_image, ColorMode.BGR)


if __name__ == "__main__":
    main()