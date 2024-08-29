import cv2 as cv
import numpy as np
import typing as tp
from enum import Enum
from argparse import ArgumentParser
from sklearn.cluster import KMeans # type: ignore


###################################################################
#                  TYPE DEFINITIONS AND CONSTANTS                 #
###################################################################

# todo


###################################################################
#                     IMAGE LOADING AND SAVING                    #
###################################################################

# todo


###################################################################
#                ARGUMENT PARSING AND MAIN FUNCTION               #
###################################################################


def arg_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        help='Path of input image.'
    )
    parser.add_argument('-o', '--output',
                        type=str, required=True,
                        help='Path of output image. \
                              The file extension is ignored and set to .png'
    )
    parser.add_argument(
        '-n', '--color-palette-size',
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
    args = arg_parser().parse_args()
    print(args) # debug
    # todo


if __name__ == "__main__":
    main()