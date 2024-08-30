# Paint by Number Generator

This project is a Python script that can convert normal images into Paint by Number images (either filled with colors or just outlines and labels) utilizing KMeans clustering.

## Features

- Command line interface (CLI)
- Choose a color mode to cluster by (RGB, HSL, HSV, Lab, grayscale)
  and the amounts of clusters (colors in the color palette)
- Resize large images to fit a chosen size, keeping the aspect ratio, to speed up the processing
- Choose what parts of the image to generate (outlines, numbers, color fill)
- Use a seed for the clustering algorithm to get consistent results
- Supports a variety of image formats (the ones supported by OpenCV)

## Installation

### Dependencies

- Python 3.9 or higher
- OpenCV 4.8.1 or higher
- Numpy 1.25.1 or higher
- Scikit-learn 1.3.2 or higher

Lover versions of the dependencies might work, but they have not been tested.

To install the dependencies, run the following command:

```none
pip install numpy opencv-python scikit-learn
```

### Setup

Just download the pbn-gen.py file and install the dependencies.
It is then ready to be run with Python.

## Usage

```none
usage: pbn-gen.py [-h] [-r WIDTH HEIGHT] [-c {BGR,HSL,HSV,LAB,GRAYSCALE}] -o OUTPUT [-k COLOR_PALETTE_SIZE] [-l] [-f] [-n] [-s SEED] input [input ...]

positional arguments:
  input                 Paths of input images.

options:
  -h, --help            show this help message and exit
  -r WIDTH HEIGHT, --resize WIDTH HEIGHT
                        Resize the input image to fit within the specified size The aspect ratio is maintained.
  -c {BGR,HSL,HSV,LAB,GRAYSCALE}, --color-mode {BGR,HSL,HSV,LAB,GRAYSCALE}
                        Color mode of the input image. Defaults to LAB.
  -o OUTPUT, --output OUTPUT
                        Path of output image.
  -k COLOR_PALETTE_SIZE, --color-palette-size COLOR_PALETTE_SIZE
                        Number of colors used in the output. Defaults to 10.
  -l, --outline         Have outlines in the output image.
  -f, --fill            Fill the output image with the colors.
  -n, --numbers         Have numbers in the output image.
  -s SEED, --seed SEED  Seed for the color clustering algorithm. If not provided, a random seed is used.
```
