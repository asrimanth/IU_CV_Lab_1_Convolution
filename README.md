# Image Convolution from scratch

## Applying 3x3 and 5x5 kernels to an image

We deal with boundaries as follows:

+ Padding an image with 1 row of white pixels for a 3x3 image.
+ Padding an image with 2 rows of white pixels for a 5x5 image.

For a deep dive, please look at the [notebook here](playground.ipynb) and the [source code here](lab1.py).

## Sample outputs

| Original Image                     | Kernel                                | Filtered Image             |
| ---------------------------------- | ------------------------------------- | -------------------------- |
| ![Error](input_clock_resized.jpg)  | -> Identity kernel ->                 | ![Image not found](a.jpg)  |
| ![Error](input_clock_resized.jpg)  | -> Box blur kernel ->                 | ![Image not found](b.jpg)  |
| ![Error](input_clock_resized.jpg)  | -> Horizontal derivative kernel ->    | ![Image not found](c.jpg)  |
| ![Error](input_clock_resized.jpg)  | -> Approximated Gaussian kernel ->    | ![Image not found](d.jpg)  |
| ![Error](input_clock_resized.jpg)  | -> Sharpening kernel (alpha = 0.9) -> | ![Image not found](e.jpg)  |
| ![Error](input_clock_resized.jpg)  | -> Derivative of Gaussian kernel ->   | ![Image not found](f.jpg)  |
