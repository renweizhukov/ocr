# ocr
Use Tesseract OCR to recognize English and Simplified-Chinese characters.

## 1. ocr-preprocessing

This exectuable preprocesses a colored image for the later OCR. After it loads the colored image, it 

* sharpens the image using Unsharp Masking with a Gaussian blurred version of the image;
* converts the image into a grayscale one;
* converts the grayscale image into a binary one by thresholding.

At the end, it will write the binary thresholded image into a file.

Below is a sample usage of the exectuable.

```bash
./ocr-preprocessing ./color-image.png ./binary-thresholded-image.png
```

## 2. extract-booktitle-batch

This executable extracts the title part from a set of book cover images based on a given title template image. After preprocessing the images, it will use either SURF+homography+perspectiveTransform or template matching to find the rectangle region inside the book cover images which contains the title and crop the title from the book cover images. At the end, it will write the cropped images into files.

Below is two sample usages of the executable.

```bash
./extract-booktitle-batch -i title-template.png -d ./book-cover-imgs/ -o ./cropped-imgs/ -m homo
./extract-booktitle-batch -i title-template.png -d ./book-cover-imgs/ -o ./cropped-imgs/ -m templ
```




