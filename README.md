# ocr
Use Tesseract OCR to recognize English and Simplified-Chinese characters.

## 1. ocr-preprocessing

This exectuable preprocesses a colored image for the later OCR. After it loads the colored image, it 

* sharpens the image using Unsharp Masking with a Gaussian blurred version of the image;
* converts the image into a grayscale one;
* converts the grayscale image into a binary one by an invert thresholding.

At the end, it will write the binary thresholded image into a file.

Below is a sample usage of the exectuable.

```bash
./ocr-preprocessing ./color-image.png ./binary-thresholded-image.png
```
