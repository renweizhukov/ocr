/*
 * ocr-preprocessing.cpp
 *
 *  Created on: Sep 14, 2017
 *      Author: renwei
 */

#include <cstdio>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        printf("[ERROR]: Either the input image or the output image or both are not specified.\n\n");
        return -1;
    }

    Mat srcImg = imread(argv[1]);
    if (srcImg.empty())
    {
        printf("[ERROR]: Can't load the input image file %s.\n\n", argv[1]);
        return -1;
    }

    // Sharpen the image using Unsharp Masking with a Gaussian blurred version of the image. Note that
    // srcImgGaussian = 1.5*srcImg - 0.5*srcImgGaussian, but to avoid overflow while multiplying srcImg
    // by 1.5, we subtract srcImgGaussian from srcImg first and then add 0.5*srcImg.
    Mat imgSharp;
    GaussianBlur(srcImg, imgSharp, Size(0, 0), 3);
    addWeighted(srcImg, 1.0, imgSharp, -0.5, 0.0, imgSharp);
    addWeighted(srcImg, 0.5, imgSharp, 1.0, 0.0, imgSharp);

    // Convert the color image after Gaussian Blur into grayscale.
    Mat imgSharpGray;
    cvtColor(imgSharp, imgSharpGray, COLOR_BGR2GRAY);

    double minVal = 0.0;
    double maxVal = 0.0;
    minMaxLoc(imgSharpGray, &minVal, &maxVal);
    printf("[INFO]: The grayscale image after Unsharp Masking: minVal = %f, maxVal = %f.\n",
        minVal, maxVal);

    double thresh = minVal + 0.3*(maxVal - minVal);
    Mat imgThresholded;
    threshold(imgSharpGray, imgThresholded, thresh, 255, THRESH_BINARY_INV);

    // Display the thresholded image.
    namedWindow("The thresholded image", WINDOW_AUTOSIZE);
    imshow("The thresholded image", imgThresholded);

    // Write the thresholded image into the output file.
    bool writeRes = imwrite(argv[2], imgThresholded);
    if (writeRes)
    {
        printf("[INFO]: Successfully write the thresholded image into %s.\n", argv[2]);
    }
    else
    {
        printf("[ERROR]: Failed to write the thresholded image into %s.\n\n", argv[2]);
        return -1;
    }

    return 0;
}


