/*
 * OcrPreprocessor.h
 *
 *  Created on: Dec 6, 2017
 *      Author: renwei
 */

#ifndef INCLUDES_OCRPREPROCESSOR_H_
#define INCLUDES_OCRPREPROCESSOR_H_

#include <cstdio>
#include <string>
#include <algorithm>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

class OcrPreprocessor
{
private:
    enum class ExtractMethod {
        None,
        Homography,
        TemplateMatching,
        HoughCircleTransform
    };

    std::string ExtractMethod2Str(const ExtractMethod method);
    ExtractMethod Str2ExtractMethod(const std::string& str);

    ExtractMethod m_method;
    cv::Mat m_titleImg;
    cv::Mat m_titleImgSobel; // The Sobel derivative of the title image

    // The circled digits may be outside the series title. After we find
    // the rectangular region of the series title in the book cover, we
    // move the center of the rectangle according to
    // (m_centerDisplacementX, m_centerDisplacementY) and resize the rectangle
    // according to (m_width, m_height). Note that the following four parameters
    // are NOT used in the Hough Circle Transform.
    int m_centerDisplacementX;
    int m_centerDisplacementY;
    unsigned int m_width;
    unsigned int m_height;

    cv::Ptr<cv::BFMatcher> m_matcher;
    cv::Ptr<cv::xfeatures2d::SurfFeatureDetector> m_detector;
    std::vector<cv::KeyPoint> m_titleImgKeyPoints;
    cv::Mat m_titleImgDescriptors;
    std::vector<cv::Point2f> m_titleImgCorners;

    // The minimum and maximum radius to consider in the Hough Circle Transform
    unsigned int m_minRadius;
    unsigned int m_maxRadius;

    cv::Mat SharpenImg(const cv::Mat& img);

    cv::Point GetTemplateMatchingPoint(
        const cv::Mat& srcImg,
        const cv::Mat& templImg,
        cv::OutputArray result);

    cv::Rect ShiftAndResizeRect(
        const int topLeftX,
        const int topLeftY);

    cv::Mat ExtractCircledDigitsViaTemplateMatching(
        const cv::Mat& bookCoverImg);

    cv::Mat ExtractCircledDigitsViaHomography(
        const cv::Mat& bookCoverImg);

    cv::Mat ExtractCircledDigitsViaHoughTransform(
        const cv::Mat& bookCoverImg);

public:

    // Constructor for the extraction methods of Template Matching and Homography
    OcrPreprocessor(
        const std::string& method,
        const cv::Mat& titleImg,
        const int centerDisplacementX = 0,
        const int centerDisplacementY = 0,
        const unsigned int width = 0,
        const unsigned int height = 0);

    // Constructor for the extraction method of Hough Circle Transform
    OcrPreprocessor(
        const std::string& method,
        const unsigned int minRadius,
        const unsigned int maxRadius);

    ~OcrPreprocessor();

    cv::Mat ExtractCircledDigits(const cv::Mat& bookCoverImg);
    cv::Mat BlackWhiteThresholding(
        const double scaleFactor,
        const cv::Mat& circledDigitsImg);
};

#endif /* INCLUDES_OCRPREPROCESSOR_H_ */
