/*
 * OcrPreprocessor.cpp
 *
 *  Created on: Dec 6, 2017
 *      Author: renwei
 */

#include "OcrPreprocessor.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

OcrPreprocessor::OcrPreprocessor(
    const string& method,
    const Mat& titleImg,
    const int centerDisplacementX,
    const int centerDisplacementY,
    const unsigned int width,
    const unsigned int height) :
    m_titleImg(titleImg),
    m_centerDisplacementX(centerDisplacementX),
    m_centerDisplacementY(centerDisplacementY),
    m_width(width),
    m_height(height)
{
    m_method = Str2ExtractMethod(method);
    if ((m_method != ExtractMethod::Homography) && (m_method != ExtractMethod::TemplateMatching))
    {
        printf("[ERROR]: Incorrect constructor for method %s.\n\n", method.c_str());
        return;
    }

    m_titleImg = SharpenImg(titleImg);
    if (m_method == ExtractMethod::Homography)
    {
        m_matcher = BFMatcher::create();

        // Compute the keypoints and the descriptors of titleImg.
        const int minHessian = 400;
        m_detector = SURF::create(minHessian);

        m_detector->detectAndCompute(titleImg, noArray(), m_titleImgKeyPoints, m_titleImgDescriptors);

        // List the four corners of the title image clockwisely.
        m_titleImgCorners.resize(4);
        m_titleImgCorners[0] = Point2f(0, 0);                                     // top-left corner
        m_titleImgCorners[1] = Point2f(m_titleImg.cols - 1, 0);                   // top-right corner
        m_titleImgCorners[2] = Point2f(m_titleImg.cols - 1, m_titleImg.rows - 1); // bottom-right corner
        m_titleImgCorners[3] = Point2f(0, m_titleImg.rows - 1);                   // bottom-left corner
    }
    else if (m_method == ExtractMethod::TemplateMatching)
    {
        Sobel(m_titleImg, m_titleImgSobel, CV_32F, 1, 1);
    }
}

OcrPreprocessor::OcrPreprocessor(
    const string& method,
    const unsigned int minRadius,
    const unsigned int maxRadius) :
    m_minRadius(minRadius),
    m_maxRadius(maxRadius)
{
    m_method = Str2ExtractMethod(method);
    if (m_method != ExtractMethod::HoughCircleTransform)
    {
        printf("[ERROR]: Incorrect constructor for method %s.\n\n", method.c_str());
    }
}

OcrPreprocessor::~OcrPreprocessor()
{

}

Mat OcrPreprocessor::ExtractCircledDigits(const Mat& bookCoverImg)
{
    // Sharpen the book cover image.
    Mat sharpenedBookCoverImg = SharpenImg(bookCoverImg);

    Mat circledDigitsImg;
    switch (m_method)
    {
    case ExtractMethod::Homography:
        circledDigitsImg = ExtractCircledDigitsViaHomography(sharpenedBookCoverImg);
        break;
    case ExtractMethod::TemplateMatching:
        circledDigitsImg = ExtractCircledDigitsViaTemplateMatching(sharpenedBookCoverImg);
        break;

    case ExtractMethod::HoughCircleTransform:
        circledDigitsImg = ExtractCircledDigitsViaHoughTransform(sharpenedBookCoverImg);
        break;

    default:
        printf("[ERROR] Unsupported extraction method %s.\n\n",
            ExtractMethod2Str(m_method).c_str());
        break;
    }

    return circledDigitsImg;
}

Mat OcrPreprocessor::BlackWhiteThresholding(
    const double scaleFactor,
    const Mat& circledDigitsImg)
{
    Mat blackWhiteImg;

    // TODO

    return blackWhiteImg;
}

string OcrPreprocessor::ExtractMethod2Str(const ExtractMethod method)
{
    switch (method)
    {
    case ExtractMethod::None:
        return "none";

    case ExtractMethod::Homography:
        return "homo";

    case ExtractMethod::TemplateMatching:
        return "templ";

    case ExtractMethod::HoughCircleTransform:
        return "hough";

    default:
        return "invalid";
    }
}

OcrPreprocessor::ExtractMethod OcrPreprocessor::Str2ExtractMethod(const string& str)
{
    // Convert all letters into small cases if they are not.
    string lowerStr(str);
    transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);

    if (lowerStr == "homo")
    {
        return ExtractMethod::Homography;
    }
    else if (lowerStr == "templ")
    {
        return ExtractMethod::TemplateMatching;
    }
    else if (lowerStr == "hough")
    {
        return ExtractMethod::HoughCircleTransform;
    }
    else
    {
        return ExtractMethod::None;
    }
}

Mat OcrPreprocessor::SharpenImg(const Mat& img)
{
    // Sharpen the image using Unsharp Masking with a Gaussian blurred version of the image. Note that
    // srcImgGaussian = 1.5*srcImg - 0.5*srcImgGaussian, but to avoid overflow while multiplying srcImg
    // by 1.5, we subtract srcImgGaussian from srcImg first and then add 0.5*srcImg.
    Mat res;
    GaussianBlur(img, res, Size(0, 0), 3);
    addWeighted(img, 1.0, res, -0.5, 0.0, res);
    addWeighted(img, 0.5, res, 1.0, 0.0, res);

    return res;
}

Point OcrPreprocessor::GetTemplateMatchingPoint(
    const Mat& srcImg,
    const Mat& templImg,
    OutputArray result)
{
    // Create the result matrix.
    const int resultRows = srcImg.rows - templImg.rows + 1;
    const int resultCols =  srcImg.cols - templImg.cols + 1;

    Mat tmpResult(resultRows, resultCols, CV_32FC1);

    // Do the Template Matching and Normalize.
    matchTemplate(srcImg, templImg, tmpResult, TM_CCOEFF_NORMED);

    // Localize the best match with minMaxLoc.
    double maxVal;
    Point maxLoc;

    minMaxLoc(tmpResult, nullptr, &maxVal, nullptr, &maxLoc);

    if (result.needed())
    {
        tmpResult.copyTo(result);
    }

    // For CCOEFF_NORMED, the best match is the maximum value.
    // Note that since result has been normalized into the range [0, 1], the maximum value is 1.
    return maxLoc;
}

Rect OcrPreprocessor::ShiftAndResizeRect(
    const int topLeftX,
    const int topLeftY)
{
    // Calculate the center point of the matched rectangle.
    Point matchCenter((topLeftX*2 + m_titleImg.cols)/2, (topLeftY*2 + m_titleImg.rows)/2);

    // Move the center of the matched rectangle according to (m_centerDisplacementX, m_centerDisplacementY).
    Point circledDigitsImgCenter(matchCenter.x + m_centerDisplacementX, matchCenter.y + m_centerDisplacementY);

    // Calculate the top-left point of the rectangle which contains the circled digits.
    Point circledDigitsImgTopLeft(circledDigitsImgCenter.x - m_width/2, circledDigitsImgCenter.y - m_height/2);

    return Rect(circledDigitsImgTopLeft.x, circledDigitsImgTopLeft.y, m_width, m_height);
}

Mat OcrPreprocessor::ExtractCircledDigitsViaTemplateMatching(
    const Mat& bookCoverImg)
{
    Mat bookCoverImgSobel;
    Sobel(bookCoverImg, bookCoverImgSobel, CV_32F, 1, 1);

    Point matchPoint = GetTemplateMatchingPoint(bookCoverImgSobel, m_titleImgSobel, noArray());

    // Shift and resize the rectangle such that it will contain the circled digits.
    Rect circledDigitsRect = ShiftAndResizeRect(matchPoint.x, matchPoint.y);

    // Crop the patch of the source image which contains the circled digits.
    Mat circledDigitsImg = bookCoverImg(circledDigitsRect);
    return circledDigitsImg;
}

Mat OcrPreprocessor::ExtractCircledDigitsViaHomography(
    const Mat& bookCoverImg)
{
    Mat circledDigitsImg;

    // Compute the keypoints and the descriptors of bookCoverImg.
    vector<KeyPoint> bookCoverImgKeyPoints;
    Mat bookCoverImgDescriptors;
    m_detector->detectAndCompute(bookCoverImg, noArray(), bookCoverImgKeyPoints, bookCoverImgDescriptors);

    // Use the brute-force matcher to find the matched descriptors for all the descriptors
    // of titleImg.
    vector<DMatch> matches;
    m_matcher->match(m_titleImgDescriptors, bookCoverImgDescriptors, matches);

    // Sort the matches based on the distance and filter out the first few "good" matches to
    // find the homography.
    sort(matches.begin(), matches.end());

    size_t cntGoodMatches = min(static_cast<size_t>(50), static_cast<size_t>(matches.size()*0.3));
    if (cntGoodMatches < 5)
    {
        printf("[ERROR]: Unable to find enough (%ld < 5) good matches for computing the homography.\n\n",
            cntGoodMatches);
        return circledDigitsImg;
    }

    vector<DMatch> goodMatches(matches.begin(), matches.begin() + cntGoodMatches);

    // Find the homography and do the perspective transformation.
    vector<Point2f> titleImgPoints;
    vector<Point2f> bookCoverPoints;
    for (size_t goodMatchIndex = 0; goodMatchIndex < cntGoodMatches; ++goodMatchIndex)
    {
        titleImgPoints.push_back(m_titleImgKeyPoints[goodMatches[goodMatchIndex].queryIdx].pt);
        bookCoverPoints.push_back(bookCoverImgKeyPoints[goodMatches[goodMatchIndex].trainIdx].pt);
    }

    Mat homo = findHomography(titleImgPoints, bookCoverPoints, RANSAC);

    vector<Point2f> bookCoverCorners(4);
    perspectiveTransform(m_titleImgCorners, bookCoverCorners, homo);

    Rect matchRect = boundingRect(bookCoverCorners);

    // Shift and resize the rectangle such that it will contain the circled digits.
    Rect circledDigitsRect = ShiftAndResizeRect(matchRect.x, matchRect.y);

    // Crop the patch of the source image which contains the circled digits.
    circledDigitsImg = bookCoverImg(circledDigitsRect);
    return circledDigitsImg;
}

Mat OcrPreprocessor::ExtractCircledDigitsViaHoughTransform(
    const Mat& bookCoverImg)
{
    // Convert the BGR image into grayscale.

    Mat circledDigitsImg;



    return circledDigitsImg;
}
