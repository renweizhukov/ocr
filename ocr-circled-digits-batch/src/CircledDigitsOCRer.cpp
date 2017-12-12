/*
 * CircledDigitsOCRer.cpp
 *
 *  Created on: Dec 10, 2017
 *      Author: renwei
 */

#include "CircledDigitsOCRer.h"

using namespace std;
using namespace cv;

// We assume that the pixel data type of the template images is CV_8UC1.
CircledDigitsOCRer::CircledDigitsOCRer(const vector<pair<string, Mat> >& templDigitImgPairs) :
    m_templDigitImgPairs(templDigitImgPairs)
{

}

// We also assume that the pixel data type of the input image is CV_8UC1, too.
void CircledDigitsOCRer::OCR(
    const Mat& circledDigitsImg,
    OcrResult& res)
{
    res.evaluatedDigits.clear();
    res.digits2MatchResMap.clear();

    double maxDigitMatchVal = -1.0;
    for (const auto& templDigitImgPair: m_templDigitImgPairs)
    {
        string digits = templDigitImgPair.first;
        Mat templImg = templDigitImgPair.second;

        Mat matchRes;

        const int matchResRows = circledDigitsImg.rows - templImg.rows + 1;
        const int matchResCols =  circledDigitsImg.cols - templImg.cols + 1;

        matchRes.create(matchResRows, matchResCols, CV_32FC1);

        matchTemplate(circledDigitsImg, templImg, matchRes, TM_CCOEFF_NORMED);

        // Localize the best match with minMaxLoc.
        double maxVal = -1.0;
        minMaxLoc(matchRes, nullptr, &maxVal, nullptr, nullptr);

        res.digits2MatchResMap.insert(make_pair(digits, maxVal));

        if (maxDigitMatchVal < maxVal)
        {
            maxDigitMatchVal = maxVal;
            res.evaluatedDigits = digits;
        }
    }
}
