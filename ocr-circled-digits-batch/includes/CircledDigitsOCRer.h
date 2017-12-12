/*
 * CircledDigitsOCRer.h
 *
 *  Created on: Dec 10, 2017
 *      Author: renwei
 */

#ifndef INCLUDES_CIRCLEDDIGITSOCRER_H_
#define INCLUDES_CIRCLEDDIGITSOCRER_H_

#include <cstdio>
#include <string>
#include <vector>
#include <map>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

struct OcrResult
{
    std::string evaluatedDigits;
    std::map<std::string, float> digits2MatchResMap;

    OcrResult()
    {
    }

    // Write serialization for this class
    void write(cv::FileStorage& fs) const
    {
        fs << "{" << "evaluatedDigits" << evaluatedDigits;

        fs << "digits2MatchResMap" << "{";
        for (const auto& digits2MatchResPair: digits2MatchResMap)
        {
            // Key names must start with a letter or '_'. Since digits2MatchResPair.first
            // starts with a digit, we prefix it with "digit_".
            fs << "digit_" + digits2MatchResPair.first << digits2MatchResPair.second;
        }
        fs << "}";  // End of digits2MatchResMap.

        fs << "}";  // End of OcrResult.
    }

    // Read de-serialization for this class
    void read(const cv::FileNode& node)
    {
        evaluatedDigits = (std::string)(node["evaluatedDigits"]);

        digits2MatchResMap.clear();
        cv::FileNode mapNode = node["digits2MatchResMap"];
        for (auto itMapNode = mapNode.begin(); itMapNode != mapNode.end(); ++itMapNode)
        {
            cv::FileNode item = *itMapNode;
            std::string digits = item.name();
            float matchRes = (float)item;
            digits2MatchResMap.insert(std::make_pair(digits, matchRes));
        }
    }
};

class CircledDigitsOCRer
{
private:
    std::vector<std::pair<std::string, cv::Mat> > m_templDigitImgPairs;

public:
    CircledDigitsOCRer(const std::vector<std::pair<std::string, cv::Mat> >& templDigitImgPairs);

    void OCR(
        const cv::Mat& circledDigitsImg,
        OcrResult& res);
};

#endif /* INCLUDES_CIRCLEDDIGITSOCRER_H_ */
