/*
 * main.cpp
 *
 *  Created on: Dec 6, 2017
 *      Author: renwei
 */

#include <memory>

#include "Utility.h"
#include "OcrPreprocessor.h"
#include "CircledDigitsOCRer.h"

using namespace std;
using namespace cv;
namespace po = boost::program_options;

static void write(
    FileStorage& fs,
    const string&,
    const OcrResult& ocrResult)
{
    ocrResult.write(fs);
}

static void read(
    const FileNode& node,
    OcrResult& ocrResult,
    const OcrResult& defaultValue = OcrResult())
{
    if (node.empty())
    {
        ocrResult = defaultValue;
    }
    else
    {
        ocrResult.read(node);
    }
}

int main(int argc, char** argv)
{
    po::options_description opt("Options");
    opt.add_options()
        ("imgDir,d", po::value<string>()->required(), "The directory containing all the book cover images")
        ("help,h", "Display the help information")
        ("titleImg,i", po::value<string>()->required(), "The baseline book series title image")
        ("method,m", po::value<string>(), "The method (homo | templ | hough) of extracting the book title from its cover. If not specified, default homo.")
        ("outputDir,o", po::value<string>()->required(), "The output directory containing the images of circled digits extracted from the book cover images and the OCR results.")
        ("templImgDir,t", po::value<string>()->required(), "The directory containing all the template images for OCRing circled digits");

    po::variables_map vm;
    try
    {
        po::store(po::command_line_parser(argc, argv).options(opt).run(), vm);

        if (vm.count("help") > 0)
        {
            printf("Usage: ./ocr-circled-digits-batch -i [title-image] -d [image-dir] -o [output-dir] -m [extract-method (homo|templ|hough)]\n\n");
            cout << opt << endl;
            return 0;
        }

        po::notify(vm);
    }
    catch (po::error& e)
    {
        cerr << "[ERROR]: " << e.what() << endl << endl;
        cout << opt << endl;
        return -1;
    }

    string titleImgFile;
    string bookCoverImgDir;
    string templImgDir;
    string outputDir;
    string extractMethod;

    titleImgFile = vm["titleImg"].as<string>();
    bookCoverImgDir = vm["imgDir"].as<string>();
    templImgDir = vm["templImgDir"].as<string>();
    outputDir = vm["outputDir"].as<string>();

    if (vm.count("method") > 0)
    {
        extractMethod = vm["method"].as<string>();
        transform(extractMethod.begin(), extractMethod.end(), extractMethod.begin(), ::tolower);
    }
    else
    {
        extractMethod = "homo";
        printf("[INFO]: No extract method is specified and use the default method homography.\n");
    }

    Mat titleImg = imread(titleImgFile, IMREAD_COLOR);
    if (titleImg.empty())
    {
        printf("[ERROR]: Cannot load image %s.\n\n", titleImgFile.c_str());
        return -1;
    }

    // Create the OcrPreprocessor based on the extraction method.
    unique_ptr<OcrPreprocessor> preprocessor;
    if ((extractMethod == "homo") || (extractMethod == "templ"))
    {
        const int centerDisplacementX = 0;
        const int centerDisplacementY = 55;
        const unsigned int width = 80;
        const unsigned int height = 60;

        preprocessor.reset(new OcrPreprocessor(
            extractMethod,
            titleImg,
            centerDisplacementX,
            centerDisplacementY,
            width,
            height));
    }
    else if (extractMethod == "hough")
    {
        const unsigned int minRadius = 10;
        const unsigned int maxRadius = 30;

        preprocessor.reset(new OcrPreprocessor(
            extractMethod,
            minRadius,
            maxRadius));
    }
    else
    {
        printf("[ERROR]: Unsupported extraction method %s.\n\n", extractMethod.c_str());
        return -1;
    }

    // Get all the template image file names in the given directory.
    vector<string> templImgFiles;
    int error = Utility::GetDirFiles(templImgDir, templImgFiles);
    if (error != 0)
    {
        printf("[ERROR]: Cannot get the template image file names in %s with error = %d", templImgDir.c_str(), error);
        return error;
    }

    sort(templImgFiles.begin(), templImgFiles.end());

    // Load the template images.
    vector<pair<string, Mat> > templDigitImgPairs;
    for (const auto& imgFile: templImgFiles)
    {
        // Note that each template image is named after the digits displayed inside.
        string dir;
        string digits;
        string extension;
        Utility::SegmentFullFilename(imgFile, dir, digits, extension);

        Mat img = imread(imgFile, IMREAD_COLOR);
        if (img.empty())
        {
            printf("[ERROR]: Cannot load template image %s.\n\n", imgFile.c_str());
            continue;
        }

#ifdef DEBUG
        printf("[DEBUG]: The pixel data type of the template image %s is %s.\n",
            imgFile.c_str(), Utility::CvType2Str(img.type()).c_str());
#endif

        // The pixel data type of the loaded template image is usually CV_8UC3,
        // so we need to convert it into gray scale (i.e., CV_8UC1) before doing
        // the template matching.
        Mat grayImg;
        cvtColor(img, grayImg, COLOR_BGR2GRAY);

#ifdef DEBUG
        printf("[DEBUG]: The pixel data type of the gray-scale template image %s is %s.\n",
            imgFile.c_str(), Utility::CvType2Str(grayImg.type()).c_str());
#endif

        templDigitImgPairs.push_back(make_pair(digits, grayImg));
    }

    // Create the CircledDigitsOCRer based on the template matching.
    unique_ptr<CircledDigitsOCRer> ocrer(new CircledDigitsOCRer(templDigitImgPairs));

    // Get all the image file names in the given directory.
    vector<string> bookCoverImgFiles;
    error = Utility::GetDirFiles(bookCoverImgDir, bookCoverImgFiles);
    if (error != 0)
    {
        printf("[ERROR]: Cannot get the image file names in %s with error = %d", bookCoverImgDir.c_str(), error);
        return error;
    }

    sort(bookCoverImgFiles.begin(), bookCoverImgFiles.end());

    vector<OcrResult> ocrResults;
    for (const auto& imgFile: bookCoverImgFiles)
    {
        Mat img = imread(imgFile, IMREAD_COLOR);
        if (img.empty())
        {
            printf("[ERROR]: Cannot load image %s.\n\n", imgFile.c_str());
            continue;
        }

#ifdef DEBUG
        printf("[DEBUG]: The pixel data type of the book cover image %s is %s.\n",
            imgFile.c_str(), Utility::CvType2Str(img.type()).c_str());
#endif

        Mat circledDigitsImg = preprocessor->ExtractCircledDigits(img);
        if (circledDigitsImg.empty())
        {
            printf("[ERROR]: Can't find the circled digits in %s.\n\n", imgFile.c_str());
            continue;
        }

        Mat blackWhiteImg = preprocessor->BlackWhiteThresholding(4.0, circledDigitsImg);
#ifdef DEBUG
        printf("[DEBUG]: The pixel data type of the preprocessed black-white book cover image %s is %s.\n",
            imgFile.c_str(), Utility::CvType2Str(blackWhiteImg.type()).c_str());
#endif

        // Write the cropped image of circled digits into an image file.
        string dir;
        string filename;
        string extension;
        Utility::SegmentFullFilename(imgFile, dir, filename, extension);

        string blackWhiteImgFile = outputDir + '/' + filename + "_circledDigits" + extension;
        bool writeRes = imwrite(blackWhiteImgFile, blackWhiteImg);
        if (writeRes)
        {
            printf("[INFO]: Successfully write the cropped black-white image of circled digits into %s.\n",
                blackWhiteImgFile.c_str());
        }
        else
        {
            printf("[ERROR]: Failed to write the cropped black-white image of circled digits into %s.\n\n",
                blackWhiteImgFile.c_str());
            return -1;
        }

        // Use CircledDigitsOCRer to recognize the digits from the cropped image.
        OcrResult res;
        ocrer->OCR(blackWhiteImg, res);

        printf("[INFO]: The digits in image %s are %s.\n", imgFile.c_str(), res.evaluatedDigits.c_str());

        ocrResults.push_back(res);
    }

    // Write results to a yml file.
    string ocrResultFile = outputDir + "/OcrResult.yml";
    FileStorage fsResult(ocrResultFile, FileStorage::WRITE);

    printf("[INFO]: Writing OCR results to %s.\n", ocrResultFile.c_str());
    for (int resultIndex = 0; resultIndex < static_cast<int>(ocrResults.size()); ++resultIndex)
    {
        // Key names must start with a letter or '_'. Since the image filename may start with a non-letter,
        // e.g., a digit, we don't use the image filename as the key name.
        fsResult << "imgfilename_" + to_string(resultIndex) << bookCoverImgFiles[resultIndex];
        fsResult << "ocrresult_" + to_string(resultIndex) << ocrResults[resultIndex];
    }

    fsResult.release();

    return 0;
}
