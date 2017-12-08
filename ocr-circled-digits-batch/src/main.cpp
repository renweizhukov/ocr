/*
 * main.cpp
 *
 *  Created on: Dec 6, 2017
 *      Author: renwei
 */

#include <memory>

#include "Utility.h"
#include "OcrPreprocessor.h"

using namespace std;
using namespace cv;
namespace po = boost::program_options;

int main(int argc, char** argv)
{
    po::options_description opt("Options");
    opt.add_options()
        ("titleImg,i", po::value<string>()->required(), "The baseline book series title image")
        ("imgDir,d", po::value<string>()->required(), "The directory containing all the book cover images")
        ("help,h", "Display the help information")
        ("method,m", po::value<string>(), "The method (homo | templ | hough) of extracting the book title from its cover. If not specified, default homo.")
        ("outputDir,o", po::value<string>()->required(), "The output directory containing the images of circled digits extracted from the book cover images and the OCR results.");

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
    string outputDir;
    string extractMethod;

    titleImgFile = vm["titleImg"].as<string>();
    bookCoverImgDir = vm["imgDir"].as<string>();
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

    // Get all the image file names in the given directory.
    vector<string> bookCoverImgFiles;
    int error = Utility::GetDirFiles(bookCoverImgDir, bookCoverImgFiles);
    if (error != 0)
    {
        printf("[ERROR]: Cannot get the image file names in %s with error = %d", bookCoverImgDir.c_str(), error);
        return error;
    }

    for (const auto& imgFile: bookCoverImgFiles)
    {
        Mat img = imread(imgFile, IMREAD_COLOR);
        if (img.empty())
        {
            printf("[ERROR]: Cannot load image %s.\n\n", imgFile.c_str());
            return -1;
        }

        Mat circledDigitsImg = preprocessor->ExtractCircledDigits(img);

        // Write the cropped image of circled digits into an image file.
        string dir;
        string filename;
        string extension;
        Utility::SegmentFullFilename(imgFile, dir, filename, extension);

        string circledDigitsImgFile = outputDir + '/' + filename + "_circledDigits" + extension;
        bool writeRes = imwrite(circledDigitsImgFile, circledDigitsImg);
        if (writeRes)
        {
            printf("[INFO]: Successfully write the cropped image of circled digits into %s.\n", circledDigitsImgFile.c_str());
        }
        else
        {
            printf("[ERROR]: Failed to write the cropped image of circled digits into %s.\n\n", circledDigitsImgFile.c_str());
            return -1;
        }
    }

    return 0;
}
