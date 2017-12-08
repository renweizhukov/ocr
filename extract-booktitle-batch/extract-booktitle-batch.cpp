/*
 * extract-booktitle-batch.cpp
 *
 *  Created on: Oct 18, 2017
 *      Author: renwei
 */

#include <sys/types.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <boost/program_options.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
namespace po = boost::program_options;

int GetDirFiles(const string& dir, vector<string>& files)
{
    DIR *dp = nullptr;
    struct dirent *dirp = nullptr;

    dp = opendir(dir.c_str());
    if (dp == nullptr)
    {
        printf("Error: opendir(%s) opening %s.\n", strerror(errno), dir.c_str());
        return errno;
    }

    struct stat info;

    files.clear();
    string completedDir(dir);
    if (completedDir.back() != '/')
    {
        completedDir.push_back('/');
    }

    while ((dirp = readdir(dp)) != nullptr)
    {
        string fileFullPath = completedDir + string(dirp->d_name);

        if (stat(fileFullPath.c_str(), &info) != 0)
        {
            printf("Error: stat(%s) for %s.\n", strerror(errno), dirp->d_name);
            continue;
        }

        if (S_ISREG(info.st_mode))
        {
            files.push_back(fileFullPath);
        }
    }

    closedir(dp);
    return 0;
}

Mat PreprocessImg(const Mat& srcImg)
{
    // Sharpen the image using Unsharp Masking with a Gaussian blurred version of the image. Note that
    // srcImgGaussian = 1.5*srcImg - 0.5*srcImgGaussian, but to avoid overflow while multiplying srcImg
    // by 1.5, we subtract srcImgGaussian from srcImg first and then add 0.5*srcImg.
    Mat imgSharp;
    GaussianBlur(srcImg, imgSharp, Size(0, 0), 3);
    addWeighted(srcImg, 1.0, imgSharp, -0.5, 0.0, imgSharp);
    addWeighted(srcImg, 0.5, imgSharp, 1.0, 0.0, imgSharp);

    return imgSharp;
}

Point GetTemplateMatchingPoint(
    const Mat& srcImg,
    const Mat& templImg,
    Mat& result)
{
    // Create the result matrix.
    const int resultRows = srcImg.rows - templImg.rows + 1;
    const int resultCols =  srcImg.cols - templImg.cols + 1;

    result.create(resultRows, resultCols, CV_32FC1);

    // Do the Template Matching and Normalize.
    matchTemplate(srcImg, templImg, result, TM_CCOEFF_NORMED);
    normalize(result, result, 0, 1, NORM_MINMAX);

    // Localize the best match with minMaxLoc.
    double maxVal;
    Point maxLoc;

    minMaxLoc(result, nullptr, &maxVal, nullptr, &maxLoc);

    // For CCOEFF_NORMED, the best match is the maximum value.
    // Note that since result has been normalized into the range [0, 1], the maximum value is 1.
    return maxLoc;
}

void ExtractTitleViaTemplateMatching(
    const Mat& titleImgSobel,
    const vector<Mat>& bookCoverImgSobels,
    const vector<Mat>& bookCoverImgs,
    vector<Mat>& croppedTitleImgs)
{
    for (size_t imgIndex = 0; imgIndex < bookCoverImgSobels.size(); ++imgIndex)
    {
        // Do the template matching and find the best match point.
        Mat result;
        Point matchPoint = GetTemplateMatchingPoint(bookCoverImgSobels[imgIndex], titleImgSobel, result);

        // Crop the patch of the source image which best matches the template image.
        Mat croppedImg = bookCoverImgs[imgIndex](Rect(matchPoint.x, matchPoint.y, titleImgSobel.cols, titleImgSobel.rows));

        croppedTitleImgs.push_back(croppedImg);
    }
}

void ExtractTitleViaHomography(
    const Mat& titleImg,
    const vector<Mat>& bookCoverImgs,
    vector<Mat>& croppedTitleImgs)
{
    BFMatcher bfMatcher;
    vector<DMatch> matches;

    // Compute the keypoints and the descriptors of titleImg.
    const int minHessian = 400;
    Ptr<SurfFeatureDetector> detector = SURF::create(minHessian);

    vector<KeyPoint> titleImgKeyPoints;
    Mat titleImgDescriptors;
    detector->detectAndCompute(titleImg, noArray(), titleImgKeyPoints, titleImgDescriptors);

    // List the four corners of the title image clockwisely.
    vector<Point2f> titleImgCorners(4);
    titleImgCorners[0] = Point2f(0, 0);                                 // top-left corner
    titleImgCorners[1] = Point2f(titleImg.cols - 1, 0);                 // top-right corner
    titleImgCorners[2] = Point2f(titleImg.cols - 1, titleImg.rows - 1); // bottom-right corner
    titleImgCorners[3] = Point2f(0, titleImg.rows - 1);                 // bottom-left corner

    for (const Mat& bookCoverImg : bookCoverImgs)
    {
        // Compute the keypoints and the descriptors of bookCoverImg.
        vector<KeyPoint> bookCoverImgKeyPoints;
        Mat bookCoverImgDescriptors;
        detector->detectAndCompute(bookCoverImg, noArray(), bookCoverImgKeyPoints, bookCoverImgDescriptors);

        // Use the brute-force matcher to find the matched descriptors for all the descriptors
        // of titleImg.
        matches.clear();
        bfMatcher.match(titleImgDescriptors, bookCoverImgDescriptors, matches);

        // Sort the matches based on the distance and filter out the first few "good" matches to
        // find the homography.
        sort(matches.begin(), matches.end());

        size_t cntGoodMatches = min(static_cast<size_t>(50), static_cast<size_t>(matches.size()*0.3));
        if (cntGoodMatches < 5)
        {
            printf("[ERROR]: Unable to find enough (%ld < 5) good matches for computing the homography.\n\n",
                cntGoodMatches);
            continue;
        }

        vector<DMatch> goodMatches(matches.begin(), matches.begin() + cntGoodMatches);

        // Find the homography and do the perspective transformation.
        vector<Point2f> titleImgPoints;
        vector<Point2f> bookCoverPoints;
        for (size_t goodMatchIndex = 0; goodMatchIndex < cntGoodMatches; ++goodMatchIndex)
        {
            titleImgPoints.push_back(titleImgKeyPoints[goodMatches[goodMatchIndex].queryIdx].pt);
            bookCoverPoints.push_back(bookCoverImgKeyPoints[goodMatches[goodMatchIndex].trainIdx].pt);
        }

        Mat homo = findHomography(titleImgPoints, bookCoverPoints, RANSAC);

        vector<Point2f> bookCoverCorners(4);
        perspectiveTransform(titleImgCorners, bookCoverCorners, homo);

        Rect rect = boundingRect(bookCoverCorners);
        croppedTitleImgs.push_back(bookCoverImg(rect));
    }
}

void SegmentFullFilename(
    const string& fullFilename,
    string& dir,
    string& filename,
    string& extension)
{
    dir = "./";
    filename = fullFilename;

    size_t posLastSlash = fullFilename.find_last_of('/');
    size_t posLastDot = fullFilename.find_last_of('.');
    if (posLastSlash != string::npos)
    {
        dir = fullFilename.substr(0, posLastSlash + 1); // Note that slash is included.
        if (posLastDot > posLastSlash)
        {
            filename = fullFilename.substr(posLastSlash + 1, posLastDot - posLastSlash - 1);
            extension = fullFilename.substr(posLastDot);  // Note that dot is included.
        }
        else
        {
            filename = fullFilename.substr(posLastSlash + 1);
        }
    }
}

int main(int argc, char** argv)
{
    po::options_description opt("Options");
    opt.add_options()
        ("titleImg,i", po::value<string>()->required(), "The baseline book title image")
        ("imgDir,d", po::value<string>()->required(), "The directory containing all the book cover images")
        ("help,h", "Display the help information")
        ("method,m", po::value<string>(), "The method (homo | templ) of extracting the book title from its cover. If not specified, default homo.")
        ("outputDir,o", po::value<string>()->required(), "The output directory containing the title images extracted from the book cover images.");

    po::variables_map vm;
    try
    {
        po::store(po::command_line_parser(argc, argv).options(opt).run(), vm);

        if (vm.count("help") > 0)
        {
            printf("Usage: ./extract-booktitle-batch -i [title-image] -d [image-dir] -o [output-dir] -m [extract-method (homo|templ)]\n\n");
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
    string outputImgDir;
    string extractMethod;

    titleImgFile = vm["titleImg"].as<string>();
    bookCoverImgDir = vm["imgDir"].as<string>();
    outputImgDir = vm["outputDir"].as<string>();

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

    titleImg = PreprocessImg(titleImg);

    // Get all the image file names in the given directory.
    vector<string> bookCoverImgFiles;
    int error = GetDirFiles(bookCoverImgDir, bookCoverImgFiles);
    if (error != 0)
    {
        printf("[ERROR]: Cannot get the image file names in %s with error = %d", bookCoverImgDir.c_str(), error);
        return error;
    }

    vector<Mat> bookCoverImgs;
    for (const auto& imgFile: bookCoverImgFiles)
    {
        Mat img = imread(imgFile, IMREAD_COLOR);
        if (img.empty())
        {
            printf("[ERROR]: Cannot load image %s.\n\n", imgFile.c_str());
            return -1;
        }

        bookCoverImgs.push_back(PreprocessImg(img));
    }

    printf("[INFO]: Load %ld images of book covers.\n", bookCoverImgs.size());

    // Get the Sobel derivative of images.
    Mat titleImgSobel;
    Sobel(titleImg, titleImgSobel, CV_32F, 1, 1);

    vector<Mat> bookCoverImgSobels;
    for (const auto& img: bookCoverImgs)
    {
        Mat bookCoverImgSobel;
        Sobel(img, bookCoverImgSobel, CV_32F, 1, 1);
        bookCoverImgSobels.push_back(bookCoverImgSobel);
    }

    vector<Mat> croppedTitleImgs;
    if (extractMethod == "homo")
    {
        printf("[INFO]: Crop the book cover images to get the titles via homography.\n");
        ExtractTitleViaHomography(
            titleImg,
            bookCoverImgs,
            croppedTitleImgs);
    }
    else if (extractMethod == "templ")
    {
        printf("[INFO]: Crop the book cover images to get the titles via template matching.\n");
        // Do the template matching, find the best match point, and then crop the patch of the book cover
        // images which best matches the template image.
        ExtractTitleViaTemplateMatching(
            titleImgSobel,
            bookCoverImgSobels,
            bookCoverImgs,
            croppedTitleImgs);
    } else {
        printf("[ERROR]: Unsupported extract method = %s.\n\n", extractMethod.c_str());
        return -1;
    }

    for (size_t imgIndex = 0; imgIndex < bookCoverImgSobels.size(); ++imgIndex)
    {
        if (croppedTitleImgs[imgIndex].empty())
        {
            printf("[ERROR]: Failed to crop the title from the image %s.\n\n", bookCoverImgFiles[imgIndex].c_str());
            continue;
        }

        // Write the cropped patch into an image file.
        string dir;
        string filename;
        string extension;
        SegmentFullFilename(bookCoverImgFiles[imgIndex], dir, filename, extension);

        string croppedImgFile = outputImgDir + '/' + filename + "_title" + extension;
        bool writeRes = imwrite(croppedImgFile, croppedTitleImgs[imgIndex]);
        if (writeRes)
        {
            printf("[INFO]: Successfully write the cropped title image into %s.\n", croppedImgFile.c_str());
        }
        else
        {
            printf("[ERROR]: Failed to write the cropped title image into %s.\n\n", croppedImgFile.c_str());
            return -1;
        }
    }

    return 0;
}
