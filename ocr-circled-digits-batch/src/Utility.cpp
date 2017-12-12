/*
 * Utility.cpp
 *
 *  Created on: Dec 6, 2017
 *      Author: renwei
 */

#include "Utility.h"

using namespace std;

int Utility::GetDirFiles(const string& dir, vector<string>& files)
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

void Utility::SegmentFullFilename(
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

// Courtesy of https://stackoverflow.com/questions/10167534/how-to-find-out-what-type-of-a-mat-object-is-with-mattype-in-opencv.
string Utility::CvType2Str(const int type)
{
    string typeStr("CV_");

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth)
    {
    case CV_8U:
        typeStr += "8U";
        break;

    case CV_8S:
        typeStr += "8S";
        break;

    case CV_16U:
        typeStr += "16U";
        break;

    case CV_16S:
        typeStr += "16S";
        break;

    case CV_32S:
        typeStr += "32S";
        break;

    case CV_32F:
        typeStr += "32F";
        break;

    case CV_64F:
        typeStr += "64F";
        break;

    default:
        typeStr += "User";
        break;
    }

    typeStr += "C";
    typeStr += (chans+'0');

    return typeStr;
}
