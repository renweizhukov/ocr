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
