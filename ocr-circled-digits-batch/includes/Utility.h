/*
 * Utility.h
 *
 *  Created on: Dec 6, 2017
 *      Author: renwei
 */

#ifndef INCLUDES_UTILITY_H_
#define INCLUDES_UTILITY_H_

#include <sys/types.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <boost/program_options.hpp>

class Utility
{
public:
    static int GetDirFiles(const std::string& dir, std::vector<std::string>& files);

    static void SegmentFullFilename(
        const std::string& fullFilename,
        std::string& dir,
        std::string& filename,
        std::string& extension);

    static std::string CvType2Str(const int type);
};

#endif /* INCLUDES_UTILITY_H_ */
