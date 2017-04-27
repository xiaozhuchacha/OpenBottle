//
// Created by mark on 4/21/17.
//

#ifndef UTIL_H
#define UTIL_H

#include <unordered_map>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

#include <dirent.h>

#include <boost/algorithm/string/classification.hpp> // Include boost::for is_any_of
#include <boost/algorithm/string/split.hpp> // Include for boost::split
#include <boost/filesystem.hpp>

std::vector<std::string> collect_paths_of_filetype_from_dir(std::string dir, std::string extension);

std::unordered_map<std::string, int> build_annotation_mapping(std::string mapping_path);

std::string extract_trial_id_from_path(std::string path);

#endif //UTIL_H
