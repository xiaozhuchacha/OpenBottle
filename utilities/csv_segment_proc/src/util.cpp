//
// Created by mark on 4/21/17.
//

#include "util.h"


// helper function to open all csvs in a directory
bool has_suffix(const std::string& s, const std::string& suffix)
{
    return (s.size() >= suffix.size()) && std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

std::vector<std::string> collect_paths_of_filetype_from_dir(std::string dir_path, std::string extension){
  std::vector<std::string> file_paths;

  DIR *dir = opendir(dir_path.c_str());
  if (!dir) {
    std::stringstream ss;
    ss << "Unable to open directory: " << dir_path.c_str();
    throw std::exception(std::runtime_error(ss.str()));
  }

  dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    if (has_suffix(entry->d_name, "." + extension)) {
      file_paths.push_back(dir_path + "/" + entry->d_name);
    }
  }
  closedir(dir);

  return file_paths;
}

// constructs hash table mapping string names of actions to integers
std::unordered_map<std::string, int> build_annotation_mapping(std::string mapping_path){
  std::unordered_map<std::string, int> mapping;
  std::ifstream infile(mapping_path);
  std::string line;

  while(std::getline(infile, line)){
    std::vector<std::string> pairs;
    boost::split(pairs, line, boost::is_any_of(","), boost::token_compress_on);
    mapping.insert({pairs[0], std::stoi(pairs[1])-1}); // subtract 1 because number starts at 1
  }

  return mapping;
}

std::string extract_trial_id_from_path(std::string path){
  boost::filesystem::path p(path);
  std::string trial_id = p.filename().string();
  unsigned long underscore_idx = trial_id.find_first_of("_");
  trial_id = trial_id.substr(0,underscore_idx);
  return trial_id;
}
