//
// Created by mark edmonds on 4/21/17.
//

#include <iostream>
#include <iterator>

#include <boost/program_options.hpp>
#include <trial_content.h>

#include "util.h"
#include "segment_parser.h"
#include "trial_content.h"
#include "window_manager.h"

int main(int argc, char *argv[]) {
  std::vector<std::string> input_csvs;
  std::unordered_map<std::string, int> annotation_mapping;
  SegmentParser segment_parser;
  TrialManager trial_manager;
  std::string output_dir;

  try {
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("csv", boost::program_options::value<std::vector<std::string>>(), "input csv file")
        ("csv-dir", boost::program_options::value<std::vector<std::string>>(), "input directory for csvs")
        ("segment-dir", boost::program_options::value<std::vector<std::string>>(), "input directory for segments. A segment refers to a single trial (a sentence in the grammar). Each trial contains multiple actions")
        ("annotation-map", boost::program_options::value<std::string>(),
         "annotation mapping file, maps strings to integers for each annotation label")
        ("output-dir", boost::program_options::value<std::string>(),
         "output directory of action files");

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 0;
    }

    // add all input files
    if (vm.count("csv")) {
      std::vector<std::string> file_paths = vm["csv"].as<std::vector<std::string>>();
      for (auto &it : file_paths) {
        input_csvs.push_back(it);
      }
    }

    // add all csvs in all paths
    if (vm.count("csv-dir")) {
      std::vector<std::string> dir_paths = vm["csv-dir"].as<std::vector<std::string>>();
      for (auto &it: dir_paths) {
        std::vector<std::string> new_csvs = collect_paths_of_filetype_from_dir(it, "csv");
        input_csvs.insert(input_csvs.end(), new_csvs.begin(), new_csvs.end());
      }
    }

    // load annotation mapping file
    if (vm.count("annotation-map")) {
      std::string mapping_file = vm["annotation-map"].as<std::string>();
      annotation_mapping = build_annotation_mapping(mapping_file);
    }

    // annotated segments of the demonstration
    if (vm.count("segment-dir")) {
      std::vector<std::string> segment_dir_vec = vm["segment-dir"].as<std::vector<std::string>>();
      for (auto &segment_dir : segment_dir_vec) {
        std::vector<std::string> segment_files = collect_paths_of_filetype_from_dir(segment_dir, "txt");
        segment_parser.add_segments(segment_files);
      }
    }

    if (vm.count("output-dir")) {
      output_dir = vm["output-dir"].as<std::string>();
    } else {
      std::cerr << "ERROR: no output-dir specified\n";
      std::cout << desc <<"\n";
      return -1;
    }
  }
  catch (std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  catch (...) {
    std::cerr << "Exception of unknown type!\n";
  }

  std::cout << "Processing the following CSVs:\n";
  for(int i = 0; i < input_csvs.size(); i++){
    std::cout << i << ") " << input_csvs[i] << std::endl;
    trial_manager.load_csv(input_csvs[i]);
  }

  //std::cout << "\nUsing the following annotation mapping:\n";
  //for(auto& it : annotation_mapping){
  //  std::cout << it.first << "->" << it.second << std::endl;
  //}
  WindowManager window_manager;
  window_manager.output_windows(segment_parser, trial_manager, output_dir);

  return 0;
}

