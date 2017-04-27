//
// Created by mark on 4/26/17.
//

#ifndef WINDOW_MANAGER_H
#define WINDOW_MANAGER_H

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <algorithm>

#include "segment_parser.h"
#include "trial_content.h"

// creates and outputs pre and postcondition windows
class WindowManager {
public:
  WindowManager() {};
  ~WindowManager();

  void output_windows(SegmentParser& segment_parser, TrialManager &trial_manager, std::string output_dir);
private:

  std::unordered_map<std::string, std::ofstream*> action_to_file;

  int get_trial_content_idx(TrialContent *trial_content, int start_idx, int sec, int nsec);
  void output_trial_row(TrialContent &trial_content, ActionSegment &action, std::ofstream* outfile_ptr,
                        int trial_content_idx);
  std::ofstream* get_outfile_ptr(std::string action, std::string output_dir);
};


#endif //WINDOW_MANAGER_H
