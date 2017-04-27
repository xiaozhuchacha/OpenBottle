//
// Created by mark on 4/25/17.
//

#ifndef SEGMENT_PARSER_H
#define SEGMENT_PARSER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <algorithm>

#include "util.h"

// stores a single action segment. Consists fo start time, end time, and the trial identifier
// (to map back to original file)
class ActionSegment{
public:
  ActionSegment(std::string action, int end_time_sec, int end_time_nsec, std::string trial_id) :
                action(action), end_time_sec(end_time_sec), end_time_nsec(end_time_nsec),
                trial_id(trial_id), start_time_sec(-1), start_time_nsec(-1)
  {

  }
  ~ActionSegment() {};

  std::string action;
  std::string next_action;
  int start_time_sec;
  int start_time_nsec;
  int end_time_sec;
  int end_time_nsec;
  std::string trial_id;
};

// stores segments by their temporal order per trial
class SegmentParser {
public:
  SegmentParser() {};
  ~SegmentParser();

  void add_segments(std::vector<std::string> segment_files);
  std::vector<ActionSegment*>* get_trial_actions(std::string trial_id);

  // key is the trial ID, value is a temporally sequential order of the events
  std::unordered_map<std::string, std::vector<ActionSegment*>*> trial_segments;
private:

  // processes each segment file. Assumes each segment_file is only for this trial
  void process_segment_files(std::string trial_id, std::vector<std::string> segment_files);

  void parse_segment_file(std::vector<ActionSegment*> &action_segments, const std::string &trial_id, const std::string &segment_file);
};

// needed for comparison operator to sort the vector of segment files
bool segment_file_cmp(std::string s1, std::string s2);


#endif //SEGMENT_PARSER_H
