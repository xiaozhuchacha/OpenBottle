//
// Created by mark on 4/25/17.
//

#include "segment_parser.h"

SegmentParser::~SegmentParser(){
  for(auto& trial_iter : trial_segments) {
    for (auto &segment_iter : *(trial_iter.second)) {
      delete segment_iter; // delete each ActionSegement
    }
    delete trial_iter.second; // delete vector
  }
}

void SegmentParser::add_segments(std::vector<std::string> segment_files) {
  // separate by trial
  std::unordered_map<std::string, std::vector<std::string>> trial_to_segment_file_vec;
  for (auto& segment_file : segment_files){
    std::string trial_id = extract_trial_id_from_path(segment_file);
    // insert into hash table
    auto trial_iter = trial_to_segment_file_vec.find(trial_id);
    if (trial_iter == trial_to_segment_file_vec.end()){
      trial_to_segment_file_vec.insert({trial_id, std::vector<std::string>(1, segment_file)});
    } else {
      trial_iter->second.push_back(segment_file);
    }
  }

  // sort the segment files per trial (numbering is per trial) and process each action
  for(auto& trial_iter : trial_to_segment_file_vec){
    std::sort(trial_iter.second.begin(), trial_iter.second.end(), segment_file_cmp);
    process_segment_files(trial_iter.first, trial_iter.second);
  }
}

void SegmentParser::process_segment_files(std::string trial_id, std::vector<std::string> segment_files){
  std::vector<ActionSegment*> action_segments;
  for(auto& segment_file : segment_files){
    parse_segment_file(action_segments, trial_id, segment_file);
  }
}

void SegmentParser::parse_segment_file(std::vector<ActionSegment*> &action_segments, const std::string &trial_id,
                                       const std::string &segment_file){
  std::ifstream infile(segment_file);
  std::string line;

  // get the ptr to the vector of action segments for this action from the hash table
  std::vector<ActionSegment*> *trial_vec;
  auto trial_segment_iter = trial_segments.find(trial_id);
  if (trial_segment_iter == trial_segments.end()){
    trial_vec = new std::vector<ActionSegment*>();
    trial_segments.insert({trial_id, trial_vec});
  }
  else{
    trial_vec = trial_segment_iter->second;
  }

  // keep track of previous actions to update prev_action and next_action for each segment
  ActionSegment *prev_action = nullptr;

  while(std::getline(infile, line)) {
    if (line.substr(0, 3) == "---") // skip first line
      continue;
    std::vector<std::string> tokens;
    boost::split(tokens, line, boost::is_any_of(","), boost::token_compress_on);

    std::string action = tokens[0];
    // each action is labeled as the end, the time between the previous label and the current label is
    // the duration of the current label
    int end_time_sec = std::stoi(tokens[1]);
    int end_time_nsec = std::stoi(tokens[2]);
    ActionSegment *cur_action = new ActionSegment(action, end_time_sec, end_time_nsec, trial_id);

    // update values and update previous actions
    if (prev_action == nullptr) {
      prev_action = cur_action;
    } else {
      // update previous and
      prev_action->next_action = cur_action->action;
      cur_action->start_time_sec = prev_action->end_time_sec;
      cur_action->start_time_nsec = prev_action->end_time_nsec;
      prev_action = cur_action;
    }

    // add segment to this trial's vector of actions
    trial_vec->push_back(cur_action);
  }
}

std::vector<ActionSegment*>* SegmentParser::get_trial_actions(std::string trial_id){
  // lookup segments of this trial
  auto segment_iter = trial_segments.find(trial_id);
  if (segment_iter == trial_segments.end()){
    std::cerr << "Unable to find segments for trial ID: \"" << trial_id << "\"\n";
    std::cerr << "Make sure the associated annotated segments are loaded\n";
    return nullptr;
  }
  return segment_iter->second;
}

// comparison between segement files.
// extract the segment number from string. File format should be X_#_segments.txt, where # is the segment number
bool segment_file_cmp(std::string s1, std::string s2)
{
  unsigned long n1 = s1.find_last_of("_");
  unsigned long n2 = s2.find_last_of("_");
  s1 = s1.substr(0,n1);
  s2 = s2.substr(0,n2);
  n1 = s1.find_last_of("_");
  n2 = s2.find_last_of("_");
  s1 = s1.substr(n1 + 1, s1.size() - n1);
  s2 = s2.substr(n2 + 1, s2.size() - n2);

  // return the comparison of the floats (may have decimal if trial was restarted)
  return std::stof(s1) < std::stof(s2);
}

