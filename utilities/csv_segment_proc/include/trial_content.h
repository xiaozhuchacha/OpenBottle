//
// Created by mark on 4/22/17.
//

#ifndef TRIAL_CONTENT_H
#define TRIAL_CONTENT_H

#include <unordered_map>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>

#include <ros/ros.h>
#include <geometry_msgs/TransformStamped.h>

#include <boost/algorithm/string/classification.hpp> // Include boost::for is_any_of
#include <boost/algorithm/string/split.hpp> // Include for boost::split

#include "util.h"

// stores the content of one trial
class TrialContent {
public:
  TrialContent() {};
  ~TrialContent();

  std::vector<int> time_secs;
  std::vector<int> time_nsecs;
  std::vector<int> image_ids;
  std::vector<std::vector<geometry_msgs::TransformStamped>*> transforms;
  std::vector<std::vector<float>*> forces;

  int find_idx_closest_time(int time_sec, int time_nsec);
};

class TrialManager{
public:
  TrialManager() {};
  ~TrialManager() {};

  void load_csv(std::string csv_path);

  TrialContent* get_trial_content(std::string trial_id);

  std::unordered_map<std::string, TrialContent*> trial_content; // key'd by the trial_id
};

#endif //TRIAL_CONTENT_H
