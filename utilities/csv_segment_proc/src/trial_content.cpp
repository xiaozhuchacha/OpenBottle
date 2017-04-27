//
// Created by mark on 4/22/17.
//

#include "trial_content.h"
TrialContent::~TrialContent() {
  assert(transforms.size() == forces.size());
  for (int i = 0; i < transforms.size(); i++) {
    delete transforms[i];
    delete forces[i];
  }
}

void TrialManager::load_csv(std::string csv_path) {
  std::ifstream infile(csv_path);
  std::string line;

  std::string trial_id = extract_trial_id_from_path(csv_path);

  TrialContent* content = get_trial_content(trial_id);

  int line_no = 0;
  while(std::getline(infile, line)){
    std::vector<std::string> tokens;
    boost::split(tokens, line, boost::is_any_of(","), boost::token_compress_on);
    int time_sec = std::stoi(tokens[0]);
    int time_nsec = std::stoi(tokens[1]);
    int image_id = std::stoi(tokens[2]);

    int num_tfs = ((int)tokens.size() - 3 - 26) / 9; // 3 for time & image_id, 26 for forces

    std::vector<geometry_msgs::TransformStamped>* transforms = new std::vector<geometry_msgs::TransformStamped>(num_tfs);
    std::vector<float>* forces = new std::vector<float>(26);

    std::vector<geometry_msgs::TransformStamped>& transforms_ref = *transforms;
    int offset = 3;
    for (int i = 0; i < num_tfs; i++){
      transforms_ref[i].header.stamp.sec = time_sec;
      transforms_ref[i].header.stamp.nsec = time_nsec;
      transforms_ref[i].header.frame_id = tokens[offset];
      transforms_ref[i].child_frame_id = tokens[offset+1];
      transforms_ref[i].transform.translation.x = std::stof(tokens[offset+2]);
      transforms_ref[i].transform.translation.y = std::stof(tokens[offset+3]);
      transforms_ref[i].transform.translation.z = std::stof(tokens[offset+4]);
      transforms_ref[i].transform.rotation.x = std::stof(tokens[offset+5]);
      transforms_ref[i].transform.rotation.y = std::stof(tokens[offset+6]);
      transforms_ref[i].transform.rotation.z = std::stof(tokens[offset+7]);
      transforms_ref[i].transform.rotation.w = std::stof(tokens[offset+8]);
      offset += 9;
    }

    std::vector<float>& forces_ref = *forces;
    for(int i = 0; i < 26; i++){
      forces_ref[i] = std::stof(tokens[offset+i]);
    }

    content->time_secs.push_back(time_sec);
    content->time_nsecs.push_back(time_nsec);
    content->image_ids.push_back(image_id);
    content->transforms.push_back(transforms);
    content->forces.push_back(forces);
    line_no++;
  }
}

TrialContent* TrialManager::get_trial_content(std::string trial_id){
  auto trial_iter = trial_content.find(trial_id);
  TrialContent* trial_ptr = nullptr;
  // new entry
  if (trial_iter == trial_content.end()){
    trial_ptr = new TrialContent();
    trial_content.insert({trial_id, trial_ptr});
  }
  else {
    trial_ptr = trial_iter->second;
  }
  return trial_ptr;
}
