//
// Created by mark on 4/26/17.
//

#include "window_manager.h"

WindowManager::~WindowManager(){
  // deallocate ofstreams
  for(auto& iter : action_to_file){
    delete iter.second;
  }
}


void WindowManager::output_windows(SegmentParser& segment_parser, TrialManager &trial_manager,
                                   std::string output_dir){
  std::vector<std::string> sorted_trial_order;
  for(auto& trial_iter : trial_manager.trial_content){
    sorted_trial_order.push_back(trial_iter.first);
  }
  std::sort(sorted_trial_order.begin(), sorted_trial_order.end(),
            [](std::string s1, std::string s2)
            {
              return std::stof(s1) < std::stof(s2);
            });

  // main outer loop. iterates over trials and writes to the file.
  // Both the trial content and segments must the accessed in order.
  // The loop walks through the trial content, looking for the closest time to the current segment's timestamp
  // Once found, loop moves to the next action after outputting pre/post windows to CSV
  // This iteration requires both the segments and trial content to be stored in temporal order
  for(auto& trial_id : sorted_trial_order){
    std::vector<ActionSegment*> *actions = segment_parser.get_trial_actions(trial_id);
    assert(actions != nullptr);

    // get trial content for this trial
    TrialContent* trial_content = nullptr;
    auto trial_content_iter = trial_manager.trial_content.find(trial_id);
    assert(trial_content_iter != trial_manager.trial_content.end());
    trial_content = trial_content_iter->second;

    int trial_content_idx = 0;
    // primary loop. Finds the windows for this action
    for (int i = 0; i < actions->size(); i++){
      ActionSegment *action = (*actions)[i];
      if (action->action == "start")
        continue;

      // get output file
      std::ofstream *outfile_ptr = get_outfile_ptr(action->action, output_dir);

      // get the index of the pre-condition of this action
      trial_content_idx = get_trial_content_idx(trial_content, trial_content_idx, action->start_time_sec,
                                                action->start_time_nsec);

      // trial_content_idx is at the pre-condition of the action
      output_trial_row(*trial_content, *action, outfile_ptr, trial_content_idx);

      // get the index of the post-condition of this action
      trial_content_idx = get_trial_content_idx(trial_content, trial_content_idx, action->end_time_sec,
                                                action->end_time_nsec);

      // trial_content_idx is at the pre-condition of the action
      output_trial_row(*trial_content, *action, outfile_ptr, trial_content_idx);
   }
  }
}

int WindowManager::get_trial_content_idx(TrialContent *trial_content, int start_idx, int sec, int nsec) {
  int idx = start_idx;
  //TODO: make this more precise. Compute actual closest timestamp (this overshoots)
  // find the closest second
  while(trial_content->time_secs[idx] < sec){
    idx++;
  }
  // find the closest nanosecond without moving to the next second
  while(trial_content->time_nsecs[idx] < nsec && trial_content->time_secs[idx] == sec){
    idx++;
  }
  return idx;
}

void WindowManager::output_trial_row(TrialContent &trial_content, ActionSegment &action,
                                     std::ofstream *outfile_ptr, int trial_content_idx){
  std::ofstream &outfile_ref = *outfile_ptr; // easier to use

  // output timestamp
  outfile_ref << std::to_string(trial_content.time_secs[trial_content_idx]) << ","
              << std::to_string(trial_content.time_nsecs[trial_content_idx]) << ","
              << std::to_string(trial_content.image_ids[trial_content_idx]) << ",";
  // output tf's
  for(auto& tf_iter : *(trial_content.transforms[trial_content_idx])){
    outfile_ref << tf_iter.header.frame_id << "," << tf_iter.child_frame_id << ","
                << tf_iter.transform.translation.x << "," << tf_iter.transform.translation.y << ","
                << tf_iter.transform.translation.z << "," << tf_iter.transform.rotation.x << ","
                << tf_iter.transform.rotation.y << "," << tf_iter.transform.rotation.z << ","
                << tf_iter.transform.rotation.w << ",";
  }
  // output forces
  for(auto& force_iter : *(trial_content.forces[trial_content_idx])){
    outfile_ref << force_iter << ",";
  }
  //output next action
  outfile_ref << action.next_action << "\n";
}

std::ofstream* WindowManager::get_outfile_ptr(std::string action, std::string output_dir){
  std::ofstream *outfile_ptr = nullptr;
  auto action_file_iter = action_to_file.find(action);
  // make new ostream
  if(action_file_iter == action_to_file.end()){
    outfile_ptr = new std::ofstream(output_dir + "/" + action + ".csv", std::ofstream::out);
    action_to_file.insert({action, outfile_ptr});
  }
  else {
    outfile_ptr = action_file_iter->second;
  }
  return outfile_ptr;
}
