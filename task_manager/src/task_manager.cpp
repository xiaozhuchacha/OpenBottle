#include "task_manager/task_manager.h"

const std::string name = "task_manager";

Manager::Manager() : spinner_(2) {
  action_seq_ = nh_.advertiseService(name + "/action_seq",
                                     &Manager::executeGrammarSequence, this);
  grammar_action_ = nh_.advertiseService(name + "/grammar_action",
                                         &Manager::executeActions, this);

  transition_action_ = nh_.advertiseService(name + "/transition_action",
                                         &Manager::executeTransitionActions, this);

  stop_task_ =
      nh_.advertiseService(name + "/action_stop", &Manager::stopAction, this);

  object_client_ = nh_.serviceClient<action_manager::setObject>("action_manager/object");
  sequence_client_ = nh_.serviceClient<sentence_srv::sentence>("/get_sentence");
  action_pub_ = nh_.advertise<std_msgs::String>("/execution_status", 1);
  action_client_ =
      nh_.serviceClient<action_manager::execAction>("action_manager/exec");

  right_stop_client_ = nh_.serviceClient<huroco_right_arm::rightStop>(
      "huroco_right_arm/rightStop");
  left_stop_client_ =
      nh_.serviceClient<huroco_left_arm::leftStop>("huroco_left_arm/leftStop");

  action_earley_client_ =
      nh_.serviceClient<action_earley_srv::action_earley>("get_earley_action");
  transition_prob_client_ =
      nh_.serviceClient<transition_srv::transition>("get_transition");
  aog_update_client_ =
      nh_.serviceClient<aog_update_srv::aog_update>("update_aog");

  std::string open_bottle_common_path =
      ros::package::getPath("open_bottle_common");
  std::cout << "open_bottle_common_path: " << open_bottle_common_path
            << std::endl;
  out_f_.open(
      std::string(open_bottle_common_path + "/output/task_manager_log.txt")
          .c_str(),
      std::fstream::in | std::fstream::out | std::fstream::app);

  spinner_.start();
}

Manager::~Manager() { out_f_.close(); }

void Manager::getSequence() {
  sentence_srv::sentence get_sentence_srv;

  get_sentence_srv.request.dummy = 0;
  while (sequence_client_.call(get_sentence_srv)) {
    if (get_sentence_srv.response.success == false) continue;

    std::cout << "***************** sample *********************" << std::endl;

    int i;
    for (i = 0; i < get_sentence_srv.response.sentence.size(); i++) {
      std::cout << get_sentence_srv.response.sentence[i] << std::endl;
    }

    std::string judgement;
    std::cout << "Reasonable Sequence? Yes[Y] or No[N]?:" << std::endl;
    std::cin >> judgement;

    if(judgement == "N") continue;
    else if(judgement == "Y") {
      sampled_sentence_ = get_sentence_srv.response.sentence;

      break;
    }
    else continue;
  }
}

void Manager::sendSequence() {
  std::cout << "***************** start *********************" << std::endl;
  stop_sign_ = false;

  action_manager::execAction action_srv;

  int i;
  for (i = 0; i < sampled_sentence_.size(); i++) {
    std_msgs::String msg;
    msg.data = sampled_sentence_[i];
    std::cout << sampled_sentence_[i] << std::endl;
    action_pub_.publish(msg);

    action_srv.request.action = sampled_sentence_[i];
    if (action_client_.call(action_srv)) {
      if (stop_sign_ == true) {
        msg.data = "end";
        action_pub_.publish(msg);
        return;
      }
      std::cout << "Action service call succeeded for: " << sampled_sentence_[i]
                << std::endl;

    } else {
      std::cout << "Action service call failed for: " << sampled_sentence_[i]
                << std::endl;
    }

    sleep(2.0);
  }
}

void Manager::execNextAction() {
  action_earley_srv::action_earley earley_srv;
  transition_srv::transition transition_prob_srv;
  action_manager::execAction action_srv;

  std::string actions_string;

  ros::Time time = ros::Time::now();

  if (actions_.size() == 0) {
    std::cout << "next action: approach" << std::endl;
    action_srv.request.action = "approach";
    action_client_.call(action_srv);

    actions_.push_back(action_srv.request.action);

    return;
  }

  int i;
  for (i = 0; i < actions_.size(); i++) {
    actions_string = actions_string + actions_[i] + " ";
  }

  std::cout << time.sec << "." << time.nsec
            << " previous_actions: " << actions_string << std::endl;
  out_f_ << time.sec << "." << time.nsec
         << " previous_actions: " << actions_string << std::endl;

  earley_srv.request.sentence = actions_string;
  transition_prob_srv.request.update = true;

  action_earley_client_.call(earley_srv);
  transition_prob_client_.call(transition_prob_srv);

  std::vector<std::string> earley_prediction = earley_srv.response.action_seq;
  std::vector<double> earley_action_prob = earley_srv.response.action_prob;

  std::map<std::string, int> transition_action_map;
  transition_action_map["approach"] = 1;
  transition_action_map["move"] = 2;
  transition_action_map["grasp_right"] = 4;
  transition_action_map["ungrasp_right"] = 6;
  transition_action_map["twist"] = 7;
  transition_action_map["push"] = 8;
  transition_action_map["neutral"] = 9;
  transition_action_map["pull"] = 10;
  transition_action_map["pinch"] = 11;
  transition_action_map["unpinch"] = 12;

  int j, action_pos;
  std::string max_action;
  double max_prob = 0;
  double total_sum = 0;

  std::string earley_str = "";
  std::string transition_str = "";
  std::string total_str = "";

  std::vector<double> action_probs;
  std::vector<std::string> action_names;

  std::cout << "no earley actions: " << earley_prediction.size() << std::endl;

  for (j = 0; j < earley_prediction.size(); j++) {
    std::string action_name = earley_prediction[j];

    double earley_prob = earley_action_prob[j];
    double trans_prob = transition_prob_srv.response.next[transition_action_map[action_name]];
    double prior_prob = transition_prob_srv.response.prior[transition_action_map[action_name]];

    earley_str += action_name + " " + std::to_string(earley_prob) + " ";
    transition_str += action_name + " " + std::to_string(trans_prob) + " ";

    if (prior_prob == 0) {
      ROS_ERROR_STREAM(action_name << ": Non-zero earley parse with zero prior, should not be possible");
      continue;
    }

    //double prob = earley_prob * (trans_prob / prior_prob);
    //double prob = earley_prob * (trans_prob);
    double prob = trans_prob;

    // store the probabilities for normalization
    action_probs.push_back(prob);
    action_names.push_back(action_name);
    total_sum += prob;

  }

  // if all probabilities are zero, default to a random selection 
  // following distribution of earley probabilities
  if (total_sum == 0) {

    // sample an action from the 
    std::srand(std::time(0));
    std::mt19937 generator;
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    double sample_prob = dis(generator);


    //std::cout << "Sampled Prob:" << sample_prob << ", sum:" << sum << std::endl;
    // std::cout << "-----------------------" << sample_prob <<
    // "----------------------" << std::endl;

    int i;
    double sum = 0;
    for (i = 0; i < earley_prediction.size(); i++) {
      sum += earley_action_prob[i];
      //  std::cout << "--" << earley_prediction[i] << "--" << sum <<
      // std::endl;
      if (sample_prob < sum) {
        max_action = earley_prediction[i];
        break;
      }
    }
  }

  // normalize computed probabilities
  for(i = 0; i < action_names.size(); i++) {
    action_probs[i] = action_probs[i] / total_sum;
    total_str += action_names[i] + " " + std::to_string(action_probs[i]) + " ";
  
    if (action_probs[i] > max_prob) {
      max_prob = action_probs[i];
      max_action = action_names[i];
      //std::cout << "new max action: " << max_action << " prob: " << max_prob << std::endl;
    }
  }

  //std::cout << " earley_action: " << earley_str << " " << std::endl;
  //out_f_ << time.sec << "." << time.nsec << " earley_action: " << earley_str << std::endl;

  std::cout << " transition_predict: " << transition_str << " " << std::endl;
  //out_f_ << time.sec << "." << time.nsec << " transition_predict: " << transition_str << std::endl;

  //std::cout << " total_predict: " << total_str << std::endl;
  //out_f_ << time.sec << "." << time.nsec << " total_predict: " << total_str << std::endl;

  //std::cout << time.sec << "." << time.nsec << " next_action: " << max_action << std::endl;

  //std::cout << "actions_size(): " << actions_.size() << std::endl;

  if(actions_.size() == 1) {
    int k = 0;
    double sum = 0;

    std::srand(std::time(0));
  /*
    int rand_var = std::rand();
    int remainder = (rand_var % 10);
    double sample_prob = (double)remainder / 10;
  */
  
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    double sample_prob = dis(generator);
  
    std::cout << "Sampled Prob: " << sample_prob << std::endl;

    for(k = 0; k < earley_prediction.size(); k++) {
      sum += earley_action_prob[k];

      std::string first_step;
      //  std::cout << "--" << earley_prediction[i] << "--" << sum <<
      // std::endl;
      if (sample_prob < sum) {
        //max_action = earley_prediction[k];
        max_action = "push";
        std::cout << "First Step: " << max_action << std::endl;
/*
        std::string whatever;
        std::cin >> whatever;
*/      
        break;
      }
    }  
  }

  action_srv.request.action = max_action;
  action_client_.call(action_srv);
  sleep(1.5);
  actions_.push_back(max_action);

}

bool Manager::stopAction(task_manager::stopAction::Request &req,
                         task_manager::stopAction::Response &res) 
{
  stop_sign_ = true;

  aog_update_srv::aog_update update_srv;

  if (req.complete_task == true) {
    update_srv.request.sentence = actions_;
    update_srv.request.update = true;

    aog_update_client_.call(update_srv);

    res.update_aog = update_srv.response.success;
  } else {
    res.update_aog = false;
  }
  /*
          huroco_right_arm::rightStop right_stop_srv;
          huroco_left_arm::leftStop left_stop_srv;

          if(right_stop_client_.call(right_stop_srv)) {

          }

          if(left_stop_client_.call(left_stop_srv)) {

          }
  */

  return true;
}

bool Manager::executeGrammarSequence(task_manager::actionSeq::Request &req,
                              task_manager::actionSeq::Response &res) 
{
  action_manager::setObject object_srv;
  object_srv.request.object = req.bottle_type;

  object_client_.call(object_srv);

  getSequence();

  sendSequence();

  return true;
}


bool Manager::executeTransitionActions(task_manager::transitionActions::Request &req,
                                       task_manager::transitionActions::Response &res)
{
  action_manager::setObject object_srv;
  object_srv.request.object = req.bottle_type;
  object_client_.call(object_srv);

  action_earley_srv::action_earley earley_srv;
  transition_srv::transition transition_prob_srv;
  action_manager::execAction action_srv;

  std::map<int, std::string> transition_action_map_reverse;

  transition_action_map_reverse[1] = "approach";
  transition_action_map_reverse[2] = "move";
  transition_action_map_reverse[4] = "grasp_right";
  transition_action_map_reverse[6] = "ungrasp_right";
  transition_action_map_reverse[7] = "twist";
  transition_action_map_reverse[8] = "push";
  transition_action_map_reverse[9] = "neutral";
  transition_action_map_reverse[10] = "pull";
  transition_action_map_reverse[11] = "pinch";
  transition_action_map_reverse[12] = "unpinch";

  std::string max_action;

  int i;
  for(i = 0; i < 12; i++) {
    if(i == 0) {
      max_action = "approach";
    }
    else if(i == 1) {
      std::string actions_string;

      int m;
      for (m = 0; m < actions_.size(); m++) {
        actions_string = actions_string + actions_[m] + " ";
      }

      std::cout << actions_string << std::endl;

      earley_srv.request.sentence = actions_string;
      action_earley_client_.call(earley_srv);

      std::vector<std::string> earley_prediction = earley_srv.response.action_seq;
      std::vector<double> earley_action_prob = earley_srv.response.action_prob;

      std::random_device rd;
      std::mt19937 generator(rd());
      std::uniform_real_distribution<double> dis(0.0, 1.0);
      double sample_prob = dis(generator);
    
      std::cout << "Sampled Prob: " << sample_prob << std::endl;

      int k;
      double sum;
      for(k = 0; k < earley_prediction.size(); k++) {
        sum += earley_action_prob[k];

        std::string first_step;
        //  std::cout << "--" << earley_prediction[k] << "--" << earley_action_prob << std::endl;
        if (sample_prob < sum) {
          max_action = earley_prediction[k];
          std::cout << "First Step: " << max_action << std::endl;
   
          break; 
        } 
      }   
    }
    else {
      transition_prob_srv.request.update = true;
      transition_prob_client_.call(transition_prob_srv);

      std::cout << "Transition predictions: " << std::endl;

      int j;
      double max_prob = 0;
      for(j = 0; j < transition_prob_srv.response.next.size(); j++) {
        std::cout << transition_action_map_reverse[j] << " : " << transition_prob_srv.response.next[j] << ". " << std::endl;

        if(transition_prob_srv.response.next[j] > max_prob) {
          max_prob = transition_prob_srv.response.next[j];
          max_action = transition_action_map_reverse[j];
        }
      }
    }

    std::cout << "max_action: " << max_action << std::endl;
    action_srv.request.action = max_action;
    action_client_.call(action_srv);
    actions_.push_back(max_action);
    sleep(2.5);  
  }

  std::cout << "Reach maximum amount of actions, terminated." << std::endl;
  std::cout << "Actions have beed executed: ";

  int n;
  for(n = 0; n < actions_.size(); n++) {
    std::cout << actions_[n] << " ";
  }
  std::cout << std::endl;

  actions_.clear();

  return true;
}


bool Manager::executeActions(task_manager::actionExec::Request &req,
                             task_manager::actionExec::Response &res) {
  stop_sign_ = true;

  action_manager::setObject object_srv;
  object_srv.request.object = req.bottle_type;

  object_client_.call(object_srv);

  std::cout << "start executing" << std::endl;

  int i = 0;
  while(stop_sign_ = true) {
    execNextAction();

    i++;
    sleep(2.5);
  }

  return true;
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "task_manager");

  Manager task_manager;

  ROS_INFO("waiting for action sequence");
  ros::waitForShutdown();

  return 0;
}