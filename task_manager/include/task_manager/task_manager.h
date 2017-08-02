#ifndef TASK_H
#define TASK_H


#include <ros/ros.h>
#include <std_msgs/String.h>
#include <ros/package.h>

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <random>


#include "huroco_right_arm/rightStop.h"
#include "huroco_left_arm/leftStop.h"

#include "action_manager/execAction.h"
#include "action_manager/setObject.h"

#include "sentence_srv/sentence.h"

#include "task_manager/actionSeq.h"
#include "task_manager/stopAction.h"
#include "task_manager/actionExec.h"
#include "task_manager/transitionActions.h"


#include "transition_srv/transition.h"
#include "action_earley_srv/action_earley.h"
#include "aog_update_srv/aog_update.h"



class Manager
{
public:
	Manager();
	~Manager();

private:
	void getSequence();

	void sendSequence();

	void execNextAction();


	ros::NodeHandle nh_;
	ros::AsyncSpinner spinner_;

	ros::ServiceServer action_seq_;
	ros::ServiceServer grammar_action_;
	ros::ServiceServer transition_action_;
	ros::ServiceServer stop_task_;

	ros::ServiceClient object_client_;
	ros::ServiceClient sequence_client_;
	ros::ServiceClient action_client_;
	ros::ServiceClient right_stop_client_;
	ros::ServiceClient left_stop_client_;
	ros::ServiceClient action_earley_client_;
	ros::ServiceClient transition_prob_client_;
	ros::ServiceClient aog_update_client_;

	ros::Publisher action_pub_;

	std::vector<std::string> sampled_sentence_;
	std::vector<std::string> actions_;

	bool stop_sign_; 

	std::fstream out_f_;

	bool stopAction(task_manager::stopAction::Request &req,
					task_manager::stopAction::Response &res);

	bool executeGrammarSequence(task_manager::actionSeq::Request &req,
						 		task_manager::actionSeq::Response &res);

	bool executeTransitionActions(task_manager::transitionActions::Request &req,
								  task_manager::transitionActions::Response &res);

	bool executeActions(task_manager::actionExec::Request &req,
						task_manager::actionExec::Response &res);
};


#endif
