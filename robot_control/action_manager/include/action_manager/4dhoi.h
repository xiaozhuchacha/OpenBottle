#ifndef ACTION_H
#define ACTION_H


#include <ros/ros.h>
#include <std_msgs/String.h>

#include <string>

#include "huroco_right_arm/rightCartesian.h"
#include "huroco_left_arm/leftCartesian.h"
#include "huroco_left_arm/rotateLeftWrist.h"
#include "huroco_right_arm/rightInit.h"

#include "huroco_grasping/graspPose.h"
#include "huroco_grasping/graspCap.h"

#include "right_gripper_receiver/rightGrasp.h"
#include "right_gripper_receiver/rightStatus.h"

#include "action_manager/execAction.h"
#include "action_manager/setBottle.h"

#include "baxter_core_msgs/EndpointState.h"



class Action
{
public:
	Action();
	~Action();


private:
	void getLeftPose(const baxter_core_msgs::EndpointState msg);

	void approach(std::string bottle);

	void twist();

	void move();

	void pinch();

	void grasp();

	void ungrasp();

	void push();

	void neutral();

	void pull();

	bool setBottles(action_manager::setBottle::Request &req,
					action_manager::setBottle::Response &res);

	bool executeActions(action_manager::execAction::Request &req,
						action_manager::execAction::Response &res);


	ros::NodeHandle nh_;
	ros::AsyncSpinner spinner_;

	ros::ServiceServer set_bottle_;
	ros::ServiceServer action_execution_;

	ros::ServiceClient right_cartesian_;
	ros::ServiceClient left_cartesian_;

	ros::ServiceClient right_approach_;
	ros::ServiceClient left_approach_;

	ros::ServiceClient right_grasp_;
	ros::ServiceClient right_grasp_status_;
	ros::Publisher left_grasp_;

	ros::ServiceClient left_wrist_;

	ros::ServiceClient right_init_;

	ros::Subscriber robot_state_sub_;

	geometry_msgs::Pose current_pose_;

	bool sub_trigger_;

	std::string bottle_;
};


#endif
