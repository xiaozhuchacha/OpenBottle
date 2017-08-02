#ifndef ACTION_H
#define ACTION_H


#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

#include <string>

#include "huroco_right_arm/rightCartesian.h"
#include "huroco_left_arm/leftCartesian.h"
#include "huroco_left_arm/rotateLeftWrist.h"
#include "huroco_right_arm/rightInit.h"
#include "huroco_left_arm/leftInit.h"

#include "huroco_grasping/graspPose.h"
#include "huroco_grasping/graspCap.h"
#include "huroco_grasping/graspDoor.h"

#include "right_gripper_receiver/rightGrasp.h"
#include "right_gripper_receiver/rightStatus.h"

#include "action_manager/execAction.h"
#include "action_manager/setObject.h"

#include "baxter_core_msgs/EndpointState.h"



class Action
{
public:
	Action();
	~Action();


private:
	void getLeftPose(const baxter_core_msgs::EndpointState msg);

	void getDoorPose(const geometry_msgs::PoseStamped msg);

	void approach(std::string bottle);

	void twist();

	void move();

	void pinch();

	void grasp();

	void ungrasp();

	void push();

	void neutral();

	void pull();

	void shake();

	void openDoor();

	void closeDoor();

	void fetch();

	void point();

	void pour();

	void officedoor();

	bool setObject(action_manager::setObject::Request &req,
					action_manager::setObject::Response &res);

	bool executeActions(action_manager::execAction::Request &req,
						action_manager::execAction::Response &res);


	ros::NodeHandle nh_;
	ros::AsyncSpinner spinner_;

	ros::ServiceServer set_object_;
	ros::ServiceServer action_execution_;

	ros::ServiceClient right_cartesian_;
	ros::ServiceClient left_cartesian_;

	ros::ServiceClient right_approach_;
	ros::ServiceClient left_approach_;

	ros::ServiceClient left_door_;

	ros::ServiceClient right_grasp_;
	ros::ServiceClient right_grasp_status_;
	ros::Publisher left_grasp_;

	ros::ServiceClient left_wrist_;

	ros::ServiceClient right_init_;
	ros::ServiceClient left_init_;

	ros::Subscriber robot_state_sub_;
	ros::Subscriber door_state_sub_;

	geometry_msgs::Pose current_pose_;
	geometry_msgs::Pose door_pose_;

	ros::Publisher action_pub_;

	bool sub_trigger_;

	tf::Transform door_tf_;

	std::string object_;

	std::vector<geometry_msgs::Pose> curve_;

	std::vector<int> cur_sequence_;
};


#endif
