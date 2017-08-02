#ifndef RIGHT_ARM_SERVER_H
#define RIGHT_ARM_SERVER_H

#include <ros/ros.h>
#include <string>
#include <vector>

#include "huroco_right_arm/moveit_right_arm_controller.h"

// service files
#include "huroco_right_arm/rightCartesian.h"
#include "huroco_right_arm/rightInit.h"
#include "huroco_right_arm/rightStop.h"

const std::string name = "huroco_right_arm";


class RightArmServer
{
public:
	RightArmServer(ArmManipulator *left_arm, moveit::planning_interface::MoveGroupInterface &group);
	~RightArmServer();

private:
	ros::NodeHandle nh_;
	ros::AsyncSpinner spinner_;

	ros::ServiceServer right_exec_;
	ros::ServiceServer right_cartesian_;
	ros::ServiceServer right_init_;
	ros::ServiceServer right_stop_;

	std::unique_ptr<ArmManipulator> controller_;
	

	bool rightCartesianPath(huroco_right_arm::rightCartesian::Request &req,
							huroco_right_arm::rightCartesian::Response &res);

	bool rightInit(huroco_right_arm::rightInit::Request &req,
				   huroco_right_arm::rightInit::Response &res);

	bool rightStop(huroco_right_arm::rightStop::Request &req,
				   huroco_right_arm::rightStop::Response &res);
};

#endif