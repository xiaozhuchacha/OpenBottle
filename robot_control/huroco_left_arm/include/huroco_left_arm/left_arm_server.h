#ifndef LEFT_ARM_SERVER_H
#define LEFT_ARM_SERVER_H

/* ros headers */
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>

/* cpp headers */
#include <vector>
#include <string>

#include "huroco_left_arm/moveit_left_arm_controller.h"

/* service headers */
#include "huroco_left_arm/leftCartesian.h"
#include "huroco_left_arm/rotateLeftWrist.h"
#include "huroco_left_arm/leftStop.h"
#include "huroco_left_arm/leftInit.h"

const std::string name = "huroco_left_arm";


class LeftArmServer
{
public:
	LeftArmServer(ArmManipulator *left_arm, moveit::planning_interface::MoveGroupInterface &group);
	~LeftArmServer();

private:
	ros::NodeHandle nh_;
	ros::AsyncSpinner spinner_;

	ros::ServiceServer left_exec_;
	ros::ServiceServer left_cartesian_;
	ros::ServiceServer rotate_wrist_;
	ros::ServiceServer left_stop_;
	ros::ServiceServer left_init_;

	std::unique_ptr<ArmManipulator> controller_;
	

	bool leftCartesianPath(huroco_left_arm::leftCartesian::Request &req,
						   huroco_left_arm::leftCartesian::Response &res);

	bool rotateLeftWrist(huroco_left_arm::rotateLeftWrist::Request &req,
						 huroco_left_arm::rotateLeftWrist::Response &res);

	bool leftStop(huroco_left_arm::leftStop::Request &req,
				  huroco_left_arm::leftStop::Response &res);

	bool leftInit(huroco_left_arm::leftInit::Request &req,
				  huroco_left_arm::leftInit::Response &res);

};

#endif