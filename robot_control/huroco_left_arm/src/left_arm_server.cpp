#include "huroco_left_arm/left_arm_server.h"

LeftArmServer::LeftArmServer(ArmManipulator *manipulator, moveit::planning_interface::MoveGroupInterface &group)
		: spinner_(2), controller_(manipulator)
{
	left_cartesian_ = nh_.advertiseService(name + "/left_cartesian", &LeftArmServer::leftCartesianPath, this);
	rotate_wrist_ = nh_.advertiseService(name + "/rotate_wrist", &LeftArmServer::rotateLeftWrist, this);
	left_stop_ = nh_.advertiseService(name + "/left_stop", &LeftArmServer::leftStop, this);
	left_init_ = nh_.advertiseService(name + "/left_init", &LeftArmServer::leftInit, this);

	controller_->setServer(this);

	spinner_.start();
}

LeftArmServer::~LeftArmServer() {}

bool LeftArmServer::leftCartesianPath(huroco_left_arm::leftCartesian::Request &req,
																			huroco_left_arm::leftCartesian::Response &res)
{
	res.status = static_cast<bool>(controller_->executeCartesianPath(req.waypoints));
	return true;
}

bool LeftArmServer::rotateLeftWrist(huroco_left_arm::rotateLeftWrist::Request &req,
																		huroco_left_arm::rotateLeftWrist::Response &res)
{
	res.status = static_cast<bool>(controller_->rotateWrist(req.radian));
	return true;
}

bool LeftArmServer::leftStop(huroco_left_arm::leftStop::Request &req,
														 huroco_left_arm::leftStop::Response &res)
{
	controller_->abortExecution();
	return true;
}

bool LeftArmServer::leftInit(huroco_left_arm::leftInit::Request &req,
														 huroco_left_arm::leftInit::Response &res)
{
	controller_->initPose();
	return true;
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "left_arm_server");

	moveit::planning_interface::MoveGroupInterface group("left_arm");
	ArmManipulator manipulator(group);
	LeftArmServer left_server(&manipulator, group);

	//ros::spin();
	ros::waitForShutdown();

	return 0;
}