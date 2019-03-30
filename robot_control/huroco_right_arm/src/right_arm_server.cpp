#include "huroco_right_arm/right_arm_server.h"

RightArmServer::RightArmServer(ArmManipulator *manipulator, moveit::planning_interface::MoveGroupInterface &group)
		: spinner_(2), controller_(manipulator)
{
	right_cartesian_ = nh_.advertiseService(name + "/right_cartesian", &RightArmServer::rightCartesianPath, this);
	right_init_ = nh_.advertiseService(name + "/right_init", &RightArmServer::rightInit, this);
	right_stop_ = nh_.advertiseService(name + "/right_stop", &RightArmServer::rightStop, this);

	controller_->setServer(this);

	spinner_.start();
}

RightArmServer::~RightArmServer() {}

bool RightArmServer::rightCartesianPath(huroco_right_arm::rightCartesian::Request &req,
																				huroco_right_arm::rightCartesian::Response &res)
{
	res.status = static_cast<bool>(controller_->executeCartesianPath(req.waypoints));
	return true;
}

bool RightArmServer::rightInit(huroco_right_arm::rightInit::Request &req,
															 huroco_right_arm::rightInit::Response &res)
{
	res.status = static_cast<bool>(controller_->initPose());
	return true;
}

bool RightArmServer::rightStop(huroco_right_arm::rightStop::Request &req,
															 huroco_right_arm::rightStop::Response &res)
{
	controller_->abortExecution();
	return true;
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "right_arm_server");

	moveit::planning_interface::MoveGroupInterface group("right_arm");
	ArmManipulator manipulator(group);
	RightArmServer right_server(&manipulator, group);

	ros::waitForShutdown();

	return 0;
}