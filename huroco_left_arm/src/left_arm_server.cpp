#include "huroco_left_arm/left_arm_server.h"

const std::string name = "huroco_left_arm";


LeftArmServer::LeftArmServer(ArmManipulator *manipulator, moveit::planning_interface::MoveGroupInterface &group)
			:spinner_(1), controller_(manipulator)
{
	left_exec_ = nh_.advertiseService(name + "/left_exec", &LeftArmServer::leftExec, this);
	left_cartesian_ = nh_.advertiseService(name + "/left_cartesian", &LeftArmServer::leftCartesianPath, this);
	rotate_wrist_ = nh_.advertiseService(name + "/rotate_wrist", &LeftArmServer::rotateLeftWrist, this);

	controller_->setServer(this);

	spinner_.start();
}


LeftArmServer::~LeftArmServer() {}


bool LeftArmServer::leftExec(huroco_left_arm::leftExec::Request &req,
							 huroco_left_arm::leftExec::Response &res)
{
	controller_->executeGoal(req.goal);
	return true;
}


bool LeftArmServer::leftCartesianPath(huroco_left_arm::leftCartesian::Request &req,
									  huroco_left_arm::leftCartesian::Response &res)
{
	res.status = controller_->executeCartesianPath(req.waypoints, req.step);
	return true;
}


bool LeftArmServer::rotateLeftWrist(huroco_left_arm::rotateLeftWrist::Request &req,
							   huroco_left_arm::rotateLeftWrist::Response &res)
{
	res.status = controller_->rotateWrist(req.radian);
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