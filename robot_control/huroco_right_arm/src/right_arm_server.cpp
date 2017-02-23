#include "huroco_right_arm/right_arm_server.h"

const std::string name = "huroco_right_arm";


RightArmServer::RightArmServer(ArmManipulator *manipulator, moveit::planning_interface::MoveGroupInterface &group)
			:spinner_(1), controller_(manipulator)
{
	right_exec_ = nh_.advertiseService(name + "/right_exec", &RightArmServer::rightExec, this);
	right_cartesian_ = nh_.advertiseService(name + "/right_cartesian", &RightArmServer::rightCartesianPath, this);
	right_init_ = nh_.advertiseService(name + "/right_init", &RightArmServer::rightInit, this);

	controller_->setServer(this);

	spinner_.start();
}


RightArmServer::~RightArmServer() {}


bool RightArmServer::rightExec(huroco_right_arm::rightExec::Request &req,
							   huroco_right_arm::rightExec::Response &res)
{
	controller_->executeGoal(req.goal);
	return true;
}

bool RightArmServer::rightCartesianPath(huroco_right_arm::rightCartesian::Request &req,
										huroco_right_arm::rightCartesian::Response &res)
{
	res.status = controller_->executeCartesianPath(req.waypoints, req.step);
	return true;
}

bool RightArmServer::rightInit(huroco_right_arm::rightInit::Request &req,
							   huroco_right_arm::rightInit::Response &res)
{
	res.status = controller_->initPose();
	return true;
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "right_arm_server");


	moveit::planning_interface::MoveGroupInterface group("right_arm");
	ArmManipulator manipulator(group);
	RightArmServer right_server(&manipulator, group);

	//ros::spin();
	ros::waitForShutdown();

	return 0;
}