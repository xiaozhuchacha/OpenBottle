#ifndef RIGHTARM_CONTROLLER_H
#define RIGHTARM_CONTROLLER_H

#define ATTEMPTS 5

#include <ros/ros.h>
#include <vector>
#include <string>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <tf/transform_broadcaster.h>


class RightArmServer;

class ArmManipulator
{
public:
	ArmManipulator(moveit::planning_interface::MoveGroupInterface &group);
	~ArmManipulator();

	void setServer(RightArmServer *server);

	int executeGoal(const geometry_msgs::PoseStamped goal);

	int executeCartesianPath(const std::vector<geometry_msgs::Pose> waypoints, const double step);

	int initPose();

private:
	int tryComputingCartesian(moveit::planning_interface::MoveGroupInterface &group, moveit_msgs::RobotTrajectory &trajectory, const double step, const std::vector<geometry_msgs::Pose> waypoints);

	int tryPlanning(moveit::planning_interface::MoveGroupInterface &group, moveit::planning_interface::MoveGroupInterface::Plan &plan);

	ros::NodeHandle nh_;

	moveit::planning_interface::MoveGroupInterface &move_group_;

	RightArmServer *server_;
};

#endif