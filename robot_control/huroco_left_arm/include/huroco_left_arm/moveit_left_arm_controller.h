#ifndef LEFTARM_CONTROLLER_H
#define LEFTARM_CONTROLLER_H

#define ATTEMPTS 5

#include <ros/ros.h>
#include <std_msgs/String.h>

#include <string>
#include <vector>
#include <iterator>
#include <map>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>


class LeftArmServer;

class ArmManipulator
{
public:
	ArmManipulator(moveit::planning_interface::MoveGroupInterface &group);
	~ArmManipulator();

	void setServer(LeftArmServer *server);

	int executeGoal(const geometry_msgs::PoseStamped goal);

	int executeCartesianPath(const std::vector<geometry_msgs::Pose> waypoints, const double step);

	int rotateWrist(const double radian);

	void abortExecution();

private:
	void getLeftArmState(const sensor_msgs::JointState msg);

	int tryComputingCartesian(moveit::planning_interface::MoveGroupInterface &group, moveit_msgs::RobotTrajectory &trajectory, const double step, const std::vector<geometry_msgs::Pose> waypoints);

	int tryPlanning(moveit::planning_interface::MoveGroupInterface &group, moveit::planning_interface::MoveGroupInterface::Plan &plan);

	ros::NodeHandle nh_;

	std::map<std::string, double> left_joint_state_;

	bool sub_trigger_;

	moveit::planning_interface::MoveGroupInterface &move_group_;
	
	LeftArmServer *server_;
};

#endif