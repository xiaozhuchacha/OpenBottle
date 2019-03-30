#ifndef RIGHTARM_CONTROLLER_H
#define RIGHTARM_CONTROLLER_H

#define ATTEMPTS 5

#include <control_msgs/FollowJointTrajectoryActionResult.h>
#include <ros/ros.h>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <string>
#include <tf/transform_broadcaster.h>
#include <vector>

class RightArmServer;

class ArmManipulator {
public:
  ArmManipulator(moveit::planning_interface::MoveGroupInterface &group);
  ~ArmManipulator();

  void setServer(RightArmServer *server);

  void getExecStatus(const control_msgs::FollowJointTrajectoryActionResult msg);

  int executeCartesianPath(std::vector<geometry_msgs::Pose> waypoints);

  moveit::planning_interface::MoveItErrorCode initPose();

  void abortExecution();

private:
  int tryComputingCartesian(
      moveit::planning_interface::MoveGroupInterface &group,
      moveit_msgs::RobotTrajectory &trajectory, const double step,
      const std::vector<geometry_msgs::Pose> waypoints);

  moveit::planning_interface::MoveItErrorCode
  tryPlanning(moveit::planning_interface::MoveGroupInterface &group,
              moveit::planning_interface::MoveGroupInterface::Plan &plan);

  ros::NodeHandle nh_;

  ros::Subscriber exec_status_sub_;

  moveit::planning_interface::MoveGroupInterface &move_group_;

  RightArmServer *server_;

  int exec_status_;

  bool sub_trigger_;
};

#endif