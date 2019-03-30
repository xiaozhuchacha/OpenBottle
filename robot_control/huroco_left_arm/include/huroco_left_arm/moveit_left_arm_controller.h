#ifndef LEFTARM_CONTROLLER_H
#define LEFTARM_CONTROLLER_H

#define ATTEMPTS 5

#include <control_msgs/FollowJointTrajectoryActionResult.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

#include <iterator>
#include <map>
#include <string>
#include <vector>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <tf/tf.h>

class LeftArmServer;

class ArmManipulator
{
public:
  ArmManipulator(moveit::planning_interface::MoveGroupInterface &group);
  ~ArmManipulator();

  void setServer(LeftArmServer *server);

  void getExecStatus(const control_msgs::FollowJointTrajectoryActionResult msg);

  int executeCartesianPath(std::vector<geometry_msgs::Pose> waypoints);

  moveit::planning_interface::MoveItErrorCode rotateWrist(const double radian);

  void abortExecution();

  moveit::planning_interface::MoveItErrorCode initPose();

private:
  void getLeftArmState(const sensor_msgs::JointState msg);

  int tryComputingCartesian(
      moveit::planning_interface::MoveGroupInterface &group,
      moveit_msgs::RobotTrajectory &trajectory, const double step,
      const std::vector<geometry_msgs::Pose> waypoints);

  moveit::planning_interface::MoveItErrorCode
  tryPlanning(moveit::planning_interface::MoveGroupInterface &group,
              moveit::planning_interface::MoveGroupInterface::Plan &plan);

  ros::NodeHandle nh_;

  ros::Subscriber joint_sub_;
  ros::Subscriber exec_status_sub_;

  std::map<std::string, double> left_joint_state_;

  moveit::planning_interface::MoveGroupInterface &move_group_;

  LeftArmServer *server_;

  bool sub_trigger_;

  bool stop_sign_;

  int exec_status_;
};

#endif