#include "huroco_left_arm/moveit_left_arm_controller.h"
#include "huroco_left_arm/left_arm_server.h"

#define SUCCESS_ 1
#define FAILURE_ 0

ArmManipulator::ArmManipulator(
		moveit::planning_interface::MoveGroupInterface &group)
		: move_group_(group)
{
	move_group_.setPlannerId("RRTConnectkConfigDefault");
	move_group_.setPlanningTime(5.0);
	move_group_.allowReplanning(true);
	move_group_.setMaxVelocityScalingFactor(0.75);
	move_group_.setMaxAccelerationScalingFactor(1.0);

	exec_status_sub_ =
			nh_.subscribe("/robot/limb/left/follow_joint_trajectory/result", 1000,
										&ArmManipulator::getExecStatus, this);
	joint_sub_ = nh_.subscribe("/robot/joint_states", 1000,
														 &ArmManipulator::getLeftArmState, this);
}

ArmManipulator::~ArmManipulator() {}

void ArmManipulator::setServer(LeftArmServer *server) { server_ = server; }

int ArmManipulator::tryComputingCartesian(
		moveit::planning_interface::MoveGroupInterface &group,
		moveit_msgs::RobotTrajectory &trajectory, const double step,
		const std::vector<geometry_msgs::Pose> waypoints)
{
	double fraction = 0;
	int computed_time = 0;

	while (computed_time < ATTEMPTS)
	{
		if (fraction < 1)
		{
			fraction = group.computeCartesianPath(waypoints, step, 0.0, trajectory);
			ROS_INFO("Waypoints that are computed successfully: %f", fraction * 100);
		}
		else if (fraction == 1)
		{
			ROS_INFO("Successfully compute complete cartesian path.");
			return SUCCESS_;
		}

		computed_time += 1;
	}
	return FAILURE_;
}

moveit::planning_interface::MoveItErrorCode ArmManipulator::tryPlanning(
		moveit::planning_interface::MoveGroupInterface &group,
		moveit::planning_interface::MoveGroupInterface::Plan &plan)
{
	int planning_times = 0;
	moveit::planning_interface::MoveItErrorCode planning_status;

	while (ros::ok() && planning_times < ATTEMPTS)
	{
		planning_times += 1;
		planning_status = group.plan(plan);
		if (planning_status == moveit_msgs::MoveItErrorCodes::SUCCESS)
		{
			ROS_INFO("Generate a plan successfully.");
			return planning_status;
		}
		else
		{
			ROS_ERROR("Fail to generate a plan, please check the pose, %d times try",
								planning_times);
			if (planning_times >= ATTEMPTS)
				return planning_status;
		}
	}
}

void ArmManipulator::getLeftArmState(const sensor_msgs::JointState msg)
{
	left_joint_state_["left_s0"] = msg.position[4];
	left_joint_state_["left_s1"] = msg.position[5];
	left_joint_state_["left_e0"] = msg.position[2];
	left_joint_state_["left_e1"] = msg.position[3];
	left_joint_state_["left_w0"] = msg.position[6];
	left_joint_state_["left_w1"] = msg.position[7];
	left_joint_state_["left_w2"] = msg.position[8];

	sub_trigger_ = false;
}

void ArmManipulator::getExecStatus(
		const control_msgs::FollowJointTrajectoryActionResult msg)
{
	exec_status_ = msg.status.status;

	std::cout << "Updated exec_status_ to " << exec_status_ << std::endl;

	sub_trigger_ = false;
}

moveit::planning_interface::MoveItErrorCode
ArmManipulator::rotateWrist(const double radian)
{
	sub_trigger_ = true;

	while (sub_trigger_)
	{
		if (sub_trigger_ == true)
			ros::spinOnce();
		else
		{
			joint_sub_.shutdown();
			break;
		}
	}

	std::map<std::string, double> joint_values;
	joint_values = left_joint_state_;

	joint_values["left_w2"] = radian;

	move_group_.setJointValueTarget(joint_values);
	moveit::planning_interface::MoveGroupInterface::Plan plan;

	moveit::planning_interface::MoveItErrorCode joint_space_status =
			tryPlanning(move_group_, plan);

	sub_trigger_ = true;

	if (joint_space_status == moveit_msgs::MoveItErrorCodes::SUCCESS)
	{

		move_group_.asyncExecute(plan);

		while (1)
		{
			if (sub_trigger_ == true)
				ros::spinOnce();
			else
			{
				exec_status_sub_.shutdown();
				break;
			}
		}

		sleep(1.0);

		return exec_status_;
	}
	else
	{
		ROS_ERROR("Invalid rotation");
		return joint_space_status;
	}
}

int ArmManipulator::executeCartesianPath(
		std::vector<geometry_msgs::Pose> waypoints)
{
	sub_trigger_ = true;

	moveit_msgs::RobotTrajectory trajectory;
	moveit::planning_interface::MoveGroupInterface::Plan plan;

	int cartesian_status;

	/* Robotiq TF */
	tf::Quaternion robotiq_q(0, 0, 0, 1);
	tf::Vector3 robotiq_t(0.0, 0.0, -0.142);

	tf::Transform tf(robotiq_q, robotiq_t);

	std::vector<geometry_msgs::Pose>::iterator iter;
	for (iter = waypoints.begin(); iter != waypoints.end(); iter++)
	{
		tf::Quaternion robotiq_qt(iter->orientation.x, iter->orientation.y,
															iter->orientation.z, iter->orientation.w);
		tf::Vector3 robotiq_tt(iter->position.x, iter->position.y,
													 iter->position.z);

		tf::Transform tf_t(robotiq_qt, robotiq_tt);

		tf::Vector3 left_eef = tf_t * robotiq_t;

		iter->position.x = left_eef.getX();
		iter->position.y = left_eef.getY();
		iter->position.z = left_eef.getZ();
	}

	double step = 0.001;
	// nh_.getParam("step", step);
	// printf("%f", step);

	cartesian_status =
			tryComputingCartesian(move_group_, trajectory, step, waypoints);
	if (cartesian_status)
	{
		plan.trajectory_ = trajectory;
		ROS_INFO("Computed cartesian path successfully, trying to execute.");
		move_group_.asyncExecute(plan);

		while (1)
		{
			if (sub_trigger_ == true)
				ros::spinOnce();
			else
			{
				exec_status_sub_.shutdown();
				break;
			}
		}

		return exec_status_;
	}
	else
	{
		ROS_ERROR("Not able to compute complete cartesian path with given "
							"waypoints, check your waypoints.");
		return cartesian_status;
	}
}

moveit::planning_interface::MoveItErrorCode ArmManipulator::initPose()
{
	std::map<std::string, double> joint_values;
	joint_values["left_s0"] = 0.96;
	joint_values["left_s1"] = -0.88;
	joint_values["left_e0"] = -0.91;
	joint_values["left_e1"] = 2.44;
	joint_values["left_w0"] = 1.297;
	joint_values["left_w1"] = -0.94;
	joint_values["left_w2"] = 0.04;

	move_group_.setJointValueTarget(joint_values);
	moveit::planning_interface::MoveGroupInterface::Plan plan;

	moveit::planning_interface::MoveItErrorCode joint_space_status = tryPlanning(move_group_, plan);
	moveit::planning_interface::MoveItErrorCode execute_status;

	if (joint_space_status == moveit_msgs::MoveItErrorCodes::SUCCESS)
	{
		ROS_INFO("Rotating joints");
		execute_status = move_group_.execute(plan);
		return execute_status;
	}
	else
	{
		ROS_ERROR("Invalid rotation");
		return joint_space_status;
	}
}

void ArmManipulator::abortExecution() { move_group_.stop(); }
