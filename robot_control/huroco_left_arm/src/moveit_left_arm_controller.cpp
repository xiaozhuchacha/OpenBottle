#include "huroco_left_arm/moveit_left_arm_controller.h"
#include "huroco_left_arm/left_arm_server.h"

#define SUCCESS 1
#define FAILURE 0


ArmManipulator::ArmManipulator(moveit::planning_interface::MoveGroupInterface &group)
						:move_group_(group)
{
	move_group_.setPlannerId("RRTConnectkConfigDefault");
	move_group_.setPlanningTime(5.0);
	move_group_.allowReplanning(true);
	move_group_.setMaxVelocityScalingFactor(0.5);
	move_group_.setMaxAccelerationScalingFactor(0.5);
}


ArmManipulator::~ArmManipulator()
{

}

void ArmManipulator::setServer(LeftArmServer *server)
{
	server_ = server;
}


int ArmManipulator::tryComputingCartesian(moveit::planning_interface::MoveGroupInterface &group, moveit_msgs::RobotTrajectory &trajectory, const double step, const std::vector<geometry_msgs::Pose> waypoints)
{
	double fraction = 0;
	int computed_time = 0;

	while(computed_time < ATTEMPTS) {
		if(fraction < 1) {
			fraction = group.computeCartesianPath(waypoints, step, 0.0, trajectory);
			ROS_INFO("Waypoints that are computed successfully: %f", fraction * 100);
		}
		else if(fraction == 1) {
			ROS_INFO("Successfully compute complete cartesian path.");
			return SUCCESS;
		}

		computed_time += 1;
	}
	return FAILURE;
}


int ArmManipulator::tryPlanning(moveit::planning_interface::MoveGroupInterface &group, moveit::planning_interface::MoveGroupInterface::Plan &plan)
{
	int planning_times = 0;
	int planning_status;

	while(ros::ok() && planning_times < ATTEMPTS) {
		planning_times += 1;
		planning_status = group.plan(plan);
		if(planning_status) {
			ROS_INFO("Generate a plan successfully.");
			return planning_status;
		}
		else {
			ROS_ERROR("Fail to generate a plan, please check the pose, %d times try", planning_times);
			if(planning_times >= ATTEMPTS) return planning_status;
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


int ArmManipulator::rotateWrist(const double radian)
{
	sub_trigger_ = true;

	ros::Subscriber sub = nh_.subscribe("/robot/joint_states", 1, &ArmManipulator::getLeftArmState, this);

	while(sub_trigger_) {
		ros::spinOnce();
	}

	std::map<std::string, double> joint_values;
	joint_values = left_joint_state_;

	joint_values["left_w2"] = radian;

	move_group_.setJointValueTarget(joint_values);
	moveit::planning_interface::MoveGroupInterface::Plan plan;

	int joint_space_status = tryPlanning(move_group_, plan);
	int execute_status;

	if(joint_space_status) {
		ROS_INFO("Rotating joints");
		execute_status = move_group_.execute(plan);
		return execute_status;
	}
	else {
		ROS_ERROR("Invalid rotation");
		return joint_space_status;
	}
}


int ArmManipulator::executeCartesianPath(std::vector<geometry_msgs::Pose> waypoints, const double step)
{
	moveit_msgs::RobotTrajectory trajectory;
	moveit::planning_interface::MoveGroupInterface::Plan plan;

	int cartesian_status;
	int execute_status;
/*
	tf::Quaternion robotiq_q(0, 0, 0, 1);
	tf::Vector3 robotiq_t(0.0, 0.0, -0.142);

	tf::Transform tf(robotiq_q, robotiq_t);


	geometry_msgs::Pose point = waypoints[0];

	tf::Quaternion robotiq_qt(point.orientation.x, point.orientation.y, point.orientation.z, point.orientation.w);
	tf::Vector3 robotiq_tt(point.position.x, point.position.y, point.position.z);

	tf::Transform tf_t(robotiq_qt, robotiq_tt);

	tf::Vector3 left_eef = tf_t * robotiq_t;

	waypoints[0].position.x = left_eef.getX();
	waypoints[0].position.y = left_eef.getY();
	waypoints[0].position.z = left_eef.getZ();
*/
	cartesian_status = tryComputingCartesian(move_group_, trajectory, step, waypoints);
	if(cartesian_status) {
		plan.trajectory_ = trajectory;
		ROS_INFO("Computed cartesian path successfully, trying to execute.");
		execute_status = move_group_.execute(plan);
		return execute_status;
	}
	else {
		ROS_ERROR("Not able to compute complete cartesian path with given waypoints, check your waypoints.");
		return cartesian_status;
	}

}


int ArmManipulator::executeGoal(const geometry_msgs::PoseStamped goal)
{
	move_group_.setPoseTarget(goal.pose, "left_gripper");
	moveit::planning_interface::MoveGroupInterface::Plan plan;
	if(tryPlanning(move_group_, plan)) {
		ROS_INFO("Now execute goal");
		move_group_.execute(plan);
		move_group_.clearPoseTarget();
	}
	else {
		ROS_INFO("No execution attempt");
		move_group_.clearPoseTarget();
	}

}


void ArmManipulator::abortExecution()
{
	move_group_.stop();
}