#include "action_manager/4dhoi.h"

#define FAIL    2
#define SUCCESS 3


const std::string name = "action_manager";


Action::Action()
			:spinner_(1)
{
	set_bottle_ = nh_.advertiseService(name + "/bottle", &Action::setBottles, this);
	action_execution_ = nh_.advertiseService(name + "/exec", &Action::executeActions, this);

	right_cartesian_ = nh_.serviceClient<huroco_right_arm::rightCartesian>("/huroco_right_arm/right_cartesian");
	left_cartesian_ = nh_.serviceClient<huroco_left_arm::leftCartesian>("/huroco_left_arm/left_cartesian");

	right_approach_ = nh_.serviceClient<huroco_grasping::graspPose>("/huroco_grasping/grasping_pose");
	left_approach_ = nh_.serviceClient<huroco_grasping::graspCap>("/huroco_grasping/cap_pose");

	right_grasp_ = nh_.serviceClient<right_gripper_receiver::rightGrasp>("/huroco_right_gripper/command");
	right_grasp_status_ = nh_.serviceClient<right_gripper_receiver::rightStatus>("/huroco_right_gripper/status");

	left_wrist_ = nh_.serviceClient<huroco_left_arm::rotateLeftWrist>("/huroco_left_arm/rotate_wrist");

	left_grasp_ = nh_.advertise<std_msgs::String>("/left_gripper", 1);

	right_init_ = nh_.serviceClient<huroco_right_arm::rightInit>("/huroco_right_arm/right_init");

	spinner_.start();
}


Action::~Action()
{

}  


void Action::getLeftPose(const baxter_core_msgs::EndpointState msg)
{
	sub_trigger_ = false;

	current_pose_.orientation.x = msg.pose.orientation.x;
	current_pose_.orientation.y = msg.pose.orientation.y;
	current_pose_.orientation.z = msg.pose.orientation.z;
	current_pose_.orientation.w = msg.pose.orientation.w;

	current_pose_.position.x = msg.pose.position.x;
	current_pose_.position.y = msg.pose.position.y;
	current_pose_.position.z = msg.pose.position.z;
}


void Action::approach(std::string bottle)
{
	huroco_grasping::graspPose right_approach_srv;
	huroco_grasping::graspCap left_approach_srv;
	huroco_right_arm::rightInit right_init_srv;
	huroco_right_arm::rightCartesian right_cartesian_srv;
	right_gripper_receiver::rightGrasp right_grasp_srv;
	right_gripper_receiver::rightStatus right_grasp_status_srv;

	int count = 0;
	while(right_init_.call(right_init_srv)) {
		count += 1;
		if(count > 5) return;

		right_approach_srv.request.bottle = bottle_;
		nh_.getParam("action_manager/" + bottle_, right_approach_srv.request.radius);

		printf("Init\n");

		right_grasp_srv.request.f1 = 0.5;
		right_grasp_srv.request.f2 = 0.5;
		right_grasp_srv.request.f3 = 0.5;

		right_grasp_.call(right_grasp_srv);
		sleep(3.5);

		if(right_approach_.call(right_approach_srv)) {
			if(right_approach_srv.response.status == false) continue;

			right_grasp_srv.request.f1 = 2.6;
			right_grasp_srv.request.f2 = 2.6;
			right_grasp_srv.request.f3 = 2.6;

			right_grasp_.call(right_grasp_srv);
			sleep(2.5);

			right_grasp_status_.call(right_grasp_status_srv);
			if(right_grasp_status_srv.response.status == false) continue;
		}
		else continue;

		printf("Successful grasp\n");

		geometry_msgs::Pose point;
		point.orientation.x = -0.50;
		point.orientation.y = 0.50;
		point.orientation.z = 0.50;
		point.orientation.w = 0.50;

		point.position.x = 0.574;
		point.position.y = 0.151;
		point.position.z = 0.259;

		right_cartesian_srv.request.waypoints.push_back(point);
		right_cartesian_.call(right_cartesian_srv);

		sleep(3.0);

		left_approach_srv.request.bottle = bottle_;
		if(left_approach_.call(left_approach_srv)) {

		}

		move();

		break;
	}
}


void Action::twist()
{
	huroco_left_arm::rotateLeftWrist wrist_srv;

	wrist_srv.request.radian = -3.0;
	if(left_wrist_.call(wrist_srv)) {
		ROS_INFO("Wrist execution status: %d", wrist_srv.response.status);
	}
}


void Action::move()
{
	huroco_left_arm::rotateLeftWrist wrist_srv;

	wrist_srv.request.radian = 0.0;
	if(left_wrist_.call(wrist_srv)) {
		ROS_INFO("Wrist execution status: %d", wrist_srv.response.status);
	}
}


void Action::pinch()
{

}

void Action::grasp()
{
	std_msgs::String msg;
	msg.data = 'c';
	left_grasp_.publish(msg);

	sleep(2.0);
}


void Action::ungrasp()
{
	std_msgs::String msg;
	msg.data = 'o';
	left_grasp_.publish(msg);

	sleep(2.0);
}


void Action::push()
{
	std_msgs::String msg;
	msg.data = 'c';
	left_grasp_.publish(msg);

	sleep(2.0);

	sub_trigger_ = true;
	robot_state_sub_ = nh_.subscribe("/robot/limb/left/endpoint_state", 1, &Action::getLeftPose, this);

	while(1) {
		if(sub_trigger_ == true) ros::spinOnce();
		else {
			robot_state_sub_.shutdown();
			break;
		}
	}

	current_pose_.position.z -= 0.152;

	huroco_left_arm::leftCartesian srv;
	srv.request.waypoints.push_back(current_pose_);

	if(left_cartesian_.call(srv)) {
		ROS_INFO("left arm execution status: %d", srv.response.status);
	}
}


void Action::neutral()
{
	sub_trigger_ = true;
	robot_state_sub_ = nh_.subscribe("/robot/limb/left/endpoint_state", 1, &Action::getLeftPose, this);

	while(1) {
		if(sub_trigger_ == true) ros::spinOnce();
		else {
			robot_state_sub_.shutdown();
			break;
		}
	}

	current_pose_.position.z -= 0.132;

	huroco_left_arm::leftCartesian srv;
	srv.request.waypoints.push_back(current_pose_);

	if(left_cartesian_.call(srv)) {
		ROS_INFO("left arm execution status: %d", srv.response.status);
	}	

	std_msgs::String msg;
	msg.data = 'o';
	left_grasp_.publish(msg);

	sleep(2.0);
}


void Action::pull()
{
	sub_trigger_ = true;
	robot_state_sub_ = nh_.subscribe("/robot/limb/left/endpoint_state", 1, &Action::getLeftPose, this);

	while(1) {
		if(sub_trigger_ == true) ros::spinOnce();
		else {
			robot_state_sub_.shutdown();
			break;
		}
	}

	current_pose_.position.z -= 0.052;

	huroco_left_arm::leftCartesian srv;
	srv.request.waypoints.push_back(current_pose_);

	if(left_cartesian_.call(srv)) {
		ROS_INFO("left arm execution status: %d", srv.response.status);
	}	
}


bool Action::setBottles(action_manager::setBottle::Request &req,
						action_manager::setBottle::Response &res)
{
	bottle_ = req.bottle;
}


bool Action::executeActions(action_manager::execAction::Request &req,
							action_manager::execAction::Response &res)
{
	if(req.action == "approach") approach(bottle_);
	else if(req.action == "twist") twist();
	else if(req.action == "move") move();
	else if(req.action == "pinch") pinch();
	else if(req.action == "unpinch") ungrasp();
	else if(req.action == "grasp") grasp();
	else if(req.action == "ungrasp") ungrasp();
	else if(req.action == "push") push();
	else if(req.action == "neutral") neutral();
	else if(req.action == "pull") pull();
	else {
		ROS_INFO("wrong action request");
		res.status = false;
		return true;
	}

	res.status = true;
	return true;
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "action_execution");

	Action action_execution;

	ROS_INFO("waiting for action execution request");
	ros::waitForShutdown();

	return 0;
}