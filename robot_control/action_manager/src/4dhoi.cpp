#include "action_manager/4dhoi.h"
#include <sstream>

#define PI 3.1415926

#define FAIL    2
#define SUCCESS 3


const std::string name = "action_manager";


Action::Action()
			:spinner_(1)
{
	set_object_ = nh_.advertiseService(name + "/object", &Action::setObject, this);
	action_execution_ = nh_.advertiseService(name + "/exec", &Action::executeActions, this);

	right_cartesian_ = nh_.serviceClient<huroco_right_arm::rightCartesian>("/huroco_right_arm/right_cartesian");
	left_cartesian_ = nh_.serviceClient<huroco_left_arm::leftCartesian>("/huroco_left_arm/left_cartesian");

	right_approach_ = nh_.serviceClient<huroco_grasping::graspPose>("/huroco_grasping/grasping_pose");
	left_approach_ = nh_.serviceClient<huroco_grasping::graspCap>("/huroco_grasping/cap_pose");
	left_door_ = nh_.serviceClient<huroco_grasping::graspDoor>("/huroco_grasping/door_pose");

	right_grasp_ = nh_.serviceClient<right_gripper_receiver::rightGrasp>("/huroco_right_gripper/command");
	right_grasp_status_ = nh_.serviceClient<right_gripper_receiver::rightStatus>("/huroco_right_gripper/status");

	left_wrist_ = nh_.serviceClient<huroco_left_arm::rotateLeftWrist>("/huroco_left_arm/rotate_wrist");

	left_grasp_ = nh_.advertise<std_msgs::String>("/left_gripper", 1);

	right_init_ = nh_.serviceClient<huroco_right_arm::rightInit>("/huroco_right_arm/right_init");
	left_init_ = nh_.serviceClient<huroco_left_arm::leftInit>("/huroco_left_arm/left_init");

	action_pub_ = nh_.advertise<std_msgs::String>("/execution_status", 1);

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


void Action::approach(std::string object)
{
	huroco_grasping::graspPose right_approach_srv;
	huroco_grasping::graspCap left_approach_srv;
	huroco_right_arm::rightInit right_init_srv;
	huroco_right_arm::rightCartesian right_cartesian_srv;
	right_gripper_receiver::rightGrasp right_grasp_srv;
	right_gripper_receiver::rightStatus right_grasp_status_srv;

	int count = 0;
	//while(right_init_.call(right_init_srv)) {
	while(count < 5) {
		count += 1;
		//if(count > 5) return;

		right_approach_srv.request.bottle = object_;
		nh_.getParam("action_manager/" + object_, right_approach_srv.request.radius);

		printf("Init\n");

		right_grasp_srv.request.f1 = 0.5;
		right_grasp_srv.request.f2 = 0.5;
		right_grasp_srv.request.f3 = 0.5;

		ungrasp();

		right_grasp_.call(right_grasp_srv);

		//if(right_approach_.call(right_approach_srv)) {
		if(true) {
			//if(right_approach_srv.response.status == false) continue;

			right_grasp_srv.request.f1 = 2.8;
			right_grasp_srv.request.f2 = 2.8;
			right_grasp_srv.request.f3 = 2.8;

			printf("Grasp\n");
			sleep(2.5);

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

		sleep(1.5);

		left_approach_srv.request.bottle = object_;
		if(left_approach_.call(left_approach_srv)) {

		}

		sleep(5.5);

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

	sleep(6.0);
}


void Action::move()
{
	
	huroco_left_arm::rotateLeftWrist wrist_srv;

	wrist_srv.request.radian = 0.0;
	if(left_wrist_.call(wrist_srv)) {
		ROS_INFO("Wrist execution status: %d", wrist_srv.response.status);
	}
	
	sleep(6.0);
}


void Action::pinch()
{
	grasp();
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
	grasp();

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
		ROS_INFO("push: left arm execution status: %d", srv.response.status);
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
		ROS_INFO("push: left arm execution status: %d", srv.response.status);
	}	

	sleep(1.5);

	ungrasp();
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


void Action::openDoor() 
{
	ungrasp();

	huroco_grasping::graspDoor left_door_srv;
	huroco_left_arm::leftInit left_init_srv;
	huroco_left_arm::leftCartesian left_cartesian_srv;

	geometry_msgs::Pose door_handle;
	left_door_srv.request.door = object_;

	if(left_door_.call(left_door_srv)) {
		door_handle.orientation.x = left_door_srv.response.door_handle_pose.orientation.x;
		door_handle.orientation.y = left_door_srv.response.door_handle_pose.orientation.y;
		door_handle.orientation.z = left_door_srv.response.door_handle_pose.orientation.z;
		door_handle.orientation.w = left_door_srv.response.door_handle_pose.orientation.w;
	
		door_handle.position.x = left_door_srv.response.door_handle_pose.position.x;
		door_handle.position.y = left_door_srv.response.door_handle_pose.position.y;
		door_handle.position.z = left_door_srv.response.door_handle_pose.position.z;
	}

	door_handle.position.z += 0.15;
	left_cartesian_srv.request.waypoints.push_back(door_handle);

	door_handle.position.z -= 0.15;
	left_cartesian_srv.request.waypoints.push_back(door_handle);

	if(left_cartesian_.call(left_cartesian_srv)) {
		ROS_INFO("left arm execution status: %d", left_cartesian_srv.response.status);
		left_cartesian_srv.request.waypoints.clear();
	}
	sleep(6.0);
	grasp();

	double yaw = 0;
	bool tf_status_ = true;

	for(yaw = 0; yaw < PI/2 * 0.75; yaw += PI/20) {
		printf("yaw: %f\n", yaw);

		tf::Quaternion axis_yaw(0, 0, 0, 1);
		tf::Vector3 axis_pos(0, 0.26, 0.19);
		tf::Transform axis_tf(axis_yaw, axis_pos);

		tf::Quaternion grasping_yaw(tf::Vector3(0, 0, 1), yaw);
		tf::Vector3 grasping_pos(0, 0, 0);
		tf::Transform grasping_tf(grasping_yaw, grasping_pos);

		tf::Quaternion trans_q(0, 0, 0, 1);
		tf::Vector3 trans_pos(0.03, -0.52, 0);
		tf::Transform trans_tf(trans_q, trans_pos);

		tf::Quaternion yaw_q(tf::Vector3(0, 0, 1), PI/4 * 3);
		tf::Vector3 yaw_pos(0, 0, 0);
		tf::Transform yaw_tf(yaw_q, yaw_pos);

		tf::Quaternion y_rotation(tf::Vector3(0, 1, 0), PI);
		tf::Transform y_rotation_tf(y_rotation, tf::Vector3(0 ,0, 0));

		tf::Transform open_pose_tf = axis_tf * grasping_tf * trans_tf * yaw_tf * y_rotation_tf;
		//tf::Transform open_pose_tf = grasping_tf * y_rotation_tf;

		tf::TransformListener listener;
		tf::StampedTransform transform;

		while(ros::ok()) {
			ros::Time t = ros::Time(0);

			try {
				listener.waitForTransform("/base", object_, t, ros::Duration(1.0));
				listener.lookupTransform("/base", object_, t, transform);
			}
			catch(tf::TransformException ex) {
				ROS_ERROR("%s", ex.what());
				tf_status_ = false;
			}

			tf::Transform pose = transform * open_pose_tf;
			door_tf_ = transform;

			geometry_msgs::Pose open_pose;

			open_pose.orientation.x = pose.getRotation().x();
			open_pose.orientation.y = pose.getRotation().y();
			open_pose.orientation.z = pose.getRotation().z();
			open_pose.orientation.w = pose.getRotation().w();

			open_pose.position.x = pose.getOrigin().getX();
			open_pose.position.y = pose.getOrigin().getY();
			open_pose.position.z = pose.getOrigin().getZ();

			printf("qx:%f, qy:%f, qz:%f, qw:%f\n", open_pose.orientation.x, open_pose.orientation.y, open_pose.orientation.z, open_pose.orientation.w);
			printf("x:%f, y:%f, z:%f\n", open_pose.position.x, open_pose.position.y, open_pose.position.z);

			left_cartesian_srv.request.waypoints.push_back(open_pose);

			break;
		}
	}

	curve_ = left_cartesian_srv.request.waypoints;
	if(left_cartesian_.call(left_cartesian_srv)) {
		ROS_INFO("left arm execution status: %d", left_cartesian_srv.response.status);
		left_cartesian_srv.request.waypoints.clear();
	}

	sleep(12.0);

	ungrasp();

	geometry_msgs::Pose leave_pose;
	leave_pose.orientation.x = curve_.back().orientation.x;
	leave_pose.orientation.y = curve_.back().orientation.y;
	leave_pose.orientation.z = curve_.back().orientation.z;
	leave_pose.orientation.w = curve_.back().orientation.w;

	leave_pose.position.x = curve_.back().position.x;
	leave_pose.position.y = curve_.back().position.y;
	leave_pose.position.z = curve_.back().position.z + 0.17;

	left_cartesian_srv.request.waypoints.push_back(leave_pose);

	if(left_cartesian_.call(left_cartesian_srv)) {
		ROS_INFO("left arm execution status: %d", left_cartesian_srv.response.status);
		left_cartesian_srv.request.waypoints.clear();
	}

	sleep(2.0);

	left_init_.call(left_init_srv); 
}


void Action::closeDoor()
{
	huroco_right_arm::rightInit right_init_srv;
	huroco_left_arm::leftInit left_init_srv;
	huroco_right_arm::rightCartesian right_cartesian_srv;
	right_gripper_receiver::rightGrasp right_grasp_srv;

	right_grasp_srv.request.f1 = 3.0;
	right_grasp_srv.request.f2 = 3.0;
	right_grasp_srv.request.f3 = 3.0;

	right_grasp_.call(right_grasp_srv);
	sleep(2.0);

	right_init_.call(right_init_srv);
	sleep(3.0);

	tf::Quaternion yaw_q(tf::Vector3(0, 0, 1), PI);
	tf::Vector3 yaw_pos(0, 0, 0);
	tf::Transform yaw_tf(yaw_q, yaw_pos);

	tf::Quaternion yaw_q1(tf::Vector3(0, 0, 1), PI/6);
	tf::Vector3 yaw_pos1(0, 0, 0);
	tf::Transform yaw_tf1(yaw_q1, yaw_pos1);

	tf::Quaternion pitch_q(tf::Vector3(0, 1, 0), PI/2 + PI/9);
	tf::Vector3 pitch_pos(0, 0, 0);
	tf::Transform pitch_tf(pitch_q, pitch_pos);

	tf::Transform push_tf = door_tf_ * yaw_tf * yaw_tf1 * pitch_tf;

	printf("empty: %d\n", curve_.empty());
	geometry_msgs::Pose push_pose;

	int i = 0;
	while(!curve_.empty()) {
		push_pose.orientation.x = push_tf.getRotation().x();
		push_pose.orientation.y = push_tf.getRotation().y();
		push_pose.orientation.z = push_tf.getRotation().z();
		push_pose.orientation.w = push_tf.getRotation().w();

		tf::Quaternion open_q(curve_.back().orientation.x, curve_.back().orientation.y, curve_.back().orientation.z, curve_.back().orientation.w);
		tf::Vector3 open_pos(curve_.back().position.x, curve_.back().position.y, curve_.back().position.z);
		tf::Transform open_tf(open_q, open_pos);

		tf::Quaternion trans_q(0, 0, 0, 1);
		tf::Vector3 trans_pos(0.15, -0.10, 0);
		tf::Transform trans_tf(trans_q, trans_pos);

		tf::Transform push_pos_tf = trans_tf * open_tf;

		push_pose.position.x = push_pos_tf.getOrigin().getX();
		push_pose.position.y = push_pos_tf.getOrigin().getY();
		push_pose.position.z = push_pos_tf.getOrigin().getZ();

		printf("qx:%f, qy:%f, qz:%f, qw:%f\n", push_pose.orientation.x, push_pose.orientation.y, push_pose.orientation.z, push_pose.orientation.w);
		printf("x:%f, y:%f, z:%f\n", push_pose.position.x, push_pose.position.y, push_pose.position.z);

		right_cartesian_srv.request.waypoints.push_back(push_pose);
		curve_.pop_back();
		i++;
	}

	//right_cartesian_srv.request.waypoints.push_back(push_pose);
	if(right_cartesian_.call(right_cartesian_srv)) {
		ROS_INFO("right arm execution status: %d", right_cartesian_srv.response.status);
		right_cartesian_srv.request.waypoints.clear();
	}
}


void Action::fetch()
{
	ungrasp();
	huroco_left_arm::leftCartesian left_cartesian_srv;

	tf::TransformListener listener;
	tf::StampedTransform transform;

	geometry_msgs::Pose fetch_pose;
	while(ros::ok()) {
		ros::Time t = ros::Time(0);

		try {
			listener.waitForTransform("/base", "eraser", t, ros::Duration(1.0));
			listener.lookupTransform("/base", "eraser", t, transform);
		}
		/*
		try {
			listener.waitForTransform("/base", "coke", t, ros::Duration(1.0));
			listener.lookupTransform("/base", "coke", t, transform);
		}
		*/
		catch(tf::TransformException ex) {
			ROS_ERROR("%s", ex.what());
		}

		fetch_pose.orientation.x = 0.07;
		fetch_pose.orientation.y = 0.98;
		fetch_pose.orientation.z = -0.01;
		fetch_pose.orientation.w = 0.17;

		fetch_pose.position.x = transform.getOrigin().getX();
		fetch_pose.position.y = transform.getOrigin().getY();
		fetch_pose.position.z = transform.getOrigin().getZ() + 0.06;

		left_cartesian_srv.request.waypoints.push_back(fetch_pose);

		fetch_pose.position.z = transform.getOrigin().getZ();
		left_cartesian_srv.request.waypoints.push_back(fetch_pose);

		break;
	}

	if(left_cartesian_.call(left_cartesian_srv)) {
		ROS_INFO("left arm execution status: %d", left_cartesian_srv.response.status);
		left_cartesian_srv.request.waypoints.clear();
	}

	sleep(8.0);

	grasp();

	fetch_pose.position.z += 0.06;
	left_cartesian_srv.request.waypoints.push_back(fetch_pose);

	if(left_cartesian_.call(left_cartesian_srv)) {
		ROS_INFO("left arm execution status: %d", left_cartesian_srv.response.status);
		left_cartesian_srv.request.waypoints.clear();
	}
}


void Action::point() 
{
	geometry_msgs::Pose point_pose;

	point_pose.orientation.x = 0.07;
	point_pose.orientation.y = 0.73;
	point_pose.orientation.z = -0.15;
	point_pose.orientation.w = 0.66;

	point_pose.position.x = 1.15;
	point_pose.position.y = 0.04;
	point_pose.position.z = 0.36;

	huroco_left_arm::leftCartesian srv;
	srv.request.waypoints.push_back(point_pose);

	if(left_cartesian_.call(srv)) {
		ROS_INFO("left arm execution status: %d", srv.response.status);
	}

	sleep(5.0);

	ungrasp();
}


void Action::pour()
{
	geometry_msgs::Pose pour_pose;

	pour_pose.orientation.x = -0.21;
	pour_pose.orientation.y = 0.67;
	pour_pose.orientation.z = 0.18;
	pour_pose.orientation.w = 0.68;

	pour_pose.position.x = 0.85;
	pour_pose.position.y = -0.06;
	pour_pose.position.z = 0.40;

	huroco_right_arm::rightCartesian srv;
	srv.request.waypoints.push_back(pour_pose);

	pour_pose.orientation.x = 0.64;
	pour_pose.orientation.y = -0.17;
	pour_pose.orientation.z = 0.60;
	pour_pose.orientation.w = -0.43;

	pour_pose.position.x = 0.85;
	pour_pose.position.y = -0.06;
	pour_pose.position.z = 0.45;

	srv.request.waypoints.push_back(pour_pose);

	pour_pose.orientation.x = -0.21;
	pour_pose.orientation.y = 0.67;
	pour_pose.orientation.z = 0.18;
	pour_pose.orientation.w = 0.68;

	pour_pose.position.x = 0.85;
	pour_pose.position.y = -0.06;
	pour_pose.position.z = 0.40;

	srv.request.waypoints.push_back(pour_pose);

	if(right_cartesian_.call(srv)) {
		ROS_INFO("right arm execution status: %d", srv.response.status);
	}

	return;
}


void Action::officedoor()
{
	huroco_left_arm::leftCartesian left_cartesian_srv;

	ungrasp();

	tf::TransformListener listener;
	tf::StampedTransform transform;

	geometry_msgs::Pose door_pose;
	while(ros::ok()) {
		ros::Time t = ros::Time(0);

		try {
			listener.waitForTransform("/base", "door", t, ros::Duration(1.0));
			listener.lookupTransform("/base", "door", t, transform);
		}
		catch(tf::TransformException ex) {
			ROS_ERROR("%s", ex.what());
		}

		door_pose.orientation.x = transform.getRotation().x();
		door_pose.orientation.y = transform.getRotation().y();
		door_pose.orientation.z = transform.getRotation().z();
		door_pose.orientation.w = transform.getRotation().w();

		door_pose.position.x = transform.getOrigin().getX() - 0.15;
		door_pose.position.y = transform.getOrigin().getY();
		door_pose.position.z = transform.getOrigin().getZ();

		left_cartesian_srv.request.waypoints.push_back(door_pose);

		break;
	}

	if(left_cartesian_.call(left_cartesian_srv)) {
		ROS_INFO("left arm execution status: %d", left_cartesian_srv.response.status);
		left_cartesian_srv.request.waypoints.clear();
	}

	sleep(4.0);

	door_pose.position.x += 0.15;

	left_cartesian_srv.request.waypoints.push_back(door_pose);
	if(left_cartesian_.call(left_cartesian_srv)) {
		ROS_INFO("left arm execution status: %d", left_cartesian_srv.response.status);
		left_cartesian_srv.request.waypoints.clear();
	}

	sleep(2.0);

	grasp();
	twist();
	ungrasp();
}


bool Action::setObject(action_manager::setObject::Request &req,
						action_manager::setObject::Response &res)
{
	object_ = req.object;
}


bool Action::executeActions(action_manager::execAction::Request &req,
							action_manager::execAction::Response &res)
{
	printf("%s\n", req.action.c_str());
	std_msgs::String msg;
	msg.data = req.action;
	action_pub_.publish(msg);

	if(req.action == "approach") approach(object_);
	else if(req.action == "twist") twist();
	else if(req.action == "move") move();
	else if(req.action == "pinch") pinch();
	else if(req.action == "unpinch") ungrasp();
	else if(req.action == "grasp_right") grasp();
	else if(req.action == "ungrasp_right") ungrasp();
	else if(req.action == "push") push();
	else if(req.action == "neutral") neutral();
	else if(req.action == "pull") pull();
	else if(req.action == "opendoor") openDoor();
	else if(req.action == "closedoor") closeDoor();
	else if(req.action == "fetch") fetch();
	else if(req.action == "point") point();
	else if(req.action == "pour") pour();
	else if(req.action == "officedoor") officedoor();
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

	//ros::spin();
	ros::waitForShutdown();

	return 0;
}