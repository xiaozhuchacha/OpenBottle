#include "huroco_grasping/grasping_point.h"


#define PI 3.1415926 


const std::string name = "/huroco_grasping";


Grasp::Grasp()
		:spinner_(1), nh_(ros::NodeHandle("~"))
{
	cap_ = nh_.advertiseService(name + "/cap_pose", &Grasp::graspingCap, this);
	grasping_ = nh_.advertiseService(name + "/grasping_pose", &Grasp::graspingPose, this);
	door_ = nh_.advertiseService(name + "/door_pose", &Grasp::graspingDoor, this);

	spinner_.start();
}


Grasp::~Grasp()
{

}

void Grasp::normalizeVect(Vect &norm_vect)
{
	tf::TransformListener listener;
	tf::StampedTransform transform;
	tf::StampedTransform lower_forearm_transform;
	tf::StampedTransform upper_forearm_transform;

	while(ros::ok()) {
		ros::Time t = ros::Time(0);

		try {
			listener.waitForTransform(bottle_, "/base", t, ros::Duration(1.0));
			listener.lookupTransform(bottle_, "/base", t, transform);
		}
		catch(tf::TransformException ex) {
			ROS_ERROR("%s", ex.what());
			tf_status_ = false;
		}

		try {
			listener.waitForTransform("/right_lower_forearm", "/base", t, ros::Duration(1.0));
			listener.lookupTransform("/right_lower_forearm", "/base", t, lower_forearm_transform);
		}
		catch(tf::TransformException ex) {
			ROS_ERROR("%s", ex.what());
			tf_status_ = false;
		}

		try {
			listener.waitForTransform("/right_upper_forearm", "/base", t, ros::Duration(1.0));
			listener.lookupTransform("/right_upper_forearm", "/base", t, upper_forearm_transform);
		}
		catch(tf::TransformException ex) {
			ROS_ERROR("%s", ex.what());
			tf_status_ = false;
		}


		tf::Vector3 lower_forearm_pos = lower_forearm_transform.inverse() * tf::Vector3(0, 0, 0);
		tf::Vector3 upper_forearm_pos = upper_forearm_transform.inverse() * tf::Vector3(0, 0, 0);


		/* get position of upper_forearm */
		double upper_forearm_x = upper_forearm_pos.getX();
		double upper_forearm_y = upper_forearm_pos.getY();
		double upper_forearm_z = upper_forearm_pos.getZ();

		/* get position of lower_forearm */
		double lower_forearm_x = lower_forearm_pos.getX();
		double lower_forearm_y = lower_forearm_pos.getY();
		double lower_forearm_z = lower_forearm_pos.getZ();


		/* compute avg position of wrist, lower_forearm and upper forearm */
		double x_avg = (upper_forearm_x + lower_forearm_x + current_pose_.position.x)/3;
		double y_avg = (upper_forearm_y + lower_forearm_y + current_pose_.position.y)/3;
		double z_avg = (upper_forearm_z + lower_forearm_z + current_pose_.position.z)/3;

		tf::Vector3 obj_endpoint = transform * tf::Vector3(x_avg, y_avg, z_avg);

		double end_x = obj_endpoint.getX();
		double end_y = obj_endpoint.getY();
		double end_z = obj_endpoint.getZ();

		printf("x:%f, y:%f, z:%f\n", end_x, end_y, end_z);

		double length = sqrt(pow(end_x,2) + pow(end_y,2) + pow(end_z,2));

		norm_vect.x = end_x / length * radius_;
		norm_vect.y = end_y / length * radius_;
		norm_vect.z = 0;
		
		printf("norm_vect: x:%f, y:%f, z:%f\n", norm_vect.x, norm_vect.y, norm_vect.z);

		break;
	}
}


void Grasp::transformToBase()
{
	tf::TransformListener listener;
	tf::StampedTransform transform;

	while(ros::ok()) {
		ros::Time t = ros::Time(0);

		try {
			listener.waitForTransform("/base", bottle_, t, ros::Duration(1.0));
			listener.lookupTransform("/base", bottle_, t, transform);
		}
		catch(tf::TransformException ex) {
			ROS_ERROR("%s", ex.what());
			tf_status_ = false;
		}

		tf::Transform pose = transform * grasping_pose_tf_;

		grasping_pose_.orientation.x = pose.getRotation().x();
		grasping_pose_.orientation.y = pose.getRotation().y();
		grasping_pose_.orientation.z = pose.getRotation().z();
		grasping_pose_.orientation.w = pose.getRotation().w();

		grasping_pose_.position.x = pose.getOrigin().getX();
		grasping_pose_.position.y = pose.getOrigin().getY();
		grasping_pose_.position.z = pose.getOrigin().getZ();

		break;
	}
}


void Grasp::getRightPose(const baxter_core_msgs::EndpointState msg)
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


void Grasp::computeGraspingPose()
{
	sub_trigger_ = true;
	robot_state_sub_ = nh_.subscribe("/robot/limb/right/endpoint_state", 1, &Grasp::getRightPose, this);

	while(1) {
		if(sub_trigger_ == true) ros::spinOnce();
		else {
			robot_state_sub_.shutdown();
			break;
		}
	}


	Vect norm_vect;
	normalizeVect(norm_vect);

	double yaw;

	if(norm_vect.x > 0 && norm_vect.y > 0) {
		printf("1\n");
		yaw = PI + atan(norm_vect.y/norm_vect.x);
	}
	else if(norm_vect.x < 0 && norm_vect.y > 0) {
		printf("2\n");
		yaw = 2*PI + atan(norm_vect.y/norm_vect.x);
	}
	else if(norm_vect.x < 0 && norm_vect.y < 0) {
		printf("3\n");
		yaw = atan(norm_vect.y/norm_vect.x);
	}
	else if(norm_vect.x > 0 && norm_vect.y < 0) {
		printf("4\n");
		yaw = PI + atan(norm_vect.y/norm_vect.x);
	}
	else if(norm_vect.y == 0 && norm_vect.x > 0) {
		yaw = 0;
	}
	else if(norm_vect.y == 0 && norm_vect.x < 0) {
		yaw = PI;
	}
	else if(norm_vect.x == 0 && norm_vect.y > 0) {
		yaw = PI / 2;
	}
	else if(norm_vect.x == 0 && norm_vect.y < 0) {
		yaw = 1.5 * PI;
	}

	printf("yaw: %f\n", yaw);

	tf::Quaternion grasping_yaw(tf::Vector3(0, 0, 1), yaw);
	tf::Vector3 grasping_pos(norm_vect.x, norm_vect.y, 0);
	tf::Transform grasping_tf(grasping_yaw, grasping_pos);

	tf::Quaternion y_rotation(tf::Vector3(0, 1, 0), PI/2);
	tf::Transform y_rotation_tf(y_rotation, tf::Vector3(0, 0, 0));

	grasping_pose_tf_ =  grasping_tf * y_rotation_tf;
	grasped_obj_tf_ = y_rotation_tf.inverse() * grasping_tf.inverse();

	printf("g_tf, qx:%f, qy:%f, qz:%f, w:%f, x:%f, y:%f, z:%f\n", grasping_pose_tf_.getRotation().x(), grasping_pose_tf_.getRotation().y(), grasping_pose_tf_.getRotation().z(), grasping_pose_tf_.getRotation().w(),\
			grasping_pose_tf_.getOrigin().getX(), grasping_pose_tf_.getOrigin().getY(), grasping_pose_tf_.getOrigin().getZ());
}


bool Grasp::graspingPose(huroco_grasping::graspPose::Request &req,
						 huroco_grasping::graspPose::Response &res)
{
	radius_ = req.radius;
	bottle_ = req.bottle;
	tf_status_ = true;

	computeGraspingPose();

	transformToBase();

	/* Only for open IROS2017 */
	/*************************************************/
	if(tf_status_ == true) {
		ros::ServiceClient right_arm_client = nh_.serviceClient<huroco_right_arm::rightCartesian>("/huroco_right_arm/right_cartesian");
		ros::ServiceClient left_arm_client = nh_.serviceClient<huroco_left_arm::leftCartesian>("/huroco_left_arm/left_cartesian");

		huroco_right_arm::rightCartesian right_srv;
		huroco_left_arm::leftCartesian left_srv;

		std::vector<geometry_msgs::Pose> waypoints;
		geometry_msgs::Pose point;

		point.orientation.x = grasping_pose_.orientation.x;
		point.orientation.y = grasping_pose_.orientation.y;
		point.orientation.z = grasping_pose_.orientation.z;
		point.orientation.w = grasping_pose_.orientation.w;
		point.position.x = grasping_pose_.position.x;
		point.position.y = grasping_pose_.position.y;
		point.position.z = grasping_pose_.position.z;

		waypoints.push_back(point);

		right_srv.request.waypoints = waypoints;

		if(right_arm_client.call(right_srv)) {
			ROS_INFO("status: %d", right_srv.response.status);

		}	
	}


	/*************************************************/


	res.grasping_pose.orientation.x = grasping_pose_.orientation.x;
	res.grasping_pose.orientation.y = grasping_pose_.orientation.y;
	res.grasping_pose.orientation.z = grasping_pose_.orientation.z;
	res.grasping_pose.orientation.w = grasping_pose_.orientation.w;

	res.grasping_pose.position.x = grasping_pose_.position.x;
	res.grasping_pose.position.y = grasping_pose_.position.y;
	res.grasping_pose.position.z = grasping_pose_.position.z;
	res.status = tf_status_;

	printf("%d", tf_status_);

	return true;
}


void Grasp::computeCapPose()
{
	robot_state_sub_ = nh_.subscribe("/robot/limb/right/endpoint_state", 1, &Grasp::getRightPose, this);

	while(1) {
		if(sub_trigger_ == true) ros::spinOnce();
		else {
			robot_state_sub_.shutdown();
			break;
		}
	}

	std::vector<double> para;
	nh_.getParam(name + "/" + bottle_, para);

	for(int i = 0; i < para.size(); i++){
		std::cout << i << ": " << para[i] << std::endl;
	}

	printf("h:%f, x:%f, y:%f, z:%f.\n", para[0], para[1], para[2], para[3]);

	tf::Quaternion lid_q(0, 0, 0, 1);
	tf::Vector3 lid_pos(para[1] - para[0], para[2], para[3]);
	tf::Transform lid_tf(lid_q, lid_pos);

	tf::TransformListener listener;
	tf::StampedTransform transform;
	tf::Transform lid_position;
	tf::Transform lid_orientation;

	while(ros::ok()) {
		ros::Time t = ros::Time(0);

		try {
			listener.waitForTransform("/base", "right_gripper", t, ros::Duration(1.0));
			listener.lookupTransform("/base", "right_gripper", t, transform);
		}
		catch(tf::TransformException ex) {
			ROS_ERROR("%s", ex.what());
			tf_status_ = false;
		}

		lid_position = transform * lid_tf;

		//printf("x:%f, y:%f, z:%f\n", pose.getOrigin().getX(), pose.getOrigin().getY(), pose.getOrigin().getZ());

		break;
	}

	tf::Quaternion x_rotation_q(tf::Vector3(1, 0, 0), PI/2);
	tf::Vector3 x_pos(0, 0, 0);
	tf::Transform x_tf(x_rotation_q, x_pos);

	tf::Quaternion y_rotation_q(tf::Vector3(0, 0, 1), PI/2);
	tf::Vector3 y_pos(0, 0, 0);
	tf::Transform y_tf(y_rotation_q, y_pos);

	tf_status_ = true;

	while(ros::ok()) {
		ros::Time t = ros::Time(0);

		try {
			listener.waitForTransform("/base", "/right_gripper", t, ros::Duration(1.0));
			listener.lookupTransform("/base", "/right_gripper", t, transform);
		}
		catch(tf::TransformException ex) {
			ROS_ERROR("%s", ex.what());
			tf_status_ = false;
		}

		lid_orientation = transform * y_tf * x_tf;
		//lid_orientation.inverse();

		lid_pose_.orientation.x = lid_orientation.getRotation().x();
		lid_pose_.orientation.y = lid_orientation.getRotation().y();
		lid_pose_.orientation.z = lid_orientation.getRotation().z();
		lid_pose_.orientation.w = lid_orientation.getRotation().w();

		lid_pose_.position.x = lid_position.getOrigin().getX();
		lid_pose_.position.y = lid_position.getOrigin().getY();
		lid_pose_.position.z = lid_position.getOrigin().getZ();

		break;
	}

}


bool Grasp::graspingCap(huroco_grasping::graspCap::Request &req,
						huroco_grasping::graspCap::Response &res)
{
	bottle_ = req.bottle;

	tf_status_ = true;
	sub_trigger_ = true;

	computeCapPose();

	res.lid_pose.orientation.x = lid_pose_.orientation.x;
	res.lid_pose.orientation.y = lid_pose_.orientation.y;
	res.lid_pose.orientation.z = lid_pose_.orientation.z;
	res.lid_pose.orientation.w = lid_pose_.orientation.w;

	res.lid_pose.position.x = lid_pose_.position.x;
	res.lid_pose.position.y = lid_pose_.position.y;
	res.lid_pose.position.z = lid_pose_.position.z;

	res.status = tf_status_;

	ros::ServiceClient left_client = nh_.serviceClient<huroco_left_arm::leftCartesian>("/huroco_left_arm/left_cartesian");
	huroco_left_arm::leftCartesian left_srv;

	geometry_msgs::Pose point;
	point.orientation.x = lid_pose_.orientation.x;
	point.orientation.y = lid_pose_.orientation.y;
	point.orientation.z = lid_pose_.orientation.z;
	point.orientation.w = lid_pose_.orientation.w;

	point.position.x = lid_pose_.position.x;
	point.position.y = lid_pose_.position.y;
	point.position.z = lid_pose_.position.z + 0.07;

	left_srv.request.waypoints.push_back(point);
	left_srv.request.waypoints.push_back(res.lid_pose);

	if(left_client.call(left_srv)) {

	}

	return true;
}


bool Grasp::graspingDoor(huroco_grasping::graspDoor::Request &req,
						 huroco_grasping::graspDoor::Response &res)
{
	door_name_ = req.door;
	tf_status_ = true;

	double yaw = PI/4 * 3;

	tf::Quaternion trans_q(0, 0, 0, 1);
	tf::Vector3 trans_pos(0.032, -0.262, 0.20);
	tf::Transform trans_tf(trans_q, trans_pos);

	tf::Quaternion yaw_q(tf::Vector3(0, 0, 1), yaw);
	tf::Vector3 yaw_pos(0, 0, 0);
	tf::Transform yaw_tf(yaw_q, yaw_pos);

	tf::Quaternion y_rotation(tf::Vector3(0, 1, 0), PI);
	tf::Transform y_rotation_tf(y_rotation, tf::Vector3(0 ,0, 0));

	grasping_pose_tf_ = trans_tf * yaw_tf * y_rotation_tf;

	huroco_left_arm::leftCartesian left_cartesian_srv;

	tf::TransformListener listener;
	tf::StampedTransform transform;

	while(ros::ok()) {
		ros::Time t = ros::Time(0);

		try {
			listener.waitForTransform("/base", door_name_, t, ros::Duration(1.0));
			listener.lookupTransform("/base", door_name_, t, transform);
		}
		catch(tf::TransformException ex) {
			ROS_ERROR("%s", ex.what());
			tf_status_ = false;
		}

		tf::Transform pose = transform * grasping_pose_tf_;

		grasping_pose_.orientation.x = pose.getRotation().x();
		grasping_pose_.orientation.y = pose.getRotation().y();
		grasping_pose_.orientation.z = pose.getRotation().z();
		grasping_pose_.orientation.w = pose.getRotation().w();

		grasping_pose_.position.x = pose.getOrigin().getX();
		grasping_pose_.position.y = pose.getOrigin().getY();
		grasping_pose_.position.z = pose.getOrigin().getZ();

		break;
	}

	res.door_handle_pose.orientation.x = grasping_pose_.orientation.x;
	res.door_handle_pose.orientation.y = grasping_pose_.orientation.y;
	res.door_handle_pose.orientation.z = grasping_pose_.orientation.z;
	res.door_handle_pose.orientation.w = grasping_pose_.orientation.w;

	res.door_handle_pose.position.x = grasping_pose_.position.x;
	res.door_handle_pose.position.y = grasping_pose_.position.y;
	res.door_handle_pose.position.z = grasping_pose_.position.z;

	res.status = tf_status_;
} 


int main(int argc, char **argv)
{
	ros::init(argc, argv, "huroco_grasping");

	Grasp grasping_planner;

	ROS_INFO("waiting for grasping information");
	ros::waitForShutdown();

	return 0;
}