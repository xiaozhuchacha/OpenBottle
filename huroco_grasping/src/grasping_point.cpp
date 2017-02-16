
#include "huroco_grasping/grasping_point.h"

#define PI     3.1415926
#define MAX    100 


const std::string name = "/huroco_grasping";


Grasp::Grasp()
		:spinner_(1)
{
	grasping_ = nh_.advertiseService(name+"/grasping_pose", &Grasp::graspingPose, this);

	spinner_.start();
}


Grasp::~Grasp()
{

}


void Grasp::normalizeVect(Vect &norm_vect)
{
	tf::Vector3 base_endpoint(current_pose_.position.x, current_pose_.position.y, current_pose_.position.z);

	tf::TransformListener listener;
	tf::StampedTransform transform;
	tf::StampedTransform lower_forearm_transform;
	tf::StampedTransform upper_forearm_transform;

	while(ros::ok()) {
		ros::Time t = ros::Time(0);

		try {
			listener.waitForTransform("/bottle64", "/base", t, ros::Duration(1.0));
			listener.lookupTransform("/bottle64", "/base", t, transform);
		}
		catch(tf::TransformException ex) {
			ROS_ERROR("%s", ex.what());
		}

		try {
			listener.waitForTransform("/right_lower_forearm", "/base", t, ros::Duration(1.0));
			listener.lookupTransform("/right_lower_forearm", "/base", t, lower_forearm_transform);
		}
		catch(tf::TransformException ex) {
			ROS_ERROR("%s", ex.what());
		}

		try {
			listener.waitForTransform("/right_upper_forearm", "/base", t, ros::Duration(1.0));
			listener.lookupTransform("/right_upper_forearm", "/base", t, upper_forearm_transform);
		}
		catch(tf::TransformException ex) {
			ROS_ERROR("%s", ex.what());
		}


		tf::Vector3 lower_forearm_pos = lower_forearm_transform.inverse() * tf::Vector3(0, 0, 0);
		tf::Vector3 upper_forearm_pos = upper_forearm_transform.inverse() * tf::Vector3(0, 0, 0);


		double upper_forearm_x = upper_forearm_pos.getX();
		double upper_forearm_y = upper_forearm_pos.getY();
		double upper_forearm_z = upper_forearm_pos.getZ();

		double lower_forearm_x = lower_forearm_pos.getX();
		double lower_forearm_y = lower_forearm_pos.getY();
		double lower_forearm_z = lower_forearm_pos.getZ();

		//printf("forearm: x:%f, y:%f, z:%f\n", forearm_x, forearm_y, forearm_z);

		tf::Vector3 obj_endpoint = transform * base_endpoint;

		double end_x = obj_endpoint.getX();
		double end_y = obj_endpoint.getY();
		double end_z = obj_endpoint.getZ();

		double x = (upper_forearm_x + lower_forearm_x + end_x)/2;
		double y = (upper_forearm_y + lower_forearm_y + end_y)/2;
		double z = (upper_forearm_z + lower_forearm_z + end_z)/2;

		double length = sqrt(pow(x,2) + pow(y,2) + pow(z, 2));

		norm_vect.x = x / length * radius_;
		norm_vect.y = y / length * radius_;
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
			listener.waitForTransform("/base", "/bottle64", t, ros::Duration(1.0));
			listener.lookupTransform("/base", "/bottle64", t, transform);
		}
		catch(tf::TransformException ex) {
			ROS_ERROR("%s", ex.what());
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

/*
void Grasp::generateGraspingPose()
{
	double radian = 2 * PI / dense_;

	double yaw = 0;
	double pitch = 0;
	double roll = 0;

	double z = 0;

	int i;
	for(i = 0; i < dense_; i++) {
		yaw += radian;
		//printf("yaw:%f\n", yaw);
		tf::Quaternion grasping_ori(tf::Vector3(0, 0, 1), yaw);

		tf::Vector3 grasping_trans(radius_ * cos(yaw + PI), radius_ * sin(yaw + PI), z);
		tf::Transform grasp_tf(grasping_ori, grasping_trans);

		
		tf::Quaternion obj_ori(0, 0, 0, 1);
		tf::Vector3 obj_pos(0, 0, 0);
		tf::Transform obj_tf(obj_ori, obj_pos);
		
		tf::Transform grasp_candidate = grasp_tf * obj_tf;

		geometry_msgs::Pose grasping_pose;

		grasping_pose.orientation.x = grasp_candidate.getRotation().x();
		grasping_pose.orientation.y = grasp_candidate.getRotation().y();
		grasping_pose.orientation.z = grasp_candidate.getRotation().z();
		grasping_pose.orientation.w = grasp_candidate.getRotation().w();

		grasping_pose.position.x = grasp_candidate.getOrigin().getX();
		grasping_pose.position.y = grasp_candidate.getOrigin().getY();
		grasping_pose.position.z = grasp_candidate.getOrigin().getZ();

		printf("qx:%f qy:%f qz:%f qw:%f x:%f y:%f z:%f\n", grasp_candidate.getRotation().x(), grasp_candidate.getRotation().y(), grasp_candidate.getRotation().z(), grasp_candidate.getRotation().w(),\
				grasp_candidate.getOrigin().getX(), grasp_candidate.getOrigin().getY(), grasp_candidate.getOrigin().getZ());

		grasping_poses_.push_back(grasping_pose);
	} 

}
*/


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
		yaw = PI + atan(norm_vect.y/norm_vect.x);
	}
	else if(norm_vect.x < 0 && norm_vect.y < 0) {
		printf("3\n");
		yaw = PI + atan(norm_vect.y/norm_vect.x);
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

	printf("yaw: %f", yaw);

	tf::Quaternion grasping_yaw(tf::Vector3(0, 0, 1), yaw);
	tf::Vector3 grasping_pos(norm_vect.x, norm_vect.y, 0);
	tf::Transform grasping_tf(grasping_yaw, grasping_pos);

	tf::Quaternion y_rotation(tf::Vector3(0, 1, 0), PI/2);
	tf::Transform y_rotation_tf(y_rotation, tf::Vector3(0, 0, 0));

	grasping_pose_tf_ =  grasping_tf * y_rotation_tf;

	printf("g_tf, qx:%f, qy:%f, qz:%f, w:%f, x:%f, y:%f, z:%f\n", grasping_pose_tf_.getRotation().x(), grasping_pose_tf_.getRotation().y(), grasping_pose_tf_.getRotation().z(), grasping_pose_tf_.getRotation().w(),\
			grasping_pose_tf_.getOrigin().getX(), grasping_pose_tf_.getOrigin().getY(), grasping_pose_tf_.getOrigin().getZ());
}


bool Grasp::graspingPose(huroco_grasping::graspPose::Request &req,
						 huroco_grasping::graspPose::Response &res)
{
	/*
	tf::Quaternion obj_q(req.obj_pose.orientation.x, req.obj_pose.orientation.y, req.obj_pose.orientation.z, req.obj_pose.orientation.w);
	tf::Vector3 obj_pos(req.obj_pose.position.x, req.obj_pose.position.y, req.obj_pose.position.z);

	obj_tf_.setRotation(obj_q);
	obj_tf_.setOrigin(obj_pos);
	*/
	radius_ = req.radius;

	//generateGraspingPose();

	computeGraspingPose();

	transformToBase();

	res.grasping_pose.orientation.x = grasping_pose_.orientation.x;
	res.grasping_pose.orientation.y = grasping_pose_.orientation.y;
	res.grasping_pose.orientation.z = grasping_pose_.orientation.z;
	res.grasping_pose.orientation.w = grasping_pose_.orientation.w;

	res.grasping_pose.position.x = grasping_pose_.position.x;
	res.grasping_pose.position.y = grasping_pose_.position.y;
	res.grasping_pose.position.z = grasping_pose_.position.z;

	return true;
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "huroco_grasping");

	Grasp grasping_planner;

	ROS_INFO("waiting for grasping information");
	ros::waitForShutdown();

	return 0;
}