#ifndef GRASPING_POINT
#define GRASPING_POINT


#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <string>
#include <cmath>

#include "huroco_right_arm/rightCartesian.h"
#include "huroco_left_arm/leftCartesian.h"

#include "huroco_grasping/graspPose.h"
#include "huroco_grasping/graspCap.h"
#include "huroco_grasping/graspDoor.h"

#include "baxter_core_msgs/EndpointState.h"


typedef struct {
	double x;
	double y;
	double z;
} Vect;


class Grasp
{
public:
	Grasp();
	~Grasp();


private:
	ros::NodeHandle nh_;
	ros::AsyncSpinner spinner_;

	ros::ServiceServer grasping_;
	ros::ServiceServer cap_;
	ros::ServiceServer door_;

	ros::Subscriber robot_state_sub_;

	geometry_msgs::Pose obj_pose_;
	geometry_msgs::Pose current_pose_;
	geometry_msgs::Pose grasping_pose_;
	geometry_msgs::Pose lid_pose_;

	tf::Transform grasping_pose_tf_;
	tf::Transform obj_tf_;
	tf::Transform grasped_obj_tf_;

	std::vector<geometry_msgs::Pose> grasping_poses_;

	std::string bottle_;
	std::string door_name_;

	double radius_;
	double dense_;

	void normalizeVect(Vect &norm_vect);

	void transformToBase();

	void getRightPose(const baxter_core_msgs::EndpointState msg);

	void computeGraspingPose();

	void computeCapPose();

	bool graspingPose(huroco_grasping::graspPose::Request &req,
					  huroco_grasping::graspPose::Response &res);

	bool graspingCap(huroco_grasping::graspCap::Request &req,
					 huroco_grasping::graspCap::Response &res);

	bool graspingDoor(huroco_grasping::graspDoor::Request &req,
					  huroco_grasping::graspDoor::Response &res);


	bool sub_trigger_;
	bool tf_status_;
};


#endif

