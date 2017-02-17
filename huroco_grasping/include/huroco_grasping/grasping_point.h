#ifndef GRASPING_POINT
#define GRASPING_POINT


#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <string>
#include <cmath>


#include "huroco_grasping/graspPose.h"
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

	ros::Subscriber robot_state_sub_;

	geometry_msgs::Pose obj_pose_;
	geometry_msgs::Pose current_pose_;
	geometry_msgs::Pose grasping_pose_;

	tf::Transform grasping_pose_tf_;
	tf::Transform obj_tf_;

	std::vector<geometry_msgs::Pose> grasping_poses_;

	double radius_;
	double dense_;

	void normalizeVect(Vect &norm_vect);

	void transformToBase();

	void getRightPose(const baxter_core_msgs::EndpointState msg);

	//void generateGraspingPose();

	void computeGraspingPose();

	bool graspingPose(huroco_grasping::graspPose::Request &req,
					  huroco_grasping::graspPose::Response &res);

	bool sub_trigger_;
};


#endif

