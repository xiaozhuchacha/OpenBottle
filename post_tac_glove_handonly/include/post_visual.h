#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <math.h>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "fsr_glove/glove.h"

#define handTFNum 14
#define forceNum 26
#define csvBottleTFNum 17


class Glove{
public:
	Glove(std::string filename, bool save_csv);
	~Glove();

	void refine_CSV();
private:
	tf::Vector3 quaternion_rotate(tf::Quaternion q, tf::Vector3 u);
	visualization_msgs::Marker genmark(tf::Vector3 pt_marker, tf::Quaternion q_marker, float length, float radius, float chroma, std::string ns);

	void publish_state_finger(int idx_from, int idx_to);
	void publish_state_palm();
	void publish_state();		

	void refine_hand_tf();
	
	std::vector<std::string> link_names_;
	std::vector<float> link_lengths_;
	std::vector<int> parents_;
	std::vector<tf::Quaternion> canonical_pose_;
	std::vector<tf::Vector3> canonical_origin_;
 
	ros::NodeHandle nh_;
	std::ifstream ifs_csv_;
	std::ofstream ofs_csv_;

	tf::TransformBroadcaster br_;
	ros::Publisher marker_pub_;

	std::vector<std::string> world_frames_;
	std::vector<std::string> wrist_frames_;
	std::vector<std::string> glove_frames_;
	std::vector<std::string> palm_frames_;

	tf::Transform world_wrist_tf;
	tf::Transform wrist_glove_tf;
	tf::Transform glove_palm_tf;
	std::vector<tf::Quaternion> hand_qs_;
	std::vector<ros::Time> hand_times_;
	std::vector<float> forces_;
	std::map<std::string,int> name_idx_map;

	bool save_csv_;
	double frame_duration_;

	const float Pi = 3.145926536;
  	//const float PalmWidth = .09;
  	//const float PalmLength = .09;
	//const float PalmLength = .075;
  	//const float PalmHeight = .03;
  	//const float ProximalPhalangeLength = .03;
  	//const float ProximalPhalanxLength = .03;
  	//const float MiddlePhalangeLength = .025;
  	//const float DistalPhalangeLength = .025;
  	//const float DistalPhalanxLength = .03;
  	//const float radius = 0.008;
  	//const float arrayLength = 0.08;
	const float PalmWidth = 0.075;
	const float PalmLength = 0.075;
	const float PalmHeight = 0.020;
	const float ProximalPhalangeLength = 0.03;
	const float ProximalPhalanxLength = 0.03;
	const float MiddlePhalangeLength = 0.025;
	const float DistalPhalangeLength = 0.021;
	const float DistalPhalanxLength = 0.028;
	const float radius = 0.006;
	const float arrayLength = 0.07;
};
