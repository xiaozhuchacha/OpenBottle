 #include <ros/ros.h>
#include <vector>
#include <iostream>

#include <tf/transform_broadcaster.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>

#include "huroco_right_arm/rightCartesian.h"
#include "huroco_right_arm/rightInit.h"
#include "huroco_left_arm/leftCartesian.h"
#include "huroco_left_arm/rotateLeftWrist.h"


#include "action_manager/setObject.h"
#include "action_manager/execAction.h"

//#include "huroco_grasping/graspPose.h"

#include "right_gripper_receiver/rightGrasp.h"
#include "right_gripper_receiver/rightStatus.h"


int main(int argc, char **argv)
{
	ros::init(argc, argv, "pipeline_test");
	ros::NodeHandle n;

	ros::ServiceClient action_client;
	ros::ServiceClient action_bottle;


	action_client = n.serviceClient<action_manager::execAction>("action_manager/exec");
	action_bottle = n.serviceClient<action_manager::setObject>("action_manager/bottle");
	/* simtrack grasp test */
/*
	ros::ServiceClient right_client = n.serviceClient<huroco_right_arm::rightCartesian>("/huroco_right_arm/right_cartesian");
	ros::ServiceClient left_client = n.serviceClient<huroco_left_arm::leftCartesian>("/huroco_left_arm/left_cartesian");
	huroco_right_arm::rightCartesian right_srv;
	huroco_left_arm::leftCartesian left_srv;
	
	ros::ServiceClient grasp_client = n.serviceClient<right_gripper_receiver::rightGrasp>("/huroco_right_gripper/command");
	ros::ServiceClient status_client = n.serviceClient<right_gripper_receiver::rightStatus>("/huroco_right_gripper/status");

	right_gripper_receiver::rightGrasp grasp_srv;
	right_gripper_receiver::rightStatus status_srv;


	std_msgs::String left_gripper_msg;
	ros::Publisher left_gripper_pub = n.advertise<std_msgs::String>("/left_gripper", 1);
	left_gripper_msg.data = "a";
	left_gripper_pub.publish(left_gripper_msg);

	ros::ServiceClient right_init = n.serviceClient<huroco_right_arm::rightInit>("/huroco_right_arm/right_init");
	huroco_right_arm::rightInit right_init_srv;

	//ros::ServiceClient grasping_client = n.serviceClient<huroco_grasping::graspPose>("/huroco_grasping/grasping_pose");
	//huroco_grasping::graspPose grasping_srv;
*/

	while(getchar()) {
		ROS_INFO("Task:\n1:bottle64, 2:bottle72, 3:bottle69 4: bottle68.");
		
		int i;
		std::string task_name;
		std::cin>>i;

		if(i == 1) task_name = "bottle64";
		else if(i == 2) task_name = "bottle72";
		else if(i == 3) task_name = "bottle69";
		else if(i == 4) task_name = "bottle68";
		else if(i == 5) task_name = "sprinkle";

		action_manager::setObject bottle_srv;
		bottle_srv.request.object = task_name;

		action_bottle.call(bottle_srv);


		action_manager::execAction action_srv;
		if(i == 1) {
			action_srv.request.action = "approach";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "push";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "twist";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "neutral";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "move";
			action_client.call(action_srv);
			sleep(1.5);
		
			action_srv.request.action = "push";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "twist";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "pull";
			action_client.call(action_srv);
			sleep(1.5);

			continue;
		}
		else if(i == 2) {
			action_srv.request.action = "approach";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "push";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "twist";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "neutral";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "move";
			action_client.call(action_srv);
			sleep(1.5);
		
			action_srv.request.action = "push";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "twist";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "pull";
			action_client.call(action_srv);
			sleep(1.5);

			continue;
		}
		else if(i == 3) {
			action_srv.request.action = "approach";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "grasp_right";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "twist";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "ungrasp_right";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "move";
			action_client.call(action_srv);
			sleep(1.5);
		
			action_srv.request.action = "grasp_right";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "twist";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "pull";
			action_client.call(action_srv);
			sleep(1.5);

			continue;
		}	

		else if(i == 4) {
			action_srv.request.action = "approach";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "pinch";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "twist";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "unpinch";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "move";
			action_client.call(action_srv);
			sleep(1.5);
		
			action_srv.request.action = "grasp_right";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "twist";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "pull";
			action_client.call(action_srv);
			sleep(1.5);

			continue;
		}
		else if(i == 5) {
			action_srv.request.action = "approach";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "pinch";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "twist";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "unpinch";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "move";
			action_client.call(action_srv);
			sleep(1.5);
		
			action_srv.request.action = "grasp_right";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "twist";
			action_client.call(action_srv);
			sleep(1.5);

			action_srv.request.action = "pull";
			action_client.call(action_srv);
			sleep(1.5);

			continue;
		}

	}
	ros::shutdown();

	return 0;
}
