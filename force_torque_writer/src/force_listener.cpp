#include <stdio.h>
#include <mutex>
#include <string>

#include <baxter_core_msgs/EndpointState.h>

#include "ros/ros.h"
#include "std_msgs/String.h"

#include "force_torque_writer/SModel_robot_input.h"

int gripper_position;
std::mutex mutex;
bool force_called, gripper_called, status_called;
std::string status;

void forceCallback_left(const baxter_core_msgs::EndpointState msg) {
  // ROS_INFO("I heard: [%s]", msg->data.c_str());
  double out_x = msg.wrench.force.x;
  double out_y = msg.wrench.force.y;
  double out_z = msg.wrench.force.z;

  double out_x_t = msg.wrench.torque.x;
  double out_y_t = msg.wrench.torque.y;
  double out_z_t = msg.wrench.torque.z;

  int sec = msg.header.stamp.sec;
  int nsec = msg.header.stamp.nsec;

  force_called = true;

  // printf("force z:%f\n", out);
  if (force_called && gripper_called && status_called) {
    mutex.lock();
    ROS_INFO_STREAM("Wrote left: " << sec << " " << nsec);
    FILE *fp = fopen("forcetorque_left.csv", "a");
    fprintf(fp, "%i,%i,%f,%f,%f,%f,%f,%f,%i,%s\n", sec, nsec, out_x, out_y,
            out_z, out_x_t, out_y_t, out_z_t, gripper_position, status.c_str());
    fclose(fp);
    mutex.unlock();
  }
}

void forceCallback_right(const baxter_core_msgs::EndpointState msg) {
  // ROS_INFO("I heard: [%s]", msg->data.c_str());
  double out_x = msg.wrench.force.x;
  double out_y = msg.wrench.force.y;
  double out_z = msg.wrench.force.z;

  double out_x_t = msg.wrench.torque.x;
  double out_y_t = msg.wrench.torque.y;
  double out_z_t = msg.wrench.torque.z;

  int sec = msg.header.stamp.sec;
  int nsec = msg.header.stamp.nsec;

  // printf("force z:%f\n", out);
  if (force_called && gripper_called && status_called) {
    mutex.lock();
    ROS_INFO_STREAM("Wrote right: " << sec << " " << nsec);
    FILE *fp = fopen("forcetorque_right.csv", "a");
    fprintf(fp, "%i,%i,%f,%f,%f,%f,%f,%f,%i,%s\n", sec, nsec, out_x, out_y,
            out_z, out_x_t, out_y_t, out_z_t, gripper_position, status.c_str());
    fclose(fp);
    mutex.unlock();
  }
}

void gripperCallback_left(const force_torque_writer::SModel_robot_input &msg) {
  gripper_called = true;
  mutex.lock();
  gripper_position = msg.gPOA;
  mutex.unlock();
}

void execStatusCallback(const std_msgs::String::ConstPtr &msg) {
  status_called = true;
  mutex.lock();
  status = msg->data;
  mutex.unlock();
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "force_listener");

  /**
   * NodeHandle is the main access point to communications with the ROS system.
   * The first NodeHandle constructed will fully initialize this node, and the
   * last
   * NodeHandle destructed will close down the node.
   */
  ros::NodeHandle n;
  ros::NodeHandle n_r;
  /**
   * The subscribe() call is how you tell ROS that you want to receive messages
   * on a given topic.  This invokes a call to the ROS
   * master node, which keeps a registry of who is publishing and who
   * is subscribing.  Messages are passed to a callback function, here
   * called chatterCallback.  subscribe() returns a Subscriber object that you
   * must hold on to until you want to unsubscribe.  When all copies of the
   * Subscriber
   * object go out of scope, this callback will automatically be unsubscribed
   * from
   * this topic.
   *
   * The second parameter to the subscribe() function is the size of the message
   * queue.  If messages are arriving faster than they are being processed, this
   * is the number of messages that will be buffered up before beginning to
   * throw
   * away the oldest ones.
   */
  ros::Subscriber sub_l =
      n.subscribe("/robot/limb/left/endpoint_state", 1000, forceCallback_left);
  // ros::Subscriber sub_r = n_r.subscribe("/robot/limb/right/endpoint_state",
  // 1000, forceCallback_right);
  ros::Subscriber sub_grip_l =
      n_r.subscribe("/SModelRobotInput", 1000, gripperCallback_left);
  ros::Subscriber sub_exec_status =
      n_r.subscribe("/execution_status", 1000, execStatusCallback);
  /**
   * ros::spin() will enter a loop, pumping callbacks.  With this version, all
   * callbacks will be called from within this thread (the main one).
   * ros::spin()
   * will exit when Ctrl-C is pressed, or the node is shutdown by the master.
   */

  force_called = gripper_called = status_called = false;
  ros::spin();

  return 0;
}
