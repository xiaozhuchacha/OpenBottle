#include "post_visual.h"

Glove::Glove(std::string filename, bool save_csv)
{
	ifs_csv_.open(filename.c_str());
	save_csv_=save_csv;

	if(save_csv_)
		ofs_csv_.open((filename.substr(0,filename.length()-4)+"_correct.csv").c_str());

	//frame_duration_=frame_duration;

	world_frames_.resize(2);
	world_frames_[0]="world";world_frames_[1]="/world";
	
	wrist_frames_.resize(2);
	wrist_frames_[0]="vicon/wrist/wrist";wrist_frames_[1]="/vicon/wrist/wrist";

	glove_frames_.resize(2);
	glove_frames_[0]="glove_link";glove_frames_[1]="/glove_link";

	palm_frames_.resize(2);
	palm_frames_[0]="palm_link";palm_frames_[1]="/palm_link";

	link_names_.resize(15);
	link_names_[0] = "palm_link";
        link_names_[1] = "proximal_phalanx_link_1";
        link_names_[2] = "distal_phalanx_link_1";
        link_names_[3] = "proximal_phalange_link_1";
        link_names_[4] = "middle_phalange_link_1";
        link_names_[5] = "distal_phalange_link_1";
        link_names_[6] = "proximal_phalange_link_2";
        link_names_[7] = "middle_phalange_link_2";
        link_names_[8] = "distal_phalange_link_2";
        link_names_[9] = "proximal_phalange_link_3";
        link_names_[10] = "middle_phalange_link_3";
        link_names_[11] = "distal_phalange_link_3";
        link_names_[12] = "proximal_phalange_link_4";
        link_names_[13] = "middle_phalange_link_4";
        link_names_[14] = "distal_phalange_link_4";

	for(int i=0;i<handTFNum+1;i++)
	{
		name_idx_map[link_names_[i]]=i;
	}

	//default values, will be changed
	link_lengths_.resize(15);
        link_lengths_[0] = -1;
        link_lengths_[1] = ProximalPhalanxLength;
        link_lengths_[2] = DistalPhalanxLength;
        link_lengths_[3] = ProximalPhalangeLength;
        link_lengths_[4] = MiddlePhalangeLength;
        link_lengths_[5] = DistalPhalangeLength;
        link_lengths_[6] = ProximalPhalangeLength;
        link_lengths_[7] = MiddlePhalangeLength;
        link_lengths_[8] = DistalPhalangeLength;
        link_lengths_[9] = ProximalPhalangeLength;
        link_lengths_[10] = MiddlePhalangeLength;
        link_lengths_[11] = DistalPhalangeLength;
        link_lengths_[12] = ProximalPhalangeLength;
        link_lengths_[13] = MiddlePhalangeLength;
        link_lengths_[14] = DistalPhalangeLength;	

	parents_.resize(15);
        parents_[0] = -1;
        parents_[1] = 0;
        parents_[2] = 1;
        parents_[3] = 0;
        parents_[4] = 3;
        parents_[5] = 4;
        parents_[6] = 0;
        parents_[7] = 6;
        parents_[8] = 7;
        parents_[9] = 0;
        parents_[10] = 9;
        parents_[11] = 10;
        parents_[12] = 0;
        parents_[13] = 12;
        parents_[14] = 13;

	forces_.resize(forceNum);

	hand_times_.resize(15);

        hand_qs_.resize(15);
	for(int i=0;i<hand_qs_.size();i++)
	{
		hand_qs_[i]=tf::Quaternion(0.0, 0.0, 0.0, 1.0);
	}

	tf::Quaternion q_ident(0.0, 0.0, 0.0, 1.0);
	tf::Quaternion q_thumb, q_1, q_2, q_3, q_4;
	q_thumb=tf::Quaternion(tf::Vector3(0.0, 0.0, 1.0), 0.2*Pi);
	q_1=tf::Quaternion(tf::Vector3(0.0, 0.0, 1.0), 0.05*Pi);
	q_2=tf::Quaternion(tf::Vector3(0.0, 0.0, 1.0), 0.0);
	q_3=tf::Quaternion(tf::Vector3(0.0, 0.0, 1.0), -0.05*Pi);
	q_4=tf::Quaternion(tf::Vector3(0.0, 0.0, 1.0), -0.1*Pi);

	canonical_pose_.resize(15);
	canonical_origin_.resize(15);
	for(int i=0;i<15;i++)
	{
		canonical_pose_[i]=q_ident;
		canonical_origin_[i]=tf::Vector3(0.0, 0.0, 0.0);
	}
	canonical_origin_[1] = tf::Vector3(-.0*PalmLength, 2*(PalmWidth/4), -0.3*PalmHeight);
        canonical_origin_[2] = tf::Vector3(ProximalPhalanxLength, 0, 0);
        canonical_origin_[3] = tf::Vector3(PalmLength/2, 1.5*(PalmWidth/4), 0);
        canonical_origin_[4] = tf::Vector3(ProximalPhalangeLength, 0, 0);
        canonical_origin_[5] = tf::Vector3(MiddlePhalangeLength, 0, 0);
        canonical_origin_[6] = tf::Vector3(PalmLength/2, 0.5*(PalmWidth/4), 0);
        canonical_origin_[7] = tf::Vector3(ProximalPhalangeLength, 0, 0);
        canonical_origin_[8] = tf::Vector3(MiddlePhalangeLength, 0, 0);
        canonical_origin_[9] = tf::Vector3(PalmLength/2, -0.5*(PalmWidth/4), 0);
        canonical_origin_[10] = tf::Vector3(ProximalPhalangeLength, 0, 0);
        canonical_origin_[11] = tf::Vector3(MiddlePhalangeLength, 0, 0);
        canonical_origin_[12] = tf::Vector3(PalmLength/2, -1.5*(PalmWidth/4), 0);
        canonical_origin_[13] = tf::Vector3(ProximalPhalangeLength, 0, 0);
        canonical_origin_[14] = tf::Vector3(MiddlePhalangeLength, 0, 0);
	canonical_pose_[1] = q_thumb;
        canonical_pose_[2] = q_thumb;
        canonical_pose_[3] = q_1;
        canonical_pose_[4] = q_1;
        canonical_pose_[5] = q_1;
        canonical_pose_[6] = q_2;
        canonical_pose_[7] = q_2;
        canonical_pose_[8] = q_2;
        canonical_pose_[9] = q_3;
        canonical_pose_[10] = q_3;
        canonical_pose_[11] = q_3;
        canonical_pose_[12] = q_4;
        canonical_pose_[13] = q_4;
        canonical_pose_[14] = q_4;

	marker_pub_ = nh_.advertise<visualization_msgs::Marker>("force_marker",1000);

}

Glove::~Glove()
{
	ifs_csv_.close();
	if(save_csv_)
		ofs_csv_.close();
}

tf::Vector3 Glove::quaternion_rotate(tf::Quaternion q, tf::Vector3 u)
{
	tf::Quaternion q_u(u.getX(),u.getY(),u.getZ(),0.0);
	tf::Quaternion q_v=q*(q_u*q.inverse());
	tf::Vector3 v(q_v.x(),q_v.y(),q_v.z());

	return v;
}

visualization_msgs::Marker Glove::genmark(tf::Vector3 pt_marker, tf::Quaternion q_marker, float length, float radius, float chroma, std::string ns)
{
	visualization_msgs::Marker cylinder;
  	cylinder.header.frame_id = palm_frames_[0].c_str();
  	//cylinder.header.stamp = ros::Time::now();
	cylinder.header.stamp = ros::Time(0);
  	cylinder.ns = ns.c_str();
  	cylinder.type = visualization_msgs::Marker::CYLINDER;

  	cylinder.pose.position.x = pt_marker.getX();
  	cylinder.pose.position.y = pt_marker.getY();
  	cylinder.pose.position.z = pt_marker.getZ();

  	cylinder.pose.orientation.w = q_marker.w();
  	cylinder.pose.orientation.x = q_marker.x();
  	cylinder.pose.orientation.y = q_marker.y();
  	cylinder.pose.orientation.z = q_marker.z();

  	cylinder.scale.x = radius * 2;
  	cylinder.scale.y = radius * 2;
  	cylinder.scale.z = length;

	float maxF=400.0f,minF=20.0f;
  	if (chroma > maxF) {
       		chroma = 1.0f;
        	cylinder.color.r = chroma;
        	cylinder.color.g = 0.0f;
        	cylinder.color.b = 0.0f;
        	cylinder.color.a = 1.0f;
  	}
  	else if (chroma < minF) {
        	cylinder.color.r = 0.0f;
        	cylinder.color.g = 1.0f;
        	cylinder.color.b = 0.0f;
        	cylinder.color.a = 1.0f;
  	}
  	else{
        	chroma/=maxF;
        	cylinder.color.r = chroma;
        	cylinder.color.g = 1.0f-chroma;
        	cylinder.color.b = 0.0f;
        	cylinder.color.a = 1.0f;
  	}

	return cylinder;
}

void Glove::publish_state_finger(int idx_from, int idx_to)
{
	if(idx_to>handTFNum)
		return;
	
	tf::Transform hand_tf;
	tf::Vector3 t_link(0.0,0.0,0.0);
	for(int i=idx_from;i<=idx_to;i++)
	{
		tf::Quaternion r_link=hand_qs_[i];
		t_link+=quaternion_rotate(hand_qs_[parents_[i]],canonical_origin_[i]);

		hand_tf.setOrigin(t_link);
		hand_tf.setRotation(r_link);

		tf::Quaternion q_marker=r_link*tf::Quaternion(tf::Vector3(0.0, 1.0, 0.0),0.5*Pi);
		tf::Vector3 pt_marker=t_link+quaternion_rotate(q_marker,tf::Vector3(0.0, 0.0, link_lengths_[i]/2));

		std::string ns="tac_glove_marker"+link_names_[i];
		float force=((i==idx_from)?forces_[idx_from*2/3]:forces_[idx_from*2/3+1]);

		visualization_msgs::Marker mk=genmark(pt_marker,q_marker,link_lengths_[i],radius,force,ns);
		br_.sendTransform(tf::StampedTransform(hand_tf, hand_times_[i], palm_frames_[0].c_str(), link_names_[i]));
		marker_pub_.publish(mk);

		//save tfs
		if(save_csv_)
		{
			ofs_csv_<<palm_frames_[0].c_str()<<","<<link_names_[i]<<",";
			ofs_csv_<<hand_tf.getOrigin().getX()<<","<<hand_tf.getOrigin().getY()<<","<<hand_tf.getOrigin().getZ()<<",";
			ofs_csv_<<hand_tf.getRotation().x()<<","<<hand_tf.getRotation().y()<<","<<hand_tf.getRotation().z()<<","<<hand_tf.getRotation().w()<<",";
		}
	}
}

void Glove::publish_state_palm()
{
	tf::Vector3 t_link=glove_palm_tf.getOrigin();
	tf::Quaternion r_link=glove_palm_tf.getRotation();

	br_.sendTransform(tf::StampedTransform(glove_palm_tf, hand_times_[0], glove_frames_[0].c_str(), palm_frames_[0].c_str()));

	float X_center=arrayLength/2-arrayLength/8;
        float Y_center=-(arrayLength/2-arrayLength/8);
        int count=10;
	for(int i=0;i<4;i++)
	{
		for(int j=0;j<4;j++)
		{
			tf::Vector3 pt_marker=t_link+quaternion_rotate(r_link,tf::Vector3(X_center,Y_center,0.0));
			tf::Quaternion q_marker=r_link;
			std::stringstream ss;
			ss<<"tac_glove_marker"<<i<<"&"<<j;
			float force=forces_[count];
                        visualization_msgs::Marker mk=genmark(pt_marker,q_marker,PalmHeight,arrayLength/8,force,ss.str());

                        marker_pub_.publish(mk);

                        Y_center+=arrayLength/4;
                        count++;
		}
		X_center-=arrayLength/4;
                Y_center=-(arrayLength/2-arrayLength/8);
	}
}

void Glove::publish_state()
{
	publish_state_palm();

        publish_state_finger(1,2);

        publish_state_finger(3,5);

        publish_state_finger(6,8);

        publish_state_finger(9,11);

        publish_state_finger(12,14);

	//save forces
	if(save_csv_)
	{
		for(int i=0;i<forceNum;i++)
		{
			ofs_csv_<<forces_[i]<<",";
		}
		ofs_csv_<<"\n";
	}
}

void Glove::refine_CSV()
{
	std::string line;
	cv::namedWindow("test_window");

	ros::Time::init();
	while(std::getline(ifs_csv_,line))
	{
		char key=cv::waitKey(1);
		if(key==32)
		{
			ROS_INFO_STREAM("waiting for pressing any key...");
			cv::waitKey(0);
		}

		std::stringstream ss;
		std::istringstream iss(line);
		std::string seg;
	
		std::getline(iss,seg,',');      //timestamp sec
		int ns;
		ss.clear();ss.str("");ss<<seg;
		ss>>ns;
		if(save_csv_)
			ofs_csv_<<seg<<",";

                std::getline(iss,seg,',');      //timestamp nsec
		int nns;
		ss.clear();ss.str("");ss<<seg;
		ss>>nns;
		if(save_csv_)
			ofs_csv_<<seg<<",";

		
                std::getline(iss,seg,',');      //image id
		if(save_csv_)
			ofs_csv_<<seg<<",";

		for(int j=0;j<csvBottleTFNum;j++)	//tfs
		{
			//printf("print tf:\n");
                        std::getline(iss,seg,',');
                        //printf("frame id: %s\n",seg.c_str());
                        std::string frame_id=seg;

                        std::getline(iss,seg,',');
                        //printf("child frame id: %s\n",seg.c_str());
                        std::string child_frame_id=seg;

			std::getline(iss,seg,',');
                        float tx;
                        ss.clear();ss.str("");ss<<seg;
                        ss>>tx;
                        //printf("translation: x=%f, ",tx);
                        std::getline(iss,seg,',');
                        float ty;
                        ss.clear();ss.str("");ss<<seg;
                        ss>>ty;
                        //printf("y=%f, ",ty);
                        std::getline(iss,seg,',');
                        float tz;
                        ss.clear();ss.str("");ss<<seg;
                        ss>>tz;
                        //printf("z=%f\n",tz);
			std::getline(iss,seg,',');
                        float rw;
                        ss.clear();ss.str("");ss<<seg;
                        ss>>rw;
                        //printf("rotation: w=%f, ",rw);
                        std::getline(iss,seg,',');
                        float rx;
                        ss.clear();ss.str("");ss<<seg;
                        ss>>rx;
                        //printf("x=%f, ",rx);
                        std::getline(iss,seg,',');
                        float ry;
                        ss.clear();ss.str("");ss<<seg;
                        ss>>ry;
                        //printf("y=%f, ",ry);
                        std::getline(iss,seg,',');
                        float rz;
                        ss.clear();ss.str("");ss<<seg;
                        ss>>rz;
                        //printf("z=%f\n",rz);

			if((strcmp(frame_id.c_str(),world_frames_[0].c_str())==0||strcmp(frame_id.c_str(),world_frames_[1].c_str())==0) && (strcmp(child_frame_id.c_str(),wrist_frames_[0].c_str())==0||strcmp(child_frame_id.c_str(),wrist_frames_[1].c_str())==0))
                        {
                                world_wrist_tf.setOrigin(tf::Vector3(tx,ty,tz));
				tf::Quaternion q(rx,ry,rz,rw);
				q=q.normalized();
                                world_wrist_tf.setRotation(q);
				
				if(save_csv_)
				{
					ofs_csv_<<world_frames_[0]<<","<<wrist_frames_[0]<<",";
					ofs_csv_<<tx<<","<<ty<<","<<tz<<",";
					ofs_csv_<<rx<<","<<ry<<","<<rz<<","<<rw<<",";
				}
				br_.sendTransform(tf::StampedTransform(world_wrist_tf, ros::Time(ns,nns), world_frames_[0].c_str(), wrist_frames_[0].c_str()));
                        }
                        else if((strcmp(frame_id.c_str(),wrist_frames_[0].c_str())==0||strcmp(frame_id.c_str(),wrist_frames_[1].c_str())==0) && (strcmp(child_frame_id.c_str(),glove_frames_[0].c_str())==0||strcmp(child_frame_id.c_str(),glove_frames_[1].c_str())==0))
                        {
                                wrist_glove_tf.setOrigin(tf::Vector3(tx,ty,tz));
				tf::Quaternion q(rx,ry,rz,rw);
                                q=q.normalized();
                                wrist_glove_tf.setRotation(q);

				if(save_csv_)
				{
					ofs_csv_<<wrist_frames_[0]<<","<<glove_frames_[0]<<",";
                                	ofs_csv_<<tx<<","<<ty<<","<<tz<<",";
                                	ofs_csv_<<rx<<","<<ry<<","<<rz<<","<<rw<<",";
				}
				br_.sendTransform(tf::StampedTransform(wrist_glove_tf, ros::Time(ns,nns), wrist_frames_[0].c_str(), glove_frames_[0].c_str()));
                        }
                        else if((strcmp(frame_id.c_str(),glove_frames_[0].c_str())==0||strcmp(frame_id.c_str(),glove_frames_[1].c_str())==0) && (strcmp(child_frame_id.c_str(),palm_frames_[0].c_str())==0||strcmp(child_frame_id.c_str(),palm_frames_[1].c_str())==0))
                        {
                                glove_palm_tf.setOrigin(tf::Vector3(tx,ty,tz));
				tf::Quaternion q(rx,ry,rz,rw);
                                q=q.normalized();
                                glove_palm_tf.setRotation(q);

				if(save_csv_)
				{
					ofs_csv_<<glove_frames_[0]<<","<<palm_frames_[0]<<",";
                                	ofs_csv_<<tx<<","<<ty<<","<<tz<<",";
                                	ofs_csv_<<rx<<","<<ry<<","<<rz<<","<<rw<<",";
				}
				hand_times_[0]=ros::Time(ns,nns);
                        }
			else
                        {
                                int idx=name_idx_map[child_frame_id];
				tf::Quaternion q(rx,ry,rz,rw);
                                q=q.normalized();
				hand_qs_[idx]=q;
				hand_times_[idx]=ros::Time(ns,nns);
                        }
		}

		//printf("force:\n");                     //forces
                for(int j=0;j<forceNum;j++)
                {
                        std::getline(iss,seg,',');
                        float f;
                        ss.clear();ss.str("");ss<<seg;
                        ss>>f;
                        //printf("%f, ",f);
			if(f<0.0)
				forces_[j]=0.0;
			else
                        	forces_[j]=f;
                }
                //printf("\n");
		publish_state();
		//ros::Duration(frame_duration_).sleep();
	}
}

int main(int argc, char** argv)
{
	ros::init(argc,argv,"post_tac_glove_handonly");

	std::string filename="";
	filename.assign(argv[1]);	

	bool save_csv;
	if(strcmp(argv[2],"true")==0)
		save_csv=true;
	else
		save_csv=false;

	//double frame_duration;
	//std::stringstream ss;
	//std::string frame_duration_str;
	//frame_duration_str.assign(argv[3]);
	//ss<<frame_duration_str;
	//ss>>frame_duration;

	Glove g(filename,save_csv);
	g.refine_CSV();

	ROS_INFO_STREAM("visualization done");
	return 0;
}
