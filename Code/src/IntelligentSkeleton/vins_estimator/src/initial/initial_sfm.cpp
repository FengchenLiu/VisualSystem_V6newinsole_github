#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state != true)
			continue;
		Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == i)
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;

}

void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = sfm_f[j].observation[k].second;
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
		if (has_0 && has_1)
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

void GlobalSFM::triangulateTwoFramesWithDepth(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
                                     int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
                                     vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
	Matrix3d Pose0_R = Pose0.block< 3,3 >(0,0);
	Matrix3d Pose1_R = Pose1.block< 3,3 >(0,0);
	Vector3d Pose0_t = Pose0.block< 3,1 >(0,3);
	Vector3d Pose1_t = Pose1.block< 3,1 >(0,3);
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		bool has_0 = false, has_1 = false;
		Vector3d point0;
		Vector2d point1;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation_depth[k].second < 0.1 || sfm_f[j].observation_depth[k].second >10) //max and min measurement
				continue;
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = Vector3d(sfm_f[j].observation[k].second.x()*sfm_f[j].observation_depth[k].second,sfm_f[j].observation[k].second.y()*sfm_f[j].observation_depth[k].second,sfm_f[j].observation_depth[k].second);
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
		if (has_0 && has_1)
		{
			Vector2d residual;
			Vector3d point_3d, point1_reprojected;
			//triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			point_3d = Pose0_R.transpose()*point0 - Pose0_R.transpose()*Pose0_t;//shan add:this is point in world;
			point1_reprojected = Pose1_R*point_3d+Pose1_t;

			residual = point1 - Vector2d(point1_reprojected.x()/point1_reprojected.z(),point1_reprojected.y()/point1_reprojected.z());

			//std::cout << residual.transpose()<<"norm"<<residual.norm()*460<<endl;
			if (residual.norm() < 1.0/460){
				sfm_f[j].state = true;
				sfm_f[j].position[0] = point_3d(0);
				sfm_f[j].position[1] = point_3d(1);
				sfm_f[j].position[2] = point_3d(2);
			}
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}
	}
}

// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size();
	//cout << "set 0 and " << l << " as known " << endl;
	// have relative_r relative_t
	// intial two view
	q[l].w() = 1;
	q[l].x() = 0;
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero();
	q[frame_num - 1] = q[l] * Quaterniond(relative_R);
	T[frame_num - 1] = relative_T;

	//rotate to cam frame
	Matrix3d c_Rotation[frame_num];
	Vector3d c_Translation[frame_num];
	Quaterniond c_Quat[frame_num];
	double c_pose[frame_num][6];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];

	c_Quat[l] = q[l].inverse();
	c_Rotation[l] = c_Quat[l].toRotationMatrix();
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]);

	Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

	c_Quat[frame_num - 1] = q[frame_num - 1].inverse();
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];


	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		if (i > l)
		{
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
				return false;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// triangulate point based on the solve pnp result
		triangulateTwoFramesWithDepth(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f);
	}
	//3: triangulate l-----l+1 l+2 ... frame_num -2
	for (int i = l + 1; i < frame_num - 1; i++)
		triangulateTwoFramesWithDepth(l, Pose[l], i, Pose[i], sfm_f);
	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f))
			return false;
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		triangulateTwoFramesWithDepth(i, Pose[i], l, Pose[l], sfm_f);
	}
	//5: triangulate all other points
	for (int j = 0; j < feature_num; j++)
	{
		if (sfm_f[j].state == true)
			continue;
		if ((int)sfm_f[j].observation.size() >= 2)
		{
			Vector3d point0;
			Vector2d point1;
			int frame_0 = sfm_f[j].observation[0].first;
			if (sfm_f[j].observation_depth[0].second < 0.1 || sfm_f[j].observation_depth[0].second > 10) //max and min measurement
				continue;
			point0 = Vector3d(sfm_f[j].observation[0].second.x()*sfm_f[j].observation_depth[0].second,sfm_f[j].observation[0].second.y()*sfm_f[j].observation_depth[0].second,sfm_f[j].observation_depth[0].second);
			int frame_1 = sfm_f[j].observation.back().first;
			point1 = sfm_f[j].observation.back().second;
			Vector3d point_3d;
			//triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);

			Matrix3d Pose0_R = Pose[frame_0].block< 3,3 >(0,0);
			Matrix3d Pose1_R = Pose[frame_1].block< 3,3 >(0,0);
			Vector3d Pose0_t = Pose[frame_0].block< 3,1 >(0,3);
			Vector3d Pose1_t = Pose[frame_1].block< 3,1 >(0,3);

			Vector2d residual;
			Vector3d point1_reprojected;
			//triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			point_3d = Pose0_R.transpose()*point0 - Pose0_R.transpose()*Pose0_t;//point in world;
			point1_reprojected = Pose1_R*point_3d+Pose1_t;

			residual = point1 - Vector2d(point1_reprojected.x()/point1_reprojected.z(),point1_reprojected.y()/point1_reprojected.z());

			if (residual.norm() < 1.0/460) {//reprojection error
				sfm_f[j].state = true;
				sfm_f[j].position[0] = point_3d(0);
				sfm_f[j].position[1] = point_3d(1);
				sfm_f[j].position[2] = point_3d(2);
			}
		}		
	}
	//full BA
	ceres::Problem problem;
	ceres::LocalParameterization* local_param = new PoseLocalLeftMultiplyParameterization();
	ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
	//cout << " begin full BA " << endl;
	for (int i = 0; i < frame_num; i++)
	{
		/* change optimizer */
		c_pose[i][0] = c_Translation[i].x();
		c_pose[i][1] = c_Translation[i].y();
		c_pose[i][2] = c_Translation[i].z();
		Eigen::Vector3d rvec = Sophus::SO3d(c_Quat[i]).log();
		c_pose[i][3] = rvec.x();
		c_pose[i][4] = rvec.y();
		c_pose[i][5] = rvec.z();

		/* change optimizer*/
		problem.AddParameterBlock(c_pose[l], 6, local_param);
		if (i == l)
		{
			problem.SetParameterBlockConstant(c_pose[i]);
		}
	}

	
	for (int i = 0; i < feature_num; i++)
	{
		if (sfm_f[i].state != true)
			continue;
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++)
		{
			int l = sfm_f[i].observation[j].first;
			ProjectionXYZFactor* projection_xyz_factor = new ProjectionXYZFactor(sfm_f[i].observation[j].second);
			problem.AddResidualBlock(projection_xyz_factor, nullptr, c_pose[l], sfm_f[i].position);	
			projection_edges.push_back(std::make_pair(std::make_pair(l, i), sfm_f[i].observation[j].second));

			// std::vector<double*> parameters{c_pose[l], sfm_f[i].position};
			// projection_xyz_factor->Check(parameters.data());
		}

		for(int j = 0; j < int(sfm_f[i].observation_depth.size()); j++) 
		{
			int l = sfm_f[i].observation_depth[j].first;
			double depth = sfm_f[i].observation_depth[j].second;
			if(depth < 0.1 || depth > 4.0)
				continue;
			double sqrt_info = 1 / std::sqrt(0.003 * depth);
			DepthXYZFactor* depth_xyz_factor = new DepthXYZFactor(depth, sqrt_info);
			problem.AddResidualBlock(depth_xyz_factor, nullptr, c_pose[l], sfm_f[i].position);
			depth_edges.push_back(std::make_pair(std::make_pair(l, i), depth)); 

			// std::vector<double*> parameters{c_pose[l], sfm_f[i].position};
			// depth_xyz_factor->Check(parameters.data());
		}

	}
	

	evaluateError(c_pose, sfm_f);
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << "\n";
	if (!summary.termination_type == ceres::CONVERGENCE)
	{
		return false;
	}
	
	evaluateError(c_pose, sfm_f);
	for (int i = 0; i < frame_num; i++)
	{
		Eigen::Quaterniond qi = Sophus::SO3d::exp(Eigen::Vector3d(c_pose[i][3], c_pose[i][4], c_pose[i][5])).unit_quaternion();
		q[i] = qi.inverse();
	}
	for (int i = 0; i < frame_num; i++)
	{
		T[i] = -1 * (q[i] * Vector3d(c_pose[i][0], c_pose[i][1], c_pose[i][2]));
	}
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state)
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]);
	}
	return true;

}

void GlobalSFM::evaluateError(double c_pose[][6], const vector<SFMFeature> &sfm_f)
{
	double reprojection_error = 0.0, depth_error = 0.0;
	int reprojection_cnt = 0, depth_cnt = 0;
	for(auto e: projection_edges) {
		int frame_id = e.first.first;
		int point_id = e.first.second;
		Eigen::Vector2d image_ob = e.second;

		Eigen::Vector3d tcw(c_pose[frame_id][0], c_pose[frame_id][1], c_pose[frame_id][2]);
		Sophus::SO3d Rcw = Sophus::SO3d::exp(Eigen::Vector3d(c_pose[frame_id][3], c_pose[frame_id][4], c_pose[frame_id][5]));
		Eigen::Vector3d Pw(sfm_f[point_id].position[0], sfm_f[point_id].position[1], sfm_f[point_id].position[2]);

		Eigen::Vector3d Pc = Rcw * Pw + tcw;
		reprojection_error += (Pc.head<2>() / Pc.z() - image_ob).norm();
		reprojection_cnt++;
	}
	if(reprojection_cnt != 0)
		reprojection_error /= reprojection_cnt;

	for(auto e: depth_edges) {
		int frame_id = e.first.first;
		int point_id = e.first.second;
		double depth_ob = e.second;

		Eigen::Vector3d tcw(c_pose[frame_id][0], c_pose[frame_id][1], c_pose[frame_id][2]);
		Sophus::SO3d Rcw = Sophus::SO3d::exp(Eigen::Vector3d(c_pose[frame_id][3], c_pose[frame_id][4], c_pose[frame_id][5]));
		Eigen::Vector3d Pw(sfm_f[point_id].position[0], sfm_f[point_id].position[1], sfm_f[point_id].position[2]);

		Eigen::Vector3d Pc = Rcw * Pw + tcw;
		depth_error += std::fabs(Pc.z() - depth_ob);
		depth_cnt++;
	}
	if(depth_cnt != 0)
		depth_error /= depth_cnt;
	printf("[GlobalSFM::evaluateError] ATE P: %lf, ATE D: %lf\n", reprojection_error, depth_error);
}

