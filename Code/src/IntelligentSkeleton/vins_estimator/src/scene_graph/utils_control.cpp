#include "utils_control.h"
#include <iostream>

namespace c_comp {
Eigen::Vector2d cal_avg_velo_xy(std::deque<Eigen::Vector3d> deq_Velo) {
    Eigen::Vector3d avg_velo;
    avg_velo << 0.0, 0.0, 0.0;
    for (std::deque<Eigen::Vector3d>::iterator iter = deq_Velo.begin(); iter != deq_Velo.end(); ++iter) {
        avg_velo += *(iter);
    }

    // std::cout << "deq_Velo.size()" << std::endl
    //           << deq_Velo.size() << std::endl;
    avg_velo /= deq_Velo.size();

    // std::cout << "avg_velo" << std::endl
    //           << avg_velo << std::endl;

    Eigen::Vector2d res;
    res << avg_velo[0], avg_velo[1];

    auto norm_res = res.norm();
    res << res[0] / norm_res, res[1] / norm_res;

    return res;
};

inline Eigen::Vector4d cvt_point_normal_plane_2_ABCD(const Eigen::Vector3d &start_point, const Eigen::Vector3d &plane_norm) {
    double A = plane_norm[0];
    double B = plane_norm[1];
    double C = plane_norm[2];
    double D = -start_point[0] * A - start_point[1] * B - start_point[2] * C;

    Eigen::Vector4d res;
    res << A, B, C, D;

    return res;
}

inline double cal_point_plane_dist(const Eigen::Vector4d &plane_paras, const pcl::PointXYZRGB &point) {
    return abs(plane_paras[0] * point.x + plane_paras[1] * point.y + plane_paras[2] * point.z + plane_paras[3]) / sqrt(plane_paras[0] * plane_paras[0] + plane_paras[1] * plane_paras[1] + plane_paras[2] * plane_paras[2]);
};

inline Eigen::Vector3d cal_projection(const Eigen::Vector4d &plane_paras, const Eigen::Vector3d &point) {
    double A_squ = plane_paras[0] * plane_paras[0];
    double B_squ = plane_paras[1] * plane_paras[1];
    double C_squ = plane_paras[2] * plane_paras[2];

    double x_p = ((B_squ + C_squ) * point[0] - plane_paras[0] * (plane_paras[1] * point[1] + plane_paras[2] * point[2] + plane_paras[3])) / (A_squ + B_squ + C_squ);

    double y_p = ((A_squ + C_squ) * point[1] - plane_paras[1] * (plane_paras[0] * point[0] + plane_paras[2] * point[2] + plane_paras[3])) / (A_squ + B_squ + C_squ);

    double z_p = ((A_squ + B_squ) * point[2] - plane_paras[2] * (plane_paras[0] * point[0] + plane_paras[1] * point[1] + plane_paras[3])) / (A_squ + B_squ + C_squ);

    Eigen::Vector3d res;
    res << x_p, y_p, z_p;

    return res;
}

void filter_and_project(pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud,
                        double dis_threshold,
                        const Eigen::Vector3d &start_point,
                        const Eigen::Vector3d &plane_norm,
                        std::vector<std::string> &vec_color_key,
                        std::unordered_map<std::string, std::vector<Eigen::Vector3d>> &map_planes,
                        std::unordered_map<std::string, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &map_planes_in_pointclouds,
                        float img_resolu,
                        cv::Mat &vis_img,
                        const Eigen::Vector3d &hori_vec) {
    Eigen::Vector4d target_plane_paras = cvt_point_normal_plane_2_ABCD(start_point, plane_norm);

    // std::cout << "the params of the target plane: " << std::endl
    //           << target_plane_paras << std::endl;

    for (const auto &point : point_cloud -> points) {

        // // lfc 删除跑台 ,判断 到当前z位置，1m以内的删掉。
        // if ( std::abs(point.z - start_point[2] )<1 ) {
        //     continue;  // 跳过
        // }

        double dist = cal_point_plane_dist(target_plane_paras, point);
        if (dist > dis_threshold) {
            continue;
        }
        Eigen::Vector3d p_pos;
        std::string color_key = std::to_string(point.r) + std::to_string(point.g) + std::to_string(point.b);
        p_pos << point.x, point.y, point.z;

        if (map_planes.count(color_key) == 0){
            map_planes.emplace(std::pair<std::string, std::vector<Eigen::Vector3d>> {color_key, std::vector<Eigen::Vector3d>{}});
            vec_color_key.push_back(color_key);
        }
        map_planes.at(color_key).push_back(p_pos);

        if (map_planes_in_pointclouds.count(color_key) == 0) {
            map_planes_in_pointclouds.emplace(std::pair<std::string, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> {color_key, new pcl::PointCloud<pcl::PointXYZRGB>});
        }
        map_planes_in_pointclouds.at(color_key) -> push_back(point);

        Eigen::Vector3d proj_p_pose = cal_projection(target_plane_paras, p_pos);

        unsigned int img_coord_1 = (- (proj_p_pose[2] - start_point[2])) / img_resolu;

        // std::cout << "proj_p_pose: " << std::endl
        //           << proj_p_pose << std::endl; 

        // std::cout << "(proj_p_pose[2] - start_point[2])" << std::endl
        //           << (proj_p_pose[2] - start_point[2]) << std::endl;

        // std::cout << "img coord 1: " << std::endl
        //           << img_coord_1 << std::endl;

        unsigned int img_coord_2 = ((proj_p_pose[0] - start_point[0]) / hori_vec[0]) / img_resolu;

        // std::cout << "img coord 2: " << std::endl
        //           << img_coord_2 << std::endl;
        
        vis_img.at<cv::Vec3b>(img_coord_1, img_coord_2) = cv::Vec3b(point.r, point.g, point.b);
    }

    // std::cout << "size of map planes in pointclouds: " << std::endl
    //           << map_planes_in_pointclouds.size() << std::endl;
};






void fit_3D_plane(std::vector<std::string> &vec_color_key,
                  std::unordered_map<std::string, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> map_planes_in_pointclouds,
                  std::vector<std::string> &eff_vec_color_key) {
    // the plane are fitted in this function
    
    for (const auto & color_key : vec_color_key) {
        // std::cout << "color_key" << std::endl
        //           << color_key << std::endl;

        // std::cout << "counts of color_key in the map" << std::endl
        //           << map_planes_in_pointclouds.count(color_key) << std::endl;

        if (color_key == "000") {
            continue;
        }

        if (map_planes_in_pointclouds[color_key] -> points.size() < 10) {
            std::cout << "do not have enough points" << std::endl;
            continue;
        }

        pcl::SACSegmentation<pcl::PointXYZRGB> seg;
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(100);
        seg.setDistanceThreshold(0.05);
        seg.setInputCloud(map_planes_in_pointclouds[color_key]);
        seg.segment(*inliers, *coefficients);

        // if ( !((*coefficients).values[2] > 0.96 || (*coefficients).values[2] < -0.96)) {            // 只保留水平 平面
        // if ( !((*coefficients).values[2] > 0.84 || (*coefficients).values[2] < -0.84)) {      //lfc 尝试把斜坡给进去 ，
        if ( !((*coefficients).values[2] > 0.80 || (*coefficients).values[2] < -0.80)) {      //lfc 尝试把斜坡给进去
            continue;
        }

        // std::cout << "coefficients" << std::endl
        //           << *coefficients << std::endl;

        // std::cout << "coefficients.values[0]" << std::endl
        //           << (*coefficients).values[0] << std::endl;

        // std::cout << "coefficients.values[1]" << std::endl
        //           << (*coefficients).values[1] << std::endl;

        // std::cout << "coefficients.values[2]" << std::endl
        //           << (*coefficients).values[2] << std::endl;

        // std::cout << "coefficients.values[3]" << std::endl
        //           << (*coefficients).values[3] << std::endl;

        if ((*coefficients).values[2] > 0.96|| (*coefficients).values[2] < -0.96) {
        // if ((*coefficients).values[2] > 0.84|| (*coefficients).values[2] < -0.84) {
            eff_vec_color_key.push_back(color_key);
        }
    };
};

void cal_min_values(std::vector<std::string> &eff_vec_color_key,
                    std::unordered_map<std::string, std::vector<Eigen::Vector3d>> &map_planes,
                    std::vector<float> &vec_min_dist,
                    Eigen::Vector3d currPwc_transferred,
                    std::vector<double> &vec_mean_z,
                    std::vector<std::pair<float, double>> &vec_pair_min_dist_mean_z) {
    // this function is used to calculated the min distance

    for (const auto & color_key : eff_vec_color_key) {
        // std::cout << "******************************************" << std::endl;
        // std::cout << "the top of for loop" << std::endl;
        // std::cout << "current color key " << std::endl
        //           << color_key << std::endl;
        auto points = map_planes.at(color_key);
        double min_dist = 1000;
        double mean_z = 0;
        // std::cout << "before the internal for loop" << std::endl;
        // std::cout << "size of current points: " << std::endl
        //           << points.size() << std::endl;
        for (const auto & point : points) {
            // std::cout << "point: " << std::endl
            //           << point << std::endl;
            double curr_dist = sqrt((currPwc_transferred[0] - point[0]) * (currPwc_transferred[0] - point[0]) + (currPwc_transferred[1] - point[1]) * (currPwc_transferred[1] - point[1]));

            mean_z += point[2];

            if (curr_dist < min_dist) {
                min_dist = curr_dist;
            }
        }
        // std::cout << "the calculated minimum distance: " << std::endl;
        // std::cout << min_dist << std::endl;
        mean_z = mean_z / points.size();
        // std::cout << "mean z: " << std::endl
        //           << mean_z << std::endl;
        vec_min_dist.push_back(min_dist);
        vec_mean_z.push_back(mean_z);
        vec_pair_min_dist_mean_z.push_back(std::pair<float, double> {min_dist, mean_z});
        // std::cout << "after the push back area" << std::endl;
        // std::cout << "******************************************" << std::endl;
    }
};
}
