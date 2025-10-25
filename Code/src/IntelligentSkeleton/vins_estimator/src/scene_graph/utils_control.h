#ifndef _UTILS_CONTROL_COMPUTATION_H
#define _UTILS_CONTROL_COMPUTATION_H
#include <deque>
#include <vector>
#include <string>

#include <Eigen/Core>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <opencv2/opencv.hpp>

#include <boost/asio.hpp>

namespace c_comp {
Eigen::Vector2d cal_avg_velo_xy(std::deque<Eigen::Vector3d> deq_Velo);
inline Eigen::Vector4d cvt_point_normal_plane_2_ABCD(const Eigen::Vector3d &start_point, const Eigen::Vector3d &plane_norm);
inline double cal_point_plane_dist(const Eigen::Vector4d &plane_paras, const pcl::PointXYZRGB &point);
inline Eigen::Vector3d cal_projection(const Eigen::Vector4d &plane_paras, const Eigen::Vector3d &point);
void filter_and_project(pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud,
                        double dis_threshold,
                        const Eigen::Vector3d &start_point,
                        const Eigen::Vector3d &plane_norm,
                        std::vector<std::string> &vec_color_key,
                        std::unordered_map<std::string, std::vector<Eigen::Vector3d>> &map_planes,
                        std::unordered_map<std::string, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &map_planes_in_pointclouds,
                        float img_resolu,
                        cv::Mat &vis_img,
                        const Eigen::Vector3d &hori_vec);
void fit_3D_plane(std::vector<std::string> &vec_color_key, 
                  std::unordered_map<std::string, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> map_planes_in_pointclouds,
                  std::vector<std::string> &eff_vec_color_key);

void cal_min_values(std::vector<std::string> &eff_vec_color_key,
                    std::unordered_map<std::string, std::vector<Eigen::Vector3d>> &map_planes,
                    std::vector<float> &vec_min_dist,
                    Eigen::Vector3d currPwc_transferred,
                    std::vector<double> &vec_mean_z,
                    std::vector<std::pair<float, double>> &vec_pair_min_dist_mean_z);
};

#endif