#ifndef _TSDF_MAPPER_H_
#define _TSDF_MAPPER_H_

#include "volume.h"

#include "opencv4/opencv2/core/core.hpp"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/Eigenvalues>

#include <pcl-1.10/pcl/point_cloud.h>
#include <pcl-1.10/pcl/point_types.h>

#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>

#include <mutex>
#include <condition_variable>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include <boost/foreach.hpp>

extern std::mutex display_mutex;
extern std::condition_variable display_cv;



struct SurfaceResidual
{
  Eigen::Vector3d vertex, host_vertex, host_normal;
  double sqrt_info;
  SurfaceResidual(): vertex(Eigen::Vector3d::Zero()), host_vertex(Eigen::Vector3d::Zero()), host_normal(Eigen::Vector3d::Zero()), sqrt_info(0.0) {}
  SurfaceResidual(const Eigen::Vector3d v, const Eigen::Vector3d& hv, const Eigen::Vector3d& hn, const double depth)
  : vertex(v), host_vertex(hv), host_normal(hn) 
  {
    sqrt_info = 1 / sqrt(0.002);
  }
};

class PlaneInfo
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<PlaneInfo> Ptr;
  PlaneInfo() {
    center_.setZero();    
    cov_.setZero();
    N = 0;
  }

  void Reset() {
    center_.setZero();
    cov_.setZero();
    N = 0;
  }

  void AddElement(const Eigen::Vector3d &element) {
    N++;
    Eigen::Vector3d last_center = center_;
    center_ = center_ + (element - center_) / N;
    cov_ = cov_ + (element - last_center) * (element - center_).transpose();
  }

  void SetNormal(const Eigen::Vector3d& normal) {
    normal_ = normal;
  }

  Eigen::Vector3d GetCenter() const { return center_; }

  Eigen::Matrix3d GetCov() const { return cov_ / N; }

  Eigen::Vector3d GetNormal() const {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov_ / N);
    return saes.eigenvectors().col(0);
  }

  unsigned int GetSize() const { return N; }

public:
  Eigen::Vector3d center_, normal_;
  Eigen::Matrix3d cov_;
  cv::Vec3b color;
  unsigned int N;
};



class TSDFMapper
{
public:
  typedef pcl::PointXYZRGB            ColorPoint;
  typedef pcl::PointCloud<ColorPoint> ColorCloud;
  typedef ColorCloud::Ptr             ColorCloudPtr;

  typedef std::shared_ptr<TSDFMapper> Ptr; 
  struct TSDFMappingOptions {
    int height;
    int width;
    int grid_dim_x;
    int grid_dim_y;
    int grid_dim_z;
    float voxel_size;
    float max_depth;
    float min_depth;
    cv::Mat K;
  };

  TSDFMapper(const TSDFMappingOptions& options);
  ~TSDFMapper();

  void Reset();

  void SetParametersnormal_();

  void UpdateTSDF(const cv::Mat& color_image, const cv::Mat& depth_image, const Eigen::Matrix4d& Twc);

  void UpdateSegment(const cv::Mat& index_image, const cv::Mat& depth_image, const Eigen::Matrix4d& Twc);

  void MoveVolume(const Eigen::Matrix4d& Twc);

  void ExtractPointCloud(const std::string& file, float tsdf_thresh, float weight_thresh);

  void ExtractSurface();

  void GetGroundPos(const Eigen::Matrix4d& Twc, unsigned short& plane_id, Eigen::Vector3d& ground_pos, std::vector<Eigen::Vector3d>& normals);

  bool SurfaceOK();

  ColorCloudPtr GetSurfaceCloud() const { return surface; }

  ColorCloudPtr GetVertexCloud() const { return vertex; }

  ColorCloudPtr GetVirtualVertexCloud() const { return virtual_vertex; }

  Volume *GetVolumePtr() const { return volume; }

  void RenderView(const Eigen::Matrix4d& Twc);
  
  std::map<ushort, ushort> FindMatch(const cv::Mat& host_index_img, const cv::Mat& index_img);

  void AddNewSegmentImage(cv::Mat& index_img, const cv::Mat& depth_img, const Eigen::Matrix4d& Twc,
                                const std::vector<cv::Vec3b>& colors, const std::vector<Eigen::Vector3d>& centers, 
                                std::vector<Eigen::Vector3d>& normals);

  Volume *volume;
  std::map<ushort, cv::Vec3b> color_table_;
  std::map<ushort, int> match_num_table_;
  std::set<ushort> visible_index_set_;
  std::map<ushort, PlaneInfo::Ptr> curr_surfaces_;
  std::map<ushort, Eigen::Vector3d> curr_normals_; 

  ColorCloudPtr surface;
  ColorCloudPtr vertex;
  ColorCloudPtr virtual_vertex;

  std::mutex render_mutex;
  cv::Mat host_depth_img;
  cv::Mat host_vertex_img;
  cv::Mat host_normal_img;
  cv::Mat host_index_img;
  cv::Mat host_segment_img;
  
  Eigen::Matrix4d Twc_;
  std::mutex surface_mutex;
  bool surface_flag;

  std::mutex state_mutex;
  std::condition_variable display;
  static ushort factory_plane_id;
};
#endif