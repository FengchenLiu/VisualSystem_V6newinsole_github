#pragma once

#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <pangolin/gl/glvbo.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>

#include <chrono>
#include <thread>
#include <condition_variable>
#include <mutex>

#include <opencv4/opencv2/opencv.hpp>
#include "tsdf_mapper.h"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>


extern std::mutex display_mutex;
extern std::condition_variable display_cv;

class PangoCloud
{
public:
  typedef std::shared_ptr<PangoCloud> Ptr;

  PangoCloud(pcl::PointCloud<pcl::PointXYZRGB> * cloud)
    : numPoints(cloud->size()),
      offset(4),
      stride(sizeof(pcl::PointXYZRGB))
  {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, cloud->points.size() * stride, cloud->points.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  PangoCloud(pcl::PointCloud<pcl::PointXYZRGBNormal> * cloud)
    : numPoints(cloud->size()),
      offset(8),
      stride(sizeof(pcl::PointXYZRGBNormal))
  {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, cloud->points.size() * stride, cloud->points.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  virtual ~PangoCloud()
  {
    glDeleteBuffers(1, &vbo);
  }

  void drawPoints()
  {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(3, GL_FLOAT, stride, 0);
    glColorPointer(3, GL_UNSIGNED_BYTE, stride, (void *)(sizeof(float) * offset));

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glDrawArrays(GL_POINTS, 0, numPoints);

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  const int numPoints;

private:
  const int offset;
  const int stride;
  GLuint vbo;

};


class Displayer 
{
public:
  typedef std::shared_ptr<Displayer> Ptr;

  Displayer(TSDFMapper::Ptr mapper);

  ~Displayer() {}

  void Run();

  pangolin::OpenGlMatrix GetPGLCameraPose();

  void DrawSurface(const pangolin::Var<bool>& bShowSurface, const pangolin::Var<bool>& bShowVertex, const pangolin::Var<bool>& bShowVirtualVertex);

  void DrawAxis(const pangolin::Var<bool> bShowBbx);

  void AddCamera(const Eigen::Matrix4d& Twc, float r, float g, float b, float lw, float scale = 1.0f);

  void DrawCamera(const Eigen::Matrix3d& Rwc, const Eigen::Vector3d& twc);

  void DrawSlidingWindow();

  void DrawRegister(const Eigen::Matrix4d& Twc_raw, const Eigen::Matrix4d& Twc_cur);

  void DrawImage();

  void DrawNormals();

  void DrawResiduals();

  void DrawPlaneSeg();

  void DrawTerrainSeg();

  void DrawGroundPos();

  void SetPose(const Eigen::Matrix3d& Rwc, const Eigen::Vector3d& twc);

  void SetImage(const cv::Mat& color_img, const cv::Mat& depth_img);

  void SetHostImage(const cv::Mat& host_vertex, const cv::Mat& host_normal);

  void SetRenderedImage(const cv::Mat& virtual_depth_img, const cv::Mat& virtual_normal_img);

  void SetSegmentationImage(const cv::Mat& segmentation_img);

  void SetRegisterResult(const Eigen::Matrix4d &Twc_raw, const Eigen::Matrix4d& Twc_cur);

  void SetAssociateResult(const std::vector<SurfaceResidual>& residuals, const Eigen::Matrix4d& Twc_host, const Eigen::Matrix4d& Twc_cur);

  void SetSlidingWindow(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3f>> &landmarks, const std::vector<Eigen::Matrix4d> &poses);

  void SetPlaneSegResult(const std::vector<cv::Vec3b>& _colors, const std::vector<Eigen::Vector3d>& _centers, const std::vector<Eigen::Vector3d>& _normals);

  void SetTerrainSegResult(const std::vector<std::vector<std::pair<Eigen::Vector3d, cv::Vec3b>>>& result);

  void SetGroundPos(const unsigned short plane_id, const Eigen::Vector3d& ground_pos);

  bool GetResetFlag() {
    bool current_reset_flag = reset_flag_;
    reset_flag_ = false; 
    return current_reset_flag;
  }

  void Stop();

  TSDFMapper::Ptr mapper_;
  PangoCloud::Ptr surface;
  PangoCloud::Ptr vertex;
  PangoCloud::Ptr virtual_vertex;

  Eigen::Matrix3d Rwc_;
  Eigen::Vector3d twc_;
  std::vector<Eigen::Vector3d> trajectory;


  // for register
  Eigen::Matrix4d Twc_raw_;
  Eigen::Matrix4d Twc_cur_;

  cv::Mat color_img_, depth_img_, vis_depth_img_, vis_normal_img_, segmentation_img_;
  cv::Mat host_vertex_img_, host_normal_img_;

  // for association
  std::vector<SurfaceResidual> residuals_;
  Eigen::Matrix4d host_Twc_, cur_Twc_;

  // for initialization
  std::vector<Eigen::Vector3d> landmarks;
  std::vector<Eigen::Matrix4d> poses;

  // for sliding window
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3f>> window_landmarks;
  std::vector<Eigen::Matrix4d> window_poses;
  
  // for plane segmentation
  std::vector<cv::Vec3b> colors;
  std::vector<Eigen::Vector3d> centers, normals;

  // for terrain segmentation
  std::vector<std::vector<std::pair<Eigen::Vector3d, cv::Vec3b>>> terrain;

  // for ground status
  unsigned short plane_id_;
  Eigen::Vector3d ground_pos_;

  int width, height;
  float max_depth, min_depth;

  bool reset_flag_ = false;

  std::mutex pose_mutex;
  std::mutex stop_mutex;
  std::mutex image_mutex;
  std::mutex window_mutex;
  std::mutex plane_seg_mutex;
  std::mutex terrain_seg_mutex;
  std::mutex ground_state_mutex;
  bool stop_;
  bool pose_ok_;
  bool image_ok_;
  bool register_ok_;
  bool match_ok_;
  bool host_data_ok_;
  bool sliding_window_ok_;
  bool plane_seg_ok_;
  bool terrain_seg_ok_;
  bool ground_state_ok_;
};

