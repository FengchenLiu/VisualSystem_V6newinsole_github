#pragma once

#include <cuda_runtime.h>
#include "voxel.h"
#include "cuda_matrix.h"

#include "opencv2/opencv.hpp"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>


/**
 * @brief      Struct representing a voxel.
 */
struct Event {
  cudaEvent_t cpy_htd;
  cudaEvent_t compute;
  cudaEvent_t cpy_dth;
};

void eventCreate(Event *event);

void eventDestroy(Event *event);

void eventSynchronize(Event *event);

__global__ void SelectRegionKernel(uchar3 *segment_img, uchar *binary_img, uchar3 target, int height, int width);

#define DIVIDE 16

class Volume {
public:
  int grid_dim_x, grid_dim_y, grid_dim_z, height, width;
  float voxel_size;

  Voxel *voxels;
  BGRPoint *surface;
  // float3 *vertex_map;
  // float3 *normal_map;
  // uchar3 *vertex_color_map;

  float3 *virtual_vertex_map;
  float3 *virtual_normal_map;
  ushort *virtual_index_map;
  // uchar3 *virtual_vertex_color_map;

  float *d_cam_K_in_; 

  float *d_virtual_depth_;

  float *d_cur_Twc, *d_host_Tcw, *h_cur_Twc;

  float grid_origin_x, grid_origin_y, grid_origin_z;
  float max_depth, min_depth;


  std::mutex warp_mutex;
  int cur_warp_x, cur_warp_y, cur_warp_z;
  int last_warp_x, last_warp_y, last_warp_z;
  int surface_num;
  int match_num;


  bool render_ok = false;
  std::mutex render_mutex;

  std::mutex volume_mutex;

  Volume() {}

  ~Volume();
  
  void Init(int _height, int _width, int _grid_dim_x, int _grid_dim_y, int _grid_dim_z, float _voxel_size, float _max_depth, float _min_depth, const float* cam_K);

  void Reset();
  
  void UpdateTSDF(const uchar3 *bgr, const float* depth, const float* Twc);

  void UpdateSegment(const ushort *index, const float* depth, const float* Twc);

  void ReleaseXPlus(int dx);

  void ReleaseXMinus(int dx);

  void ReleaseYPlus(int dy);

  void ReleaseYMinus(int dy);

  void ReleaseZPlus(int dz);

  void ReleaseZMinus(int dz);

  void Move(int vx, int vy, int vz);

  void ExtractSurface();

  void RayCasting(const float* Twc, cv::Mat& depth, cv::Mat& normal, cv::Mat& vertex, cv::Mat& index);

  void GetGroundPos(const float *Twc, unsigned short &plane_id, float &gx, float &gy, float& gz, float &nx, float &ny, float &nz);

};

