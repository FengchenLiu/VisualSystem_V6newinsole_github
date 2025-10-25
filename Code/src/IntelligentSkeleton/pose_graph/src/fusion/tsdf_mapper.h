#pragma once


#include "volume.h"
#include "opencv2/opencv.hpp"

class TSDFMapper
{
public:

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

  void UpdateTSDF(cv::Mat& color_image, cv::Mat depth_image, const cv::Mat& Twc);


  Volume *volume;
  
};