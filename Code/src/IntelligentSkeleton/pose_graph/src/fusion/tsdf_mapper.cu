#include "tsdf_mapper.h"

TSDFMapper::TSDFMapper(const TSDFMappingOptions& options)
{
  gpuErrchk(cudaMallocManaged(&volume, sizeof(Volume)));
  assert(options.K.type() == CV_32F);
  volume->Init(options.height, 
               options.width, 
               options.grid_dim_x, 
               options.grid_dim_y, 
               options.grid_dim_z, 
               options.voxel_size, 
               options.max_depth,
               options.min_depth,
               (float*)options.K.data);

}

TSDFMapper::~TSDFMapper()
{
  cudaDeviceSynchronize();
  gpuErrchk(cudaFree(&volume));
}

void TSDFMapper::UpdateTSDF(cv::Mat& color_image, cv::Mat depth_image, const cv::Mat& Twc)
{
  volume->UpdateTSDF((uchar3*)color_image.data, (float*)depth_image.data, (float*)Twc.data);
}