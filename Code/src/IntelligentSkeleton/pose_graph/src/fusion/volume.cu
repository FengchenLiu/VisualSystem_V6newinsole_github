#include "volume.h"




void eventCreate(Event *event) {
  gpuErrchk(cudaEventCreateWithFlags(&event->cpy_htd, eventFlags()));
  gpuErrchk(cudaEventCreateWithFlags(&event->compute, eventFlags()));
  gpuErrchk(cudaEventCreateWithFlags(&event->cpy_dth, eventFlags()));
}

void eventDestroy(Event *event) {
  gpuErrchk(cudaEventDestroy(event->cpy_htd));
  gpuErrchk(cudaEventDestroy(event->compute));
  gpuErrchk(cudaEventDestroy(event->cpy_dth));
}

void eventSynchronize(Event *event) {
  gpuErrchk(cudaEventSynchronize(event->cpy_dth));
}


__global__ void Integrate2D(Voxel *voxels, uchar3 *colormap, float *depthmap, float *K, float *Twc,
                            int grid_dim_x, int grid_dim_y, int grid_dim_z,
                            float grid_origin_x, float grid_origin_y, float grid_origin_z, 
                            float voxel_size, int height, int width, float trunc_margin)
{
  int pt_grid_z = blockIdx.x;
  int pt_grid_y = threadIdx.x;

  for (int pt_grid_x = 0; pt_grid_x < grid_dim_x; pt_grid_x+=1) {
    // Convert voxel center from grid coordinates to base frame camera coordinates
    float pt_base_x = grid_origin_x + pt_grid_x * voxel_size;
    float pt_base_y = grid_origin_y + pt_grid_y * voxel_size;
    float pt_base_z = grid_origin_z + pt_grid_z * voxel_size;

    // Convert from base frame camera coordinates to current frame camera coordinates
    float tmp_pt[3] = {0};
    tmp_pt[0] = pt_base_x - Twc[0 * 4 + 3];
    tmp_pt[1] = pt_base_y - Twc[1 * 4 + 3];
    tmp_pt[2] = pt_base_z - Twc[2 * 4 + 3];
    float pt_cam_x = Twc[0 * 4 + 0] * tmp_pt[0] + Twc[1 * 4 + 0] * tmp_pt[1] + Twc[2 * 4 + 0] * tmp_pt[2];
    float pt_cam_y = Twc[0 * 4 + 1] * tmp_pt[0] + Twc[1 * 4 + 1] * tmp_pt[1] + Twc[2 * 4 + 1] * tmp_pt[2];
    float pt_cam_z = Twc[0 * 4 + 2] * tmp_pt[0] + Twc[1 * 4 + 2] * tmp_pt[1] + Twc[2 * 4 + 2] * tmp_pt[2];

    if (pt_cam_z <= 0)
      continue;

    int pt_pix_x = roundf(K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + K[0 * 3 + 2]);
    int pt_pix_y = roundf(K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + K[1 * 3 + 2]);
    if (pt_pix_x < 0 || pt_pix_x >= width || pt_pix_y < 0 || pt_pix_y >= height)
      continue;

    uchar3 color_val = colormap[pt_pix_y * width + pt_pix_x];
    float depth_val = depthmap[pt_pix_y * width + pt_pix_x];

    if (depth_val <= 0 || depth_val > 6)
      continue;

    float sdf = depth_val - pt_cam_z;


    // Integrate
    int volume_idx = pt_grid_z * grid_dim_y * grid_dim_x + pt_grid_y * grid_dim_x + pt_grid_x;
    voxels[volume_idx].Combine(sdf, trunc_margin, color_val, 1, 64);
  }
}


__global__ void Integrate3D(Voxel *voxels, uchar3 *colormap, float *depthmap, float *K, float *Twc,
                            int grid_dim_x, int grid_dim_y, int grid_dim_z,
                            float grid_origin_x, float grid_origin_y, float grid_origin_z, 
                            float voxel_size, int height, int width, float trunc_margin, float max_depth, float min_depth)
{

  int pt_grid_z = blockIdx.x * blockDim.x + threadIdx.x;
  int pt_grid_y = blockIdx.y * blockDim.y + threadIdx.y;


  for (int pt_grid_x = 0; pt_grid_x < grid_dim_x; pt_grid_x+=1) {
    // Convert voxel center from grid coordinates to base frame camera coordinates
    float pt_base_x = grid_origin_x + pt_grid_x * voxel_size;
    float pt_base_y = grid_origin_y + pt_grid_y * voxel_size;
    float pt_base_z = grid_origin_z + pt_grid_z * voxel_size;

    // Convert from base frame camera coordinates to current frame camera coordinates
    float tmp_pt[3] = {0};
    tmp_pt[0] = pt_base_x - Twc[0 * 4 + 3];
    tmp_pt[1] = pt_base_y - Twc[1 * 4 + 3];
    tmp_pt[2] = pt_base_z - Twc[2 * 4 + 3];
    float pt_cam_x = Twc[0 * 4 + 0] * tmp_pt[0] + Twc[1 * 4 + 0] * tmp_pt[1] + Twc[2 * 4 + 0] * tmp_pt[2];
    float pt_cam_y = Twc[0 * 4 + 1] * tmp_pt[0] + Twc[1 * 4 + 1] * tmp_pt[1] + Twc[2 * 4 + 1] * tmp_pt[2];
    float pt_cam_z = Twc[0 * 4 + 2] * tmp_pt[0] + Twc[1 * 4 + 2] * tmp_pt[1] + Twc[2 * 4 + 2] * tmp_pt[2];

    if (pt_cam_z <= 0)
      continue;

    int pt_pix_x = roundf(K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + K[0 * 3 + 2]);
    int pt_pix_y = roundf(K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + K[1 * 3 + 2]);
    if (pt_pix_x < 0 || pt_pix_x >= width || pt_pix_y < 0 || pt_pix_y >= height)
      continue;

    uchar3 color_val = colormap[pt_pix_y * width + pt_pix_x];
    float depth_val = depthmap[pt_pix_y * width + pt_pix_x];

    if (depth_val <= min_depth || depth_val > max_depth)
      continue;

    float sdf = depth_val - pt_cam_z;


    // Integrate
    int volume_idx = pt_grid_z * grid_dim_y * grid_dim_x + pt_grid_y * grid_dim_x + pt_grid_x;
    voxels[volume_idx].Combine(sdf, trunc_margin, color_val, 1, 64);
  }
}

void Volume::UpdateTSDF(uchar3 *bgr, float* depth, const float* Twc)
{
  if (int_event_) {
    gpuErrchk(cudaEventSynchronize(int_event_->cpy_dth))
    eventDestroy(int_event_);
  } else {
    int_event_ = new Event;
  }
  eventCreate(int_event_);

  // Copy inputs to page-locked memory
  int num_pixels = height * width;

  memcpy((void *) h_bgr_in_, (void *) bgr, sizeof(uchar3) * num_pixels);
  memcpy((void *) h_depth_in_, (void *) depth, sizeof(float) * num_pixels);
  memcpy((void *) h_Twc_in_, (void *) Twc, sizeof(float) * 16);

  // Copy mem to device
  gpuErrchk(cudaMemcpyAsync(d_bgr_in_, h_bgr_in_, sizeof(uchar3) * num_pixels, cudaMemcpyHostToDevice, int_stream_));
  gpuErrchk(cudaMemcpyAsync(d_depth_in_, h_depth_in_, sizeof(float) * num_pixels, cudaMemcpyHostToDevice, int_stream_));
  gpuErrchk(cudaMemcpyAsync(d_Twc_in_, h_Twc_in_, sizeof(float) * 16, cudaMemcpyHostToDevice, int_stream_));
  gpuErrchk(cudaEventRecord(int_event_->cpy_htd, int_stream_));
  
  // Integrate2D<<< grid_dim_z, grid_dim_y >>>(voxels, d_bgr_in_, d_depth_in_, d_cam_K_in_, d_Twc_in_, 
  //                                           grid_dim_x, grid_dim_y, grid_dim_z, grid_origin_x, grid_origin_y, grid_origin_z, 
  //                                           voxel_size, height, width, 5 * voxel_size);

  dim3 grid_dim(32, 32, 1);
  dim3 block_dim(16, 16, 1);
  Integrate3D<<< grid_dim, block_dim >>>(voxels, d_bgr_in_, d_depth_in_, d_cam_K_in_, d_Twc_in_, 
                                            grid_dim_x, grid_dim_y, grid_dim_z, grid_origin_x, grid_origin_y, grid_origin_z, 
                                            voxel_size, height, width, 5 * voxel_size, max_depth, min_depth);
  cudaDeviceSynchronize();
}
