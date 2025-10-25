// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include <cuda_runtime.h>
#include "voxel.h"
#include "cuda_matrix.h"

#include "opencv2/opencv.hpp"


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

class Volume {
public:
  int grid_dim_x, grid_dim_y, grid_dim_z, height, width;
  float voxel_size;

  Voxel *voxels;

  uchar3 *d_bgr_in_, *h_bgr_in_;
  float *d_depth_in_, *h_depth_in_;
  float *d_cam_K_in_, *h_cam_K_in_; 
  float *d_Twc_in_, *h_Twc_in_; 

  float grid_origin_x, grid_origin_y, grid_origin_z;
  float max_depth, min_depth;

  cudaStream_t int_stream_;
  Event *int_event_ = NULL;

  Volume() {}

  ~Volume()
  {
    cudaDeviceSynchronize();
    gpuErrchk(cudaStreamDestroy(int_stream_));
    delete int_event_;

    gpuErrchk(cudaFree(d_bgr_in_));
    gpuErrchk(cudaFree(d_depth_in_));
    gpuErrchk(cudaFree(d_cam_K_in_));
    gpuErrchk(cudaFree(d_Twc_in_));
    
    gpuErrchk(cudaFreeHost(h_bgr_in_));
    gpuErrchk(cudaFreeHost(h_depth_in_));
    gpuErrchk(cudaFreeHost(h_cam_K_in_));
    gpuErrchk(cudaFreeHost(h_Twc_in_));

    gpuErrchk(cudaFree(&voxels));
  }
  
  void Init(int _height, int _width, int _grid_dim_x, int _grid_dim_y, int _grid_dim_z, float _voxel_size, float _max_depth, float _min_depth, const float* cam_K)
  {
    height = _height;
    width = _width;
    max_depth = _max_depth;
    min_depth = _min_depth;
    grid_dim_x = _grid_dim_x;
    grid_dim_y = _grid_dim_y;
    grid_dim_z = _grid_dim_z;
    voxel_size = _voxel_size;

    assert(grid_dim_x%2 == 0);
    assert(grid_dim_y%2 == 0);
    assert(grid_dim_z%2 == 0);
    grid_origin_x = -(grid_dim_x/2) * voxel_size;
    grid_origin_y = -(grid_dim_y/2) * voxel_size;
    grid_origin_z = -(grid_dim_z/2) * voxel_size;
    gpuErrchk(cudaMallocManaged((void **)&voxels, grid_dim_x * grid_dim_y * grid_dim_z * sizeof(Voxel)));

    printf("grid origin: %f, %f, %f\n", grid_origin_x, grid_origin_y, grid_origin_z);

    int num_pixels = height * width;
    gpuErrchk(cudaMalloc(&d_bgr_in_, sizeof(uchar3) * num_pixels));
    gpuErrchk(cudaMalloc(&d_depth_in_, sizeof(float) * num_pixels));
    gpuErrchk(cudaMalloc(&d_cam_K_in_, sizeof(float) * 9));
    gpuErrchk(cudaMalloc(&d_Twc_in_, sizeof(float) * 16));

    gpuErrchk(cudaMallocHost(&h_bgr_in_, sizeof(uchar3) * num_pixels));
    gpuErrchk(cudaMallocHost(&h_depth_in_, sizeof(float) * num_pixels));
    gpuErrchk(cudaMallocHost(&h_cam_K_in_, sizeof(float) * 9));
    gpuErrchk(cudaMallocHost(&h_Twc_in_, sizeof(float) * 16));
    
    gpuErrchk(cudaStreamCreate(&int_stream_));

    memcpy(h_cam_K_in_, cam_K, sizeof(float) * 9);
    gpuErrchk(cudaMemcpy(d_cam_K_in_, h_cam_K_in_, sizeof(float) * 9, cudaMemcpyHostToDevice));
  } 


  void UpdateTSDF(uchar3 *bgr, float* depth, const float* Twc);

};

