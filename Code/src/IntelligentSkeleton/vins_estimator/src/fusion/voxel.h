// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cmath>
/**
 * @brief      Struct representing a voxel.
 */
#define MAX_ID_UPDATE_NUM  4
#define MAX_VOXEL_WEIGHT   4

struct Voxel {

  float tsdf;
  float weight;

  uchar3 color;  
  unsigned char is_surface;  

  unsigned short plane_id;
  unsigned char plane_cnt;
  unsigned char semantic_label;

  
  
  __host__ __device__ Voxel() 
  {
    tsdf = 1.0f;
    weight = 0.0f;
    color = make_uchar3(0, 0, 0);   
    plane_id = 0;
    is_surface = 0;
  }

  __host__ __device__ void Reset()
  {
    tsdf = 1.0f;
    weight = 0.0f;
    color = make_uchar3(0, 0, 0);
    plane_id = 0;
    is_surface = 0;
  }

  /**
   * @brief      Copy From another voxel
   *
   * @param[in]  other       another voxel
   */
  __host__ __device__ void CopyFrom(const Voxel& other)
  {
    tsdf = other.tsdf;
    weight = other.weight;
    color = other.color;
  } 

  /**
   * @brief      Combine the voxel with a given one
   *
   * @param[in]  voxel       The voxel to be combined with
   * @param[in]  _max_weight  The maximum weight
   */
  __host__ __device__ void Combine(float _sdf,
                                   float _trunc_margin,
                                   uchar3 _color, // bgr
                                   float _weight,
                                   float _max_weight) 
  {

    const float curr_weight = weight;

    if(_sdf < -_trunc_margin) return;

    float _tsdf;
    if(_sdf > 0) 
      _tsdf = fmin(1.0f, _sdf / _trunc_margin);
    else if(_sdf < 0) 
      _tsdf = fmax(-1.0f, _sdf / _trunc_margin);
    else 
      _tsdf = 0.0f;

    tsdf = (tsdf * curr_weight + _tsdf * _weight) / (curr_weight + _weight);
    weight = weight + _weight;
    if (weight > _max_weight) weight = _max_weight;
    
    if(_sdf < _trunc_margin && _sdf > -_trunc_margin) {
      color.x = static_cast<unsigned char>(
            (static_cast<float>(color.x) * curr_weight + static_cast<float>(_color.z) * static_cast<float>(_weight)) /
                (curr_weight + static_cast<float>(_weight)));

      color.y = static_cast<unsigned char>(
            (static_cast<float>(color.y) * curr_weight + static_cast<float>(_color.y) * static_cast<float>(_weight)) /
                (curr_weight + static_cast<float>(_weight)));

      color.z = static_cast<unsigned char>(
            (static_cast<float>(color.z) * curr_weight + static_cast<float>(_color.x) * static_cast<float>(_weight)) /
                (curr_weight + static_cast<float>(_weight)));

    }
  }

  __host__ __device__ void Update(ushort id) {
    if(id == 0)
      return;

    if(plane_cnt == 0) {
      plane_id = id;
      plane_cnt++;
    }
    else {
      if(plane_id == id) {
        plane_cnt++;
        if(plane_cnt > MAX_ID_UPDATE_NUM)
          plane_cnt = MAX_ID_UPDATE_NUM;
      }
      else {
        plane_cnt--;
        if(plane_cnt == 0) {
          plane_id = id;
          plane_cnt++;
        }
      }
    }
  }

};

/**
 * @brief      Struct representing a BGRPoint.
 */
struct BGRPoint {
  float x, y, z;
  ushort index;
  uchar3 rgb;

  __host__ __device__ BGRPoint() 
  {
    x = 0.0f; y = 0.0f; z = 0.0f;
    rgb = make_uchar3(0, 0, 0);     
  }
  
  __host__ __device__ void Set(float _x, float _y, float _z, uchar3 _rgb, ushort _index) 
  {
    x = _x; y = _y; z = _z;
    rgb = _rgb; 
    index = _index;
  }
};