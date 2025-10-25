// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include <cuda_runtime.h>


/**
 * @brief      Struct representing a voxel.
 */
struct Voxel {
  /** Truncated Signed distance function */
  float tsdf;

  /** Color */
  uchar3 color;  

  /** Accumulated SDF weight */
  unsigned char weight;
  
  Voxel() 
  {
    tsdf = 1.0f;
    weight = 0.0f;
    color = make_uchar3(0, 0, 0);      
  }

  /**
   * @brief      Combine the voxel with a given one
   *
   * @param[in]  voxel       The voxel to be combined with
   * @param[in]  _max_weight  The maximum weight
   */
  __host__ __device__ void Combine(float _sdf,
                                   float _trunc_margin,
                                   uchar3 _color,
                                   unsigned char _weight,
                                   unsigned char _max_weight) 
  {

    const float curr_weight = weight;
    const float _tsdf = fmin(1.0f, _sdf / _trunc_margin);

    if (_sdf < -_trunc_margin) return;

    tsdf = (tsdf * static_cast<float>(curr_weight) + _tsdf * static_cast<float>(_weight)) / (static_cast<float>(curr_weight) + static_cast<float>(_weight));
    weight = weight + _weight;
    if (weight > _max_weight) weight = _max_weight;
    
    if(_sdf <= _trunc_margin && _sdf >= -_trunc_margin) {
      color.x = static_cast<unsigned char>(
            (static_cast<float>(color.x) * static_cast<float>(curr_weight) + static_cast<float>(_color.x) * static_cast<float>(_weight)) /
                (static_cast<float>(curr_weight) + static_cast<float>(_weight)));

      color.y = static_cast<unsigned char>(
            (static_cast<float>(color.y) * static_cast<float>(curr_weight) + static_cast<float>(_color.y) * static_cast<float>(_weight)) /
                (static_cast<float>(curr_weight) + static_cast<float>(_weight)));

      color.z = static_cast<unsigned char>(
            (static_cast<float>(color.z) * static_cast<float>(curr_weight) + static_cast<float>(_color.z) * static_cast<float>(_weight)) /
                (static_cast<float>(curr_weight) + static_cast<float>(_weight)));

    }
  }

};
