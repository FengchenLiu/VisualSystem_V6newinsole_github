#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"
#include "../fusion/tsdf_mapper.h"

#include "so3.hpp"
#include "se3.hpp"

class SurfaceFactor : public ceres::SizedCostFunction<1, 7>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SurfaceFactor(const SurfaceResidual &_res): res(_res) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

  void Check(double **parameters);
  
private:
  SurfaceResidual res;

};

