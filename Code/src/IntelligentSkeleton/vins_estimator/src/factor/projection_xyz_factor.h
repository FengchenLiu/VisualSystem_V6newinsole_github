#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"

#include "so3.hpp"
#include "se3.hpp"

class ProjectionXYZFactor : public ceres::SizedCostFunction<2, 6, 3>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ProjectionXYZFactor(const Eigen::Vector2d &_Pc_m): Pc_m(_Pc_m) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

  void Check(double **parameters);
  
  Eigen::Vector2d Pc_m;
  static Eigen::Matrix2d sqrt_info;
  static double sum_t;

};

