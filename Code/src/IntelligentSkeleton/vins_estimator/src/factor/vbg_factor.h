#pragma once

#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"
#include "integration_base.h"

#include "so3.hpp"
#include "se3.hpp"

class VBGdirFactor : public ceres::SizedCostFunction<15, 3, 3, 3, 3, 3, 3, 2>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VBGdirFactor() = delete;
  VBGdirFactor(IntegrationBase* _integration_ij, 
               const Eigen::Matrix3d& _Ri, 
               const Eigen::Matrix3d& _Rj,
               const Eigen::Vector3d& _Pi,
               const Eigen::Vector3d& _Pj) :
  integration_ij(_integration_ij), Ri(_Ri), Rj(_Rj), Pi(_Pi), Pj(_Pj) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
  
  bool CheckJacobian(double const* para_Vi, double const* para_Bai, double const* para_Bgi,
                                 double const* para_Vj, double const* para_Baj, double const* para_Bgj,
                                 double const* para_euler);

private:
  IntegrationBase* integration_ij;
  Eigen::Matrix3d Ri, Rj;
  Eigen::Vector3d Pi, Pj;
};