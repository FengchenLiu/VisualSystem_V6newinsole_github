#include "surface_factor.h"

bool SurfaceFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

  residuals[0] = res.sqrt_info * res.host_normal.dot(Qi * res.vertex + Pi - res.host_vertex);

  if(jacobians) {
    if(jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
      jacobian_pose.setZero();
      jacobian_pose.block<1, 3>(0, 0) = res.host_normal.transpose();
      jacobian_pose.block<1, 3>(0, 3) = -res.host_normal.transpose() * Qi.toRotationMatrix() * Utility::skewSymmetric(res.vertex);
      jacobian_pose = res.sqrt_info * jacobian_pose;
    }
  }

  return true;

}
