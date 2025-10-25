#include "projection_xyz_factor.h"

Eigen::Matrix2d ProjectionXYZFactor::sqrt_info;
double ProjectionXYZFactor::sum_t;

bool ProjectionXYZFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
  Eigen::Vector3d tcw(parameters[0][0], parameters[0][1], parameters[0][2]);
  Sophus::SO3d Rcw = Sophus::SO3d::exp(Eigen::Vector3d(parameters[0][3], parameters[0][4], parameters[0][5]));
  Eigen::Vector3d Pw(parameters[1][0], parameters[1][1], parameters[1][2]);

  Eigen::Vector3d Pc = Rcw * Pw + tcw;
  double x = Pc.x(), y = Pc.y(), z = Pc.z(), z2 = z*z;
  Eigen::Map<Eigen::Vector2d> residual(residuals);
  residual = sqrt_info * (Pc.head<2>() / z - Pc_m);

  Eigen::Matrix<double, 2, 3, Eigen::RowMajor> duv_dpc;
  duv_dpc << 1/z,   0, -x/z2, 
                0, 1/z, -y/z2;

  if(jacobians) {
    if(jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
      jacobian_pose.setZero();
      Eigen::Matrix<double, 3, 6, Eigen::RowMajor> dpc_dpw;
      dpc_dpw.leftCols<3>().setIdentity();
      dpc_dpw.rightCols<3>() = -Utility::skewSymmetric(Rcw * Pw);
      jacobian_pose = sqrt_info * duv_dpc * dpc_dpw;
    }
    if(jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian_point(jacobians[1]);
      jacobian_point = sqrt_info * duv_dpc * Rcw.matrix();
    }
  }

  return true;

}

void ProjectionXYZFactor::Check(double **parameters)
{
  Eigen::Vector3d tcw(parameters[0][0], parameters[0][1], parameters[0][2]);
  Sophus::SO3d Rcw = Sophus::SO3d::exp(Eigen::Vector3d(parameters[0][3], parameters[0][4], parameters[0][5]));
  Eigen::Vector3d Pw(parameters[1][0], parameters[1][1], parameters[1][2]);

  Eigen::Vector3d Pc = Rcw * Pw + tcw;
  double x = Pc.x(), y = Pc.y(), z = Pc.z(), z2 = z*z;
  Eigen::Vector2d residual = (Pc.head<2>() / z - Pc_m);

  Eigen::Matrix<double, 2, 3> duv_dpc;
  duv_dpc << 1/z,   0, -x/z2, 
                0, 1/z, -y/z2;

  Eigen::Matrix<double, 2, 6> jacobian_pose;
  jacobian_pose.setZero();
  Eigen::Matrix<double, 3, 6> dpc_dpw;
  dpc_dpw.leftCols<3>().setIdentity();
  dpc_dpw.rightCols<3>() = -Utility::skewSymmetric(Rcw * Pw);
  jacobian_pose = duv_dpc * dpc_dpw;

  Eigen::Matrix<double, 2, 3, Eigen::RowMajor> jacobian_point;
  jacobian_point =  duv_dpc * Rcw.matrix();

  const double norm = 1e-3;

  //turb pose
  {
    Eigen::Vector3d tcw_noise = Eigen::Vector3d::Random() * norm;
    Eigen::Vector3d rcw_noise = Eigen::Vector3d::Random() * norm;
    Eigen::Vector3d tcw_trubed = tcw + tcw_noise;
    Sophus::SO3d Rcw_turbed = Sophus::SO3d::exp(rcw_noise) * Rcw;
    Eigen::Vector3d Pc_turbed = Rcw_turbed * Pw + tcw_trubed;
    Pc_turbed /= Pc_turbed.z();
    Eigen::Vector2d residual_turbed = (Pc_turbed.head<2>() - Pc_m);
    Eigen::Matrix<double, 6, 1> state_diff;
    state_diff << tcw_noise, rcw_noise;
    std::cout << "residual diff: " << std::endl << (residual_turbed - residual).transpose() << std::endl;
    std::cout << "residual linearized diff: " << std::endl << (jacobian_pose * state_diff).transpose() << std::endl;
    std::cout << "jacobian: " << std::endl << jacobian_pose << std::endl;
    std::cout << "state diff: " << std::endl << state_diff.transpose() << std::endl;
  }
  std::cout << "\n";
  //turb point
  {
    Eigen::Vector3d point_noise = Eigen::Vector3d::Random() * norm;
    Eigen::Vector3d Pw_trubed = Pw + point_noise;
    Eigen::Vector3d Pc_turbed = Rcw * Pw_trubed + tcw;
    Pc_turbed /= Pc_turbed.z();
    Eigen::Vector2d residual_turbed = (Pc_turbed.head<2>() - Pc_m);
    Eigen::Matrix<double, 3, 1> state_diff;
    state_diff = point_noise;
    std::cout << "residual diff: " << std::endl << (residual_turbed - residual).transpose() << std::endl;
    std::cout << "residual linearized diff: " << std::endl << (jacobian_point * state_diff).transpose() << std::endl;
    std::cout << "jacobian: " << std::endl << jacobian_point << std::endl;
    std::cout << "state diff: " << std::endl << state_diff.transpose() << std::endl;
  }

  
}