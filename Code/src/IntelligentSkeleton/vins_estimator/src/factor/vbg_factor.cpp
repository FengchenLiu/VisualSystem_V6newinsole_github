#include "vbg_factor.h"


bool VBGdirFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
  Eigen::Vector3d Vi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Vector3d Bai(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Vector3d Bgi(parameters[2][0], parameters[2][1], parameters[2][2]);

  Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
  Eigen::Vector3d Baj(parameters[4][0], parameters[4][1], parameters[4][2]);
  Eigen::Vector3d Bgj(parameters[5][0], parameters[5][1], parameters[5][2]);

  double roll = parameters[6][0], pitch = parameters[6][1];
  double sinr = sin(roll), cosr = cos(roll), sinp = sin(pitch), cosp = cos(pitch);

  double dt = integration_ij->sum_dt;
  Eigen::Quaterniond Qi(Ri), Qj(Rj);
  Eigen::Vector3d Gf(-sinp * GRAVITY, sinr * cosp * GRAVITY, cosr * cosp * GRAVITY);

  Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
  Eigen::Matrix3d dp_dba = integration_ij->jacobian.template block<3, 3>(O_P, O_BA);
  Eigen::Matrix3d dp_dbg = integration_ij->jacobian.template block<3, 3>(O_P, O_BG);
  Eigen::Matrix3d dr_dbg = integration_ij->jacobian.template block<3, 3>(O_R, O_BG);
  Eigen::Matrix3d dv_dba = integration_ij->jacobian.template block<3, 3>(O_V, O_BA);
  Eigen::Matrix3d dv_dbg = integration_ij->jacobian.template block<3, 3>(O_V, O_BG);

  const Eigen::Quaterniond dQ = integration_ij->getDeltaRotation(Bgi);
  const Eigen::Vector3d dV = integration_ij->getDeltaVelocity(Bai, Bgi);
  const Eigen::Vector3d dP = integration_ij->getDeltaPosition(Bai, Bgi);
  const Sophus::SO3d eR = Sophus::SO3d(dQ.inverse().toRotationMatrix() * (Ri.transpose() * Rj));
  const Eigen::Vector3d er = eR.log();
  const Eigen::Vector3d ev = Ri.transpose() * (Vj - Vi + Gf * dt) - dV;
  const Eigen::Vector3d ep = Ri.transpose() * (Pj - Pi - Vi * dt + 0.5 * Gf * dt * dt) - dP;

  const Eigen::Vector3d eBa = Baj - Bai;
  const Eigen::Vector3d eBg = Bgj - Bgi;

  residual << ep, er, ev, eBa, eBg;
  Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(integration_ij->covariance.inverse()).matrixL().transpose();
  residual = sqrt_info * residual;


  if(jacobians) {
    if(jacobians[0]) {
      Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_speed_i(jacobians[0]);
      jacobian_speed_i.setZero();
      jacobian_speed_i.block<3, 3>(O_P, 0) = -Ri.transpose() * dt;
      jacobian_speed_i.block<3, 3>(O_V, 0) = -Ri.transpose();
      jacobian_speed_i = sqrt_info * jacobian_speed_i;
    }
    if(jacobians[1]) {
      Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_ba_i(jacobians[1]);
      jacobian_ba_i.setZero();
      jacobian_ba_i.block<3, 3>(O_P, 0) = -dp_dba;
      jacobian_ba_i.block<3, 3>(O_V, 0) = -dv_dba;
      jacobian_ba_i.block<3, 3>(O_BA, 0) = -Eigen::Matrix3d::Identity();
      jacobian_ba_i = sqrt_info * jacobian_ba_i;
    }
    if(jacobians[2]) {
      Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_bg_i(jacobians[2]);
      jacobian_bg_i.setZero();
      jacobian_bg_i.block<3, 3>(O_P, 0) = -dp_dbg;
      jacobian_bg_i.block<3, 3>(O_V, 0) = -dv_dbg;
      jacobian_bg_i.block<3, 3>(O_R, 0) = -Utility::Qleft(Qj.inverse() * Qi * integration_ij->delta_q).bottomRightCorner<3, 3>() * dr_dbg;

      jacobian_bg_i.block<3, 3>(O_BG, 0) = -Eigen::Matrix3d::Identity();
      jacobian_bg_i = sqrt_info * jacobian_bg_i;
    }
    if(jacobians[3]) {
      Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_speed_j(jacobians[3]);
      jacobian_speed_j.setZero();
      jacobian_speed_j.block<3, 3>(O_V, 0) = Ri.transpose();
      jacobian_speed_j = sqrt_info * jacobian_speed_j;
    }
    if(jacobians[4]) {
      Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_ba_j(jacobians[4]);
      jacobian_ba_j.setZero();
      jacobian_ba_j.block<3, 3>(O_BA, 0) = Eigen::Matrix3d::Identity();
      jacobian_ba_j = sqrt_info * jacobian_ba_j;
    }
    if(jacobians[5]) {
      Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> jacobian_bg_j(jacobians[5]);
      jacobian_bg_j.setZero();
      jacobian_bg_j.block<3, 3>(O_BG, 0) = Eigen::Matrix3d::Identity();
      jacobian_bg_j = sqrt_info * jacobian_bg_j;
    }
    if(jacobians[6]) {
      Eigen::Map<Eigen::Matrix<double, 15, 2, Eigen::RowMajor>> jacobian_euler(jacobians[6]);
      Eigen::Matrix<double, 15, 3, Eigen::RowMajor> dr_dGf;
      Eigen::Matrix<double, 3, 2, Eigen::RowMajor> dGf_deuler;
      dr_dGf.setZero(); dGf_deuler.setZero();
      dr_dGf.block<3, 3>(O_P, 0) = 0.5 * Ri.transpose() * dt * dt;
      dr_dGf.block<3, 3>(O_V, 0) = Ri.transpose() * dt;

      dGf_deuler <<         0.0,        -cosp,
                    cosr * cosp, sinr * -sinp,
                   -sinr * cosp, cosr * -sinp;

      jacobian_euler = dr_dGf * dGf_deuler * GRAVITY;
      jacobian_euler = sqrt_info * jacobian_euler;
    }
  }
  return true;

}