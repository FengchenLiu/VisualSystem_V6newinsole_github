/*
Created by shunyi, 20, 03, 2023
email: zhaoshunyi@bigai.ai
*/
#ifndef _SCENE_GRAPH_TRAJECTORY_TRACKER_
#define _SCENE_GRAPH_TRAJECTORY_TRACKER_
#include <iostream>
#include <cstdio>
#include <vector>
#include <array>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace gait_divider {

// define the shorter names for gait divider
typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;

typedef Eigen::Matrix<double, 11, 1> Vector11d;

class Kinematic_State {
private:
    Vector11d kine_State;
    Eigen::Vector3d position;
    Eigen::Vector4d pose;
    Eigen::Vector3d velocity;

protected:

public:
    Kinematic_State();
    Kinematic_State(std::array<double, 11> estimatedState);
    Kinematic_State(Vector11d estimatedState);
    ~Kinematic_State();
};

// the class of this Kalman Filter is not defined as a template class
class KalmanFilter {
private:
    Vector6d x_mean; // the positions and velocities <x y z vx vy vz>
    Matrix6d P; // the P matrix in Kalman Filter
    Matrix6d Q; // the Q matrix in Kalman Filter
    Vector6d B; // the B matrix in Kalman Filter
    Matrix6d H; // the H matrix in Kalman Filter

    unsigned long numReceived;

    // function to generate inti states and matrices
    Matrix6d gnrtInitPMat(double initSigma);
    Matrix6d gnrtInitHMat();
    Matrix6d gnrtInitQMat();

protected:
    // protected functions
public:
    // public methods and attributes for this class

    // constructor for this kalman filter
    KalmanFilter();
    KalmanFilter(Vector6d firstLine);
    KalmanFilter(Vector6d firstLine, double initSigma);

    // de-constructor
    ~KalmanFilter();

    // getter
    unsigned long getNumReceived();

    // printer
    void printCurrentState();
    void printNumReceived();
    void printRecordedTimestamp();
    void printAccelerations();
    void printXMean();

    // methods
    Matrix6d gnrtFMat(double deltaT);
    Eigen::Vector2d gnrt2DimensionB(double deltaT);


};
};
#endif
