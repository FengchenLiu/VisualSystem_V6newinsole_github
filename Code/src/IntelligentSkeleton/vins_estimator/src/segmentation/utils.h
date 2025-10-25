#ifndef _UTILS_H_
#define _UTILS_H_
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <cmath>

// init the matrix and bias
class CameraIntrinsics{
private:
    float cameraFx;
    float cameraFy;
    float cameraCx;
    float cameraCy;
public:
    CameraIntrinsics(float cameraFxIn, float cameraFyIn, float cameraCxIn, float cameraCyIn);

    // get methods
    float getCameraFx();
    float getCameraFy();
    float getCameraCx();
    float getCameraCy();
};

const std::vector<std::vector<float>> matrixAccel{
    {1.01794,     0.0187408794, -0.031033956   },
    {0.0,         1.0181,        0.000672895426},
    {0.0,         0.0,           1.00656       }
};

const std::vector<std::vector<float>> matrixGyro{
    { 0.988744,    0.02267488,   -0.02087184   },
    {-0.00478232,  0.984033,     -0.01133844   },
    {0.02845437,  -0.0032004,     0.996707     }
};

const std::vector<float> vecBiasAccel{0.116397,  0.791815,    0.346381};
const std::vector<float> vecBiasGyro{0.00041101, 0.00043892, -0.00101735};

std::vector<float> matmulMatrix3dVector3(std::vector<std::vector<float>> matrix3d, std::vector<float> vector3d);
std::vector<float> reduceVector3(std::vector<float> vec1, std::vector<float> vec2);
std::vector<float> calibVector3(std::vector<float> vecValue, std::vector<float> vecBias, std::vector<std::vector<float>> matrix3d);
std::vector<std::string> compressSpaceString(std::string strInputted);
template <class T>
T cvtStringToNum(std::string str);
float calCurrPitchAccel(std::vector<float> vecAccel);
float calCurrPitchGyro(std::vector<float> vecGyro, float lastPitch, float deltaT, float lastGyroPitch);
std::vector<unsigned char> encodeFloatNumber(std::string strNumber);
#endif