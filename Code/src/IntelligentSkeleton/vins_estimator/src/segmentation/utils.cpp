#include "utils.h"

using std::cout;
using std::cerr;
using std::endl;

std::vector<float> matmulMatrix3dVector3(std::vector<std::vector<float>> matrix3d, std::vector<float> vector3d){
    std::vector<float> resultVector;
    for (int i = 0; i < 3; ++i){
        float currResult = matrix3d[i][0] * vector3d[0] + matrix3d[i][1] * vector3d[1] + matrix3d[i][2] * vector3d[2];
        resultVector.push_back(currResult);
    }
    return resultVector;
};

std::vector<float> reduceVector3(std::vector<float> vec1, std::vector<float> vec2){
    return std::vector<float>{vec1[0] - vec2[0], vec1[1] - vec2[1], vec1[2] - vec2[2]};
}

std::vector<float> calibVector3(std::vector<float> vecValue, std::vector<float> vecBias, std::vector<std::vector<float>> matrix3d){
    vecValue = reduceVector3(vecValue, vecBias);
    vecValue = matmulMatrix3dVector3(matrix3d, vecValue);

    return vecValue;
}

std::vector<std::string> compressSpaceString(std::string strInputted){
    int headIndex = 0, endIndex = strInputted.size();
    while (strInputted[headIndex] == ' '){
        ++headIndex;
    }
    std::string str = strInputted.substr(headIndex, endIndex);
    std::vector<std::string> vecResults;
    int leftIndex = 0, rightIndex = 0;
    for (size_t i = 0; i < str.size(); ++i){
        if (str[i] == ' '){
            continue;
        }
        else{
            leftIndex = i;
            rightIndex = i;
            while(str[rightIndex] != ' ' && rightIndex != (int)str.size()){
                ++rightIndex;
            }
            vecResults.push_back(str.substr(leftIndex, rightIndex - leftIndex));
            i = rightIndex;
        }
    }
    return vecResults;
};

template <class T>
T cvtStringToNum(std::string str){
    std::istringstream iss(str);
    T num;
    iss >> num;
    return num;
};

float calCurrPitchAccel(std::vector<float> vecAccel){
    return atan(vecAccel[2] / vecAccel[1]);
}

float calCurrPitchGyro(std::vector<float> vecGyro, float lastPitch, float deltaT, float lastGyroPitch){
    
    // cal the curr pitch with gyro
    return (-vecGyro[0] + lastGyroPitch) / 2 * deltaT + lastPitch;
}

std::vector<unsigned char> encodeFloatNumber(std::string strNumber){
    std::vector<unsigned char> vecResult;
    int pointPostion = strNumber.find(".");
    cout << "encode fucntion, number string: " << strNumber << endl;
    //cout << "encode function, point position: " << pointPostion << endl;
    std::string strNumberBeforePoint = strNumber.substr(0, pointPostion);
    std::string strNumberAfterPoint = strNumber.substr(pointPostion + 1, 1);

    cout << "encode function, number before point: " << strNumberBeforePoint << endl;
    cout << "encode function, number after point: " << strNumberAfterPoint << endl;

    bool flagPositive = true;
    if (strNumberBeforePoint[0] == '-'){
        flagPositive = false;
        strNumberBeforePoint = strNumberBeforePoint.substr(1, strNumberBeforePoint.size() - 1);
    }
    else{
        flagPositive = true;
    }

    std::istringstream sstrmBefore(strNumberBeforePoint), sstrmAfter(strNumberAfterPoint);
    unsigned int nBeforePoint, nAfterPoint;
    sstrmBefore >> nBeforePoint, sstrmAfter >> nAfterPoint;
    cout << "encode function, number before point(unsigned int): " << nBeforePoint << endl;
    cout << "encode function, number after point(unsigned int): " << nAfterPoint << endl;
    unsigned char ucAfterPoint = nAfterPoint;
    //cout << "encode function, number after point(unsigned char): " << (int)ucAfterPoint << endl;

    unsigned char lowFourBitMask = 0b00001111;
    unsigned char result1 = nBeforePoint & lowFourBitMask;
    //cout << "lower 6 bit in before number: " << (int)result1 << endl;
    result1 = result1 << 4;
    result1 += ucAfterPoint;
    cout << "result1: " << (int)result1 << endl;

    unsigned int highFiveMask = 0b111110000;
    unsigned int result2FiveBit = nBeforePoint & highFiveMask;
    unsigned char result2 = result2FiveBit >> 4;
    if (!flagPositive){
        result2 = result2 | 0b10000000;
    }
    cout << "result2: " << (int) result2 << endl;

    vecResult.push_back(result2);
    vecResult.push_back(result1);

    return vecResult;
}

CameraIntrinsics::CameraIntrinsics(float cameraFxIn, float cameraFyIn, float cameraCxIn, float cameraCyIn){
    this -> cameraFx = cameraFxIn;
    this -> cameraFy = cameraFyIn;
    this -> cameraCx = cameraCxIn;
    this -> cameraCy = cameraCyIn;
}

float CameraIntrinsics::getCameraFx(){
    return cameraFx;
}

float CameraIntrinsics::getCameraFy(){
    return cameraFy;
}

float CameraIntrinsics::getCameraCx(){
    return cameraCx;
}

float CameraIntrinsics::getCameraCy(){
    return cameraCy;
}

// declare functions
template float cvtStringToNum<float>(std::string str);