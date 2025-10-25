#ifndef PLANEFEATUREFLAG
#define PLANEFEATUREFLAG
#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <unordered_map>
#include <utility>

// include opencv header files
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <eigen3/Eigen/Core>

// define the Magic number in this programme
#define NORMAL_THRESHOLD 0.1

class PlaneFeature{
public:
    Eigen::Vector3d center, normal, global_center, global_normal;
    cv::Vec3b color;
    double mse;

    PlaneFeature(){};
    
    PlaneFeature(const Eigen::Vector3d &centerIn, Eigen::Vector3d normalIn, double _mse, const cv::Vec3b _color){
        center = centerIn;
        normal = normalIn;
        mse = _mse;
        color = cv::Vec3b(_color[0], _color[1], _color[2]);
    };

    void transformNormal(const Eigen::Matrix4d& Twc) {
        global_center = Twc.block<3, 3>(0, 0) * center + Twc.block<3, 1>(0, 3);
        global_normal = Twc.block<3, 3>(0, 0) * normal;
    }

    // void print(){
    //     std::cout << "(";
    //     for (std::vector<float>::iterator iter = center.begin(); iter != center.end(); ++iter){
    //         std::cout << *iter << ", ";
    //     }
    //     std::cout << ")" << std::endl;

    //     std::cout << "[";
    //     for (std::vector<float>::iterator iter = normal.begin(); iter != normal.end(); ++iter){
    //         std::cout << *iter << ", ";
    //     }
    //     std::cout << "]" << std::endl;
    //     std::cout << std::endl;
    // };

    Eigen::Vector3d getCenterVec() {
        return center;
    }

    Eigen::Vector3d getNormalVec() {
        return normal;
    }
};

class ImagePlaneFeature {
private:
    std::vector<PlaneFeature> vecImageFeatures;
    std::unordered_map<uint, uint> colorHashMap;

    std::vector<cv::Vec3b> colors;
    std::vector<ushort> indices;
    std::vector<float> targetNormal;
    std::vector<std::pair<float, float>> vecPairedHeightAndDepth;
    std::vector<std::vector<float>> vecRotatedCenters;
    cv::Mat segedImage, vertexImage, normalImage, indexImage;

public:
    ImagePlaneFeature() {}

    ImagePlaneFeature(std::vector<PlaneFeature> initVecPlaneFeatures){
        vecImageFeatures.assign(initVecPlaneFeatures.begin(), initVecPlaneFeatures.end());
    };

    void push(PlaneFeature feature){
        vecImageFeatures.push_back(feature);
    }

    void transformNormal(const Eigen::Matrix4d& Twc) {
        for(auto &p: vecImageFeatures) {
            p.transformNormal(Twc);
        }
    }

    // void print(){
    //     //std::cout << vecImageFeatures.size() << std::endl;
    //     for (auto iter = (this -> vecImageFeatures).begin(); iter != (this -> vecImageFeatures).end(); ++iter){
    //         std::vector<float> rawNormalVec = (*iter).getNormalVec();
    //         if (rawNormalVec[1] > 0){
    //             continue;
    //         }
    //         (*iter).print();

    //         std::vector<float> rawCenterVec = (*iter).getCenterVec();
    //     }
    // }

    // void printRawCenter(){
    //     for (auto iter = (this -> vecImageFeatures).begin(); iter != (this -> vecImageFeatures).end(); ++iter){
    //         (*iter).print();
    //     }
    // }

    // void printRawCenterAngleRatio(float angleDegree){
    //     for (auto iter = (this -> vecImageFeatures).begin(); iter != (this -> vecImageFeatures).end(); ++iter){
    //         std::cout << "**********************************" << std::endl;
    //         std::vector<float> rawNormalVec = (*iter).getNormalVec();
    //         if (rawNormalVec[1] > 0){
    //             continue;
    //         }
    //         (*iter).print();

    //         std::vector<float> center = (*iter).getCenterVec();
    //         float length = sqrt(center[1] * center[1] + center[2] * center[2]);
    //         /*cal angle degree*/
    //         float angleDegree2 = atan2(center[1], center[2]);
    //         //angleDegree2 = 0;
    //         std::cout << "angle degree 2: " << angleDegree2 << std::endl;
    //         std::cout << "sum of angles: " << (angleDegree + angleDegree2) << std::endl;

    //         std::cout << "length: " << length << std::endl; 
    //         std::cout << "height: " << length * sin(angleDegree + angleDegree2) << std::endl;
    //         std::cout << "depth: " << length * cos(angleDegree + angleDegree2) << std::endl;
    //         std::vector<float> rotatedCenter = rotateCenterPointReturn(rawNormalVec, targetNormal, center);
    //         std::cout << "rotated center: " << std::endl;
    //         for (size_t i = 0; i < rotatedCenter.size(); ++i){
    //             std::cout << rotatedCenter[i] << " ";
    //         }
    //         std::cout << std::endl;
    //         std::cout << "**********************************" << std::endl;
    //     }
    // }

    // // setVecPairedHeightAndDepth
    // void setVecPairedHeightAndDepth(std::vector<std::pair<float, float>> vecInput){
    //     this -> vecPairedHeightAndDepth = vecInput;
    // }

    // std::vector<std::pair<float, float>> getVecPairedHeightAndDepth(){
    //     return this -> vecPairedHeightAndDepth;
    // }

    // std::vector<std::vector<float>> getVecRotatedCenters(){
    //     return this -> vecRotatedCenters;
    // }

    // // calculate the height and depth, save to a std pair vector
    // std::vector<std::pair<float, float>> calPairedRotatedHeightDepth(float angleDegree){
    //     // declare the return pair vector
    //     std::vector<std::pair<float, float>> vecPairedHeightDepth;

    //     float targetYNormalVal = cos(angleDegree) * (-1);
    //     std::cout << "target second normal value: " << targetYNormalVal << std::endl;

    //     // for loop to filter and save results
    //     for (auto iter = (this -> vecImageFeatures).begin(); iter != (this -> vecImageFeatures).end(); ++iter){
    //         std::vector<float> rawNormalVec = (*iter).getNormalVec();
    //         if (abs(targetYNormalVal - rawNormalVec[1]) > 0.2){
    //             continue;
    //         }
    //         //std::cout << "**********************************" << std::endl;
    //         //(*iter).print();

    //         std::vector<float> center = (*iter).getCenterVec();
    //         float length = sqrt(center[1] * center[1] + center[2] * center[2]);
    //         float angleDegree2 = atan2(center[1], center[2]);

    //         float height = length * sin(angleDegree + angleDegree2);
    //         float depth = length * cos(angleDegree + angleDegree2);

    //         // insert or push_back to the result

    //         if (vecPairedHeightDepth.size() == 0){
    //             vecPairedHeightDepth.push_back(std::pair<float, float>(depth, height));
    //         }
    //         else {
    //             for (auto resIter = vecPairedHeightDepth.begin(); resIter != vecPairedHeightDepth.end(); ++resIter){
    //                 if (depth < (*resIter).first) {
    //                     vecPairedHeightDepth.insert(resIter, std::pair<float, float>(depth, height));
    //                     break;
    //                 }
    //                 if (resIter == vecPairedHeightDepth.end() - 1){
    //                     vecPairedHeightDepth.push_back(std::pair<float, float>(depth, height));
    //                     break;
    //                 }
    //             }
    //         }
    //         //std::cout << "**********************************" << std::endl;
    //     }

    //     return vecPairedHeightDepth;
    // }

    // // calculate the normal value of the horizontal plane
    // float calYAxisNormalVal(float angleDegree){
    //     return -1 * cos(angleDegree);
    // }

    // // calculate the 
    // std::vector<std::vector<float>> calRotatedCenterPoint(float angleDegree){
    //     std::vector<std::vector<float>> rotatedCenters;
    //     std::cout << "Raw Size:" << vecImageFeatures.size() << std::endl;

    //     float targetYNormalVal = calYAxisNormalVal(angleDegree);
    //     for (auto iter = vecImageFeatures.begin(); iter != vecImageFeatures.end(); ++iter){
            
    //         // jump to next step if this plane is not vec
    //         std::vector<float> rawNormalVec = (*iter).getNormalVec();
    //         if (abs(rawNormalVec[1] - targetYNormalVal) > 0.1){
    //             continue;
    //         }

    //         std::vector<float> currRawCenterVec = (*iter).getCenterVec();
            
    //         currRawCenterVec = rotateCenterPointReturn(rawNormalVec, targetNormal, currRawCenterVec);

    //         if (rotatedCenters.size() == 0){
    //             rotatedCenters.push_back(currRawCenterVec);
    //             continue;
    //         }

    //         bool insertFlag = false;
    //         for (auto iter2 = rotatedCenters.begin(); iter2 != rotatedCenters.end(); ++iter2){
    //             if (currRawCenterVec[2] < (*iter2)[2]){
    //                 rotatedCenters.insert(iter2, currRawCenterVec);
    //                 insertFlag = true;
    //                 break;
    //             }
    //         }
            
    //         if (!insertFlag){
    //             rotatedCenters.push_back(currRawCenterVec);
    //         }
    //     }

    //     vecRotatedCenters = rotatedCenters;
    //     return rotatedCenters;
    // }

    // int size(){
    //     return vecImageFeatures.size();
    // }

    // std::vector<float> rotateCenterPointReturn(std::vector<float> normal1, std::vector<float> normal2, std::vector<float> rawCoord){
    //     Eigen::Vector3d eigenNormal1(normal1[0], normal1[1], normal1[2]);
    //     Eigen::Vector3d eigenNormal2(normal2[0], normal2[1], normal2[2]);
    //     Eigen::Matrix3d rotMatrix = Eigen::Quaterniond::FromTwoVectors(eigenNormal1, eigenNormal2).toRotationMatrix();

    //     Eigen::Vector3d eigenRawCoord(rawCoord[0], rawCoord[1], rawCoord[2]);
    //     Eigen::Vector3d result = rotMatrix * eigenRawCoord;

    //     std::vector<float> rotatedResult;
    //     rotatedResult.push_back(result[0]);
    //     rotatedResult.push_back(result[1]);
    //     rotatedResult.push_back(result[2]);

    //     return rotatedResult;
    // };

    void setColorHashMap(const std::vector<cv::Vec3b>& colors)
    {
        for(uint i = 0; i < (uint)colors.size(); i++) {
            uint key = (uint)((uint)colors[i][0] << 16) + (uint)((uint)colors[i][1] << 8) + (uint)((uint)colors[i][2]);
            // printf("color: %d, %d, %d, key: %d\n", (uint)colors[i][0], (uint)colors[i][1], (uint)colors[i][2], key);

            const uint mask_b = 0x00FF0000, mask_g = 0x0000FF00, mask_r = 0x000000FF;
            uint b = (key & mask_b) >> 16;
            uint g = (key & mask_g) >> 8;
            uint r = (key & mask_r);
            assert((uint)colors[i][0] == b);
            assert((uint)colors[i][1] == g);
            assert((uint)colors[i][2] == r);

            // printf("recover color: %d, %d, %d\n", b, g, r);
            colorHashMap.insert(std::make_pair(key, i));
        }
    }

    void fillNormalImage() {
        if(segedImage.empty()) {
            printf("[fillNormalImage] Error! segedImage is empty!\n");
            return;
        }
        int rows = segedImage.rows, cols = segedImage.cols;
        normalImage = cv::Mat(rows, cols, CV_32FC3);
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < cols; j++) {
                cv::Vec3b color = segedImage.at<cv::Vec3b>(i, j);
                uint key = (uint)((uint)color[0] << 16) + (uint)((uint)color[1] << 8) + (uint)((uint)color[2]);
                auto iter = colorHashMap.find(key);
                if(iter == colorHashMap.end()) {
                    normalImage.at<cv::Vec3f>(i, j) = cv::Vec3f(0, 0, 0);
                }
                else {
                    int index = iter->second;
                    Eigen::Vector3d normal = vecImageFeatures[index].global_normal; normal.normalize();
                    normalImage.at<cv::Vec3f>(i, j) = cv::Vec3f(normal.x(), normal.y(), normal.z());
                }
            }
        }
    }

    void setSegImg(const cv::Mat& inputMat) {
        this->segedImage = inputMat;
    };
    
    void setIndexImg(const cv::Mat& inputMat) {
        this->indexImage = inputMat;
    }

    void setVertexImg(const cv::Mat& inputMat) {
        this->vertexImage = inputMat;
    }

    void setColors(const std::vector<cv::Vec3b>& inputVec) {
        colors = inputVec;
    }

    void setIndices(const std::vector<ushort>& inputVec) {
        indices = inputVec;
    }

    std::unordered_map<uint, uint> getColorHashMap() {
        return this -> colorHashMap;
    }

    std::vector<PlaneFeature> getFeatureVec() {
        return this -> vecImageFeatures;
    }

    cv::Mat getSegImg() {
        return this -> segedImage;
    }

    cv::Mat getVertexImg() {
        return this -> vertexImage;
    }

    cv::Mat getNormalImg() {
        return this -> normalImage;
    }

    std::vector<cv::Vec3b> getColors() {
        return this->colors;
    }

    cv::Mat getIndexImg() {
        return this->indexImage;
    }
};
#endif
