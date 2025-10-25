// include the std cpp header files
#include <iostream>
#include <string>
#include <vector>
#include <ctime>


// include own headers
#include "SurfaceExtraction.h"

// using names
using std::cout;
using std::cin;
using std::endl;
using std::vector;
using std::string;

#define DEPTH_CAMERA_FX 322.236
#define DEPTH_CAMERA_FY 322.236
#define DEPTH_CAMERA_CX 320.448
#define DEPTH_CAMERA_CY 182

#define MAX_USAGE_DISTANCE 4   //5 之外的 都舍弃掉

__global__ void GenerateVertexMapKernel(float *depth_map, float3 *vertex_map, int height, int width, float *K, float max_depth, float min_depth)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;


    if(x >= width || y >= height)
        return;

    int id = x + y * width;

    float depth_value = depth_map[id];
    if (depth_value > max_depth || depth_value < min_depth) { 
        depth_value = 0.f; 
    }

    vertex_map[id].x = (x - K[2]) * depth_value / K[0];
    vertex_map[id].y = (y - K[5]) * depth_value / K[4];
    vertex_map[id].z = depth_value;
}

__global__ void RemoveInvalidRegionKernel(uchar3 *seg_img, uchar3 *invalid_color_vec, int length, int height, int width)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;


    if(x >= width || y >= height)
        return;

    int image_id = x + y * width;
    uchar3 oc = seg_img[image_id];
    bool flag = false;
    for(int i = 0; i < length; i++) {
        if(oc.x == invalid_color_vec[i].x && oc.y == invalid_color_vec[i].y && oc.z == invalid_color_vec[i].z) {
            seg_img[image_id] = make_uchar3(0, 0, 0);
            return;
        }
    }
}

__global__ void BuildIndexImageKernel(uchar3 *valid_color_vec, ushort *index_vec, uchar3 *seg_img, ushort *index_img, int length, int height, int width)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;


    if(x >= width || y >= height)
        return;

    int image_id = x + y * width;
    index_img[image_id] = (ushort)0;
    uchar3 oc = seg_img[image_id];
    for(int i = 0; i < length; i++) {
        if(oc.x == valid_color_vec[i].x && oc.y == valid_color_vec[i].y && oc.z == valid_color_vec[i].z) {
            index_img[image_id] = index_vec[i];
            break;
        }
    }
}



ImagePlaneFeature extractSurfaces(const cv::Mat& imgDepth, const cv::Mat& K, const Eigen::Matrix4d& Twc, float max_depth, float min_depth) 
{
    
    // float fx = K.at<float>(0, 0), fy = K.at<float>(1, 1), cx = K.at<float>(0, 2), cy = K.at<float>(1, 2);
    // // init the result vector
    // std::vector<PlaneFeature> surfaces;
    
    // cv::Mat_<cv::Vec3f> cloud(imgDepth.rows, imgDepth.cols);

    // for (int r = 0; r < imgDepth.rows; ++r){
    //     cv::Vec3f* pt_ptr = cloud.ptr<cv::Vec3f>(r);
    //     for (int c = 0; c < imgDepth.cols; ++c){
    //         float z = imgDepth.at<float>(r, c);
    //         if (z < min_depth || z > max_depth){
    //             z = 0;
    //         }
    //         pt_ptr[c][0] = (c - cx) / fx * z;
    //         pt_ptr[c][1] = (r - cy) / fy * z;
    //         pt_ptr[c][2] = z;
    //     }
    // }

    /* double minValue, maxValue;
    cv::Point minIdx, maxIdx;
    cv::minMaxLoc(imgDepth, &minValue, &maxValue, &minIdx, &maxIdx);
    printf("---------------------------------------------------------------------------\n");
    printf("***************************************************************************\n");
    std::cout << "max value of depth image: " << maxValue << std::endl;
    std::cout << "min value of depth image: " << minValue << std::endl;
    std::cout << "index of max: " << maxIdx << "index of min: " << minIdx << std::endl;
    printf("***************************************************************************\n");
    printf("---------------------------------------------------------------------------\n"); */
    double t_start, t_pre, t_vertex, t_ahc, t_encode;

    t_start = cv::getTickCount();
    int height = imgDepth.rows, width = imgDepth.cols;
    float *cam_K;
    CudaSafeCall(cudaMalloc((void **)&cam_K, sizeof(float) * 9));
    CudaSafeCall(cudaMemcpy(cam_K, K.data, sizeof(float) * 9, cudaMemcpyHostToDevice));
    
    float *depth_map;
    float3 *vertex_map;
    CudaSafeCall(cudaMalloc((void **)&depth_map, sizeof(float) * height * width));
    CudaSafeCall(cudaMalloc((void **)&vertex_map, sizeof(float3) * height * width));
    CudaSafeCall(cudaMemcpy(depth_map, imgDepth.data, sizeof(float) * height * width, cudaMemcpyHostToDevice));

    t_pre = (cv::getTickCount() - t_start) / cv::getTickFrequency();

    t_start = cv::getTickCount();
    dim3 grid_dim(1, 1, 1);
    dim3 block_dim(32, 32, 1);
    grid_dim.x = DIV_UP(width, block_dim.x);
    grid_dim.y = DIV_UP(height, block_dim.y);


    GenerateVertexMapKernel<<<grid_dim, block_dim>>>(depth_map, vertex_map, height, width, cam_K, max_depth, min_depth);
    CudaSafeCall(cudaDeviceSynchronize());

    cv::Mat_<cv::Vec3f> cloud(imgDepth.rows, imgDepth.cols);
    CudaSafeCall(cudaMemcpy(cloud.data, vertex_map, sizeof(float3) * height * width, cudaMemcpyDeviceToHost));

    t_vertex = (cv::getTickCount() - t_start) / cv::getTickFrequency();
    t_start = cv::getTickCount();

    PlaneFitter pf;
    pf.minSupport = 1500;
    pf.windowWidth = 5;       //小平面的大小
    pf.windowHeight = 5;
    pf.doRefine = true;

    cv::Mat segImg = cv::Mat(imgDepth.rows, imgDepth.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat indexImg = cv::Mat(height, width, CV_16U, cv::Scalar(0));
    OrganizedImage3d Ixyz(cloud);

    std::vector<PlaneFeature> vecPlaneFeature;
    
    pf.run(&Ixyz, vecPlaneFeature, &K, 0, &segImg);

    /* printf("---------------------------------------------------------------------------\n");
    printf("***************************************************************************\n");
    printf("size of detected plane feature vector: %ld\n", vecPlaneFeature.size());
    printf("***************************************************************************\n");
    printf("---------------------------------------------------------------------------\n"); */

    t_ahc = (cv::getTickCount() - t_start) / cv::getTickFrequency();
    t_start = cv::getTickCount();

    std::vector<cv::Vec3b> invalidColors, validColors;
    std::vector<ushort> validIndices;
    std::vector<PlaneFeature> validPlaneFeature;
    int validCnt = 1;
    for(auto iter: vecPlaneFeature) {
        if(iter.mse > 1e-4) {
            invalidColors.push_back(iter.color);
        }
        else {
            if(iter.color[0] != 0 || iter.color[1] != 0 || iter.color[2] != 0) {
                validPlaneFeature.push_back(iter);
                validIndices.push_back(validCnt++);
                validColors.push_back(iter.color);
            }
        }
    }
    /* std::cout << "******************************************************" << std::endl;
    std::cout << "size of total: " << vecPlaneFeature.size() << std::endl;
    std::cout << "size of valid: " << validPlaneFeature.size() << std::endl;
    std::cout << "******************************************************" << std::endl; */
    int invalidNum = invalidColors.size(), validNum = validColors.size();

    ImagePlaneFeature imgFeatures(validPlaneFeature);
    // ImagePlaneFeature imgFeatures(vecPlaneFeature);
    // refine segmentation image, remove regions that have large MSE
    uchar3 *d_invalid_color_vec, *d_valid_color_vec, *d_seg_img;
    ushort *d_valid_index_vec, *d_index_img;
    CudaSafeCall(cudaMalloc((void**)&d_invalid_color_vec, sizeof(uchar3) * invalidNum));
    CudaSafeCall(cudaMalloc((void**)&d_valid_color_vec, sizeof(uchar3) * validNum));
    CudaSafeCall(cudaMalloc((void**)&d_valid_index_vec, sizeof(ushort) * validNum));
    CudaSafeCall(cudaMalloc((void**)&d_seg_img, sizeof(uchar3) * height * width));
    CudaSafeCall(cudaMalloc((void**)&d_index_img, sizeof(ushort) * height * width));

    CudaSafeCall(cudaMemcpy(d_seg_img, segImg.data, sizeof(uchar3) * height * width, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_invalid_color_vec, invalidColors.data(), sizeof(uchar3) * invalidNum, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_valid_color_vec, validColors.data(), sizeof(uchar3) * validNum, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_valid_index_vec, validIndices.data(), sizeof(ushort) * validNum, cudaMemcpyHostToDevice));

    RemoveInvalidRegionKernel<<<grid_dim, block_dim>>>(d_seg_img, d_invalid_color_vec, invalidNum, height, width);
    CudaSafeCall(cudaDeviceSynchronize());
    BuildIndexImageKernel<<<grid_dim, block_dim>>>(d_valid_color_vec, d_valid_index_vec, d_seg_img, d_index_img, validNum, height, width);
    CudaSafeCall(cudaDeviceSynchronize());

    CudaSafeCall(cudaMemcpy(segImg.data, d_seg_img, sizeof(uchar3) * height * width, cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(indexImg.data, d_index_img, sizeof(ushort) * height * width, cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaDeviceSynchronize());

    imgFeatures.setSegImg(segImg);
    imgFeatures.setIndexImg(indexImg);
    imgFeatures.setColors(validColors);
    imgFeatures.setIndices(validIndices);
    imgFeatures.transformNormal(Twc);
    t_encode = (cv::getTickCount() - t_start) / cv::getTickFrequency();

    CudaSafeCall(cudaFree(cam_K));
    CudaSafeCall(cudaFree(depth_map));
    CudaSafeCall(cudaFree(vertex_map));
    CudaSafeCall(cudaFree(d_invalid_color_vec));
    CudaSafeCall(cudaFree(d_valid_color_vec));
    CudaSafeCall(cudaFree(d_valid_index_vec));
    CudaSafeCall(cudaFree(d_seg_img));
    CudaSafeCall(cudaFree(d_index_img));

    // printf("[extractSurfaces] t_pre: %lf, t_vertex: %lf, t_ahc: %lf, t_encode: %lf\n", t_pre, t_vertex, t_ahc, t_encode);

    return imgFeatures;
}

void printVector(std::vector<std::vector<float>> vecs)
{
    for (size_t i = 0; i < vecs.size(); ++i){
        for (size_t j = 0; j < vecs[i].size(); ++j){
            cout << vecs[i][j] << ", ";
        }
        cout << endl;
    }
}

void printVectorPair(std::vector<std::pair<float, float>> vecPairs)
{
    cout << "Pairs: " << endl;
    for (size_t i = 0; i < vecPairs.size(); ++i){
        cout << "(" << vecPairs[i].first << ", " << vecPairs[i].second << ")" << endl;
    }
}
