#ifndef SURFACEEXTRACTIONFLAG
#define SURFACEEXTRACTIONFLAG

// include the std cpp header files
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

// include the AHC Plane Fitter header files
#include "AHC/AHCPlaneFitter.hpp"
#include "utils.h"

#define CudaSafeCall(ans) { gpuCheck((ans), __FILE__, __LINE__); }

#define DIV_UP(n,div) (((n) + (div) - 1) / (div))

inline void gpuCheck(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


class OrganizedImage3d {
public:
	const cv::Mat_<cv::Vec3f>& cloud;

    // constructors
	OrganizedImage3d(const cv::Mat_<cv::Vec3f>& c) : cloud(c) {}; //
	inline int width() const { return cloud.cols; };
	inline int height() const { return cloud.rows; };
	inline bool get(const int row, const int col, double& x, double& y, double& z) const {
		const cv::Vec3f& p = cloud.at<cv::Vec3f>(row, col);
		x = p[0];
		y = p[1];
		z = p[2];

		return z > 0 && isnan(z) == 0; // return false fi current depth is NaN
	};
};

typedef ahc::PlaneFitter<OrganizedImage3d> PlaneFitter;

ImagePlaneFeature extractSurfaces(const cv::Mat& imgDepth, const cv::Mat& K, const Eigen::Matrix4d& Twc, float max_depth, float min_depth) ;
std::vector<float> rotatedCenterPoint(std::vector<float> normal1, std::vector<float> normal2, std::vector<float> rawCoord);
void printVectorPair(std::vector<std::pair<float, float>> vecPairs);
void printVector(std::vector<std::vector<float>> vecs);
#endif
