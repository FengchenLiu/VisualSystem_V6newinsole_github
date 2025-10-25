#include "tsdf_mapper.h"

ushort TSDFMapper::factory_plane_id = 1;

// Eigen::Vector3d PlaneInfo::GetNormal() {
//   Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(cov_ / N);
//   return saes.eigenvectors().col(0);
// }


TSDFMapper::TSDFMapper(const TSDFMappingOptions& options)
{
  gpuErrchk(cudaMallocManaged(&volume, sizeof(Volume)));
  assert(options.K.type() == CV_32F);

  volume->Init(options.height, 
               options.width, 
               options.grid_dim_x, 
               options.grid_dim_y, 
               options.grid_dim_z, 
               options.voxel_size, 
               options.max_depth,
               options.min_depth,
               (float*)options.K.data);

  surface_flag = false;
  Twc_.setIdentity();

  surface.reset(new ColorCloud());
  vertex.reset(new ColorCloud());
  virtual_vertex.reset(new ColorCloud());
  
  color_table_.insert(std::make_pair(0, cv::Vec3b(0,0,0)));
  match_num_table_.insert(std::make_pair(0, 0));
  srand((unsigned int)time(0));
	for(unsigned short i=1; i<10000; ++i) {
		cv::Vec3b color;
    while(color[0]<40 && color[1]<40 && color[2]<40) {
      color[0]=rand()%256;
      color[1]=rand()%256;
      color[2]=rand()%256;
      if(color[0]>40 || color[1]>40 || color[2]>40) {
        color_table_.insert(std::make_pair(i, color));
        match_num_table_.insert(std::make_pair(i, 0));
        break;
      }
    }

	}
}

TSDFMapper::~TSDFMapper()
{
  cudaDeviceSynchronize();
  gpuErrchk(cudaFree(&volume));
}

void TSDFMapper::Reset() 
{
  // std::unique_lock<std::mutex> lock(state_mutex);
  volume->Reset();

  // std::unique_lock<std::mutex> lock2(surface_mutex);
  surface.reset(new ColorCloud());
  vertex.reset(new ColorCloud());
  virtual_vertex.reset(new ColorCloud());
  surface_flag = false;

  //   ColorCloudPtr surface;


  // std::mutex render_mutex;
  // cv::Mat host_depth_img;
  // cv::Mat host_vertex_img;
  // cv::Mat host_normal_img;
  // cv::Mat host_index_img;
  // cv::Mat host_segment_img;
  
  // Eigen::Matrix4d Twc_;
  // std::mutex surface_mutex;
  // bool surface_flag;
}

void TSDFMapper::UpdateTSDF(const cv::Mat& color_image, const cv::Mat& depth_image, const Eigen::Matrix4d& Twc)
{
  {
    std::unique_lock<std::mutex> lock(surface_mutex);
    surface_flag = false;    
  }
  std::unique_lock<std::mutex> lock(state_mutex);
  Twc_ = Twc;
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Twcr = Twc.cast<float>();
  volume->UpdateTSDF((uchar3*)color_image.data, (float*)depth_image.data, (float*)Twcr.data());
}

void TSDFMapper::UpdateSegment(const cv::Mat& index_image, const cv::Mat& depth_image, const Eigen::Matrix4d& Twc)
{
  Twc_ = Twc;
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Twcr = Twc.cast<float>();
  volume->UpdateSegment((ushort*)index_image.data, (float*)depth_image.data, (float*)Twcr.data());
}

void TSDFMapper::MoveVolume(const Eigen::Matrix4d& Twc)
{
  std::unique_lock<std::mutex> lock(state_mutex);
  Eigen::Vector3d twc = Twc.topRightCorner(3, 1);
  int vx = roundf64(twc.x() / volume->voxel_size);
  int vy = roundf64(twc.y() / volume->voxel_size);
  int vz = roundf64(twc.z() / volume->voxel_size);

  volume->Move(vx, vy, vz);
}

void TSDFMapper::ExtractPointCloud(const std::string& file, float tsdf_thresh, float weight_thresh)
{
  // Count total number of points in point cloud
  int num_pts = 0;
  for (int i = 0; i < volume->grid_dim_x * volume->grid_dim_y * volume->grid_dim_z; i++) {
    if (std::abs(volume->voxels[i].tsdf) < tsdf_thresh && volume->voxels[i].weight > weight_thresh)
      num_pts++;    
  }


  printf("num_pts: %d\n", num_pts);
  // Create header for .ply file
  FILE *fp = fopen(file.c_str(), "w");
  fprintf(fp, "ply\n");
  fprintf(fp, "format binary_little_endian 1.0\n");
  fprintf(fp, "element vertex %d\n", num_pts);
  fprintf(fp, "property float x\n");
  fprintf(fp, "property float y\n");
  fprintf(fp, "property float z\n");
  fprintf(fp, "end_header\n");

  // Create point cloud content for ply file
  for (int i = 0; i < volume->grid_dim_x * volume->grid_dim_y * volume->grid_dim_z; i++) {
    // If TSDF value of voxel is less than some threshold, add voxel coordinates to point cloud
    if (std::abs(volume->voxels[i].tsdf) < tsdf_thresh && volume->voxels[i].weight > weight_thresh) {

      // Compute voxel indices in int for higher positive number range
      int z = floor(i / (volume->grid_dim_x * volume->grid_dim_y));
      int y = floor((i - (z * volume->grid_dim_x * volume->grid_dim_y)) / volume->grid_dim_x);
      int x = i - (z * volume->grid_dim_x * volume->grid_dim_y) - (y * volume->grid_dim_x);

      // Convert voxel indices to float, and save coordinates to ply file
      int bottom_x = volume->cur_warp_x % volume->grid_dim_x;
      int bottom_y = volume->cur_warp_y % volume->grid_dim_y;
      int bottom_z = volume->cur_warp_z % volume->grid_dim_z;

      int offset_x = volume->cur_warp_x - bottom_x;
      int offset_y = volume->cur_warp_y - bottom_y;
      int offset_z = volume->cur_warp_z - bottom_z;

      int3 voxel_position;
      voxel_position.x = (x < bottom_x) ? (x + offset_x + volume->grid_dim_x) : (x + offset_x);
      voxel_position.y = (y < bottom_y) ? (y + offset_y + volume->grid_dim_y) : (y + offset_y);
      voxel_position.z = (z < bottom_z) ? (z + offset_z + volume->grid_dim_z) : (z + offset_z);

      float pt_base_x = volume->grid_origin_x + voxel_position.x * volume->voxel_size + 0.5;
      float pt_base_y = volume->grid_origin_y + voxel_position.y * volume->voxel_size + 0.5;
      float pt_base_z = volume->grid_origin_z + voxel_position.z * volume->voxel_size + 0.5;

      fwrite(&pt_base_x, sizeof(float), 1, fp);
      fwrite(&pt_base_y, sizeof(float), 1, fp);
      fwrite(&pt_base_z, sizeof(float), 1, fp);
    }
  }
  fclose(fp);
}

void TSDFMapper::ExtractSurface()
{
  double start = cv::getTickCount();
  volume->ExtractSurface();
  printf("ExtractSurfaceImpl: %lf, SurfaceNum: %d\n", (cv::getTickCount() - start) / cv::getTickFrequency(), volume->surface_num);

  surface.reset(new ColorCloud());
  surface->reserve(volume->surface_num);
  std::map<ushort, PlaneInfo::Ptr> surfaces;
  curr_surfaces_.clear();
  for (int i = 0; i < volume->surface_num; i++) {
    ColorPoint point;
    point.x = volume->surface[i].x;
    point.y = volume->surface[i].y;
    point.z = volume->surface[i].z;
    
    // point.b = volume->surface[i].rgb.x;
    // point.g = volume->surface[i].rgb.y;
    // point.r = volume->surface[i].rgb.z;
    
    ushort index = volume->surface[i].index;
    cv::Vec3b color = color_table_[index];
    point.b = color[2];
    point.g = color[1];
    point.r = color[0];
    surface->push_back(point);

    bool exist_index = (surfaces.find(index) != surfaces.end());
    if(exist_index) {
      surfaces[index]->AddElement(Eigen::Vector3d(point.x, point.y, point.z));
    }
    else {
      PlaneInfo::Ptr plane_info = PlaneInfo::Ptr(new PlaneInfo);
      plane_info->AddElement(Eigen::Vector3d(point.x, point.y, point.z));
      plane_info->color = color_table_[index];
      surfaces.insert(std::make_pair(index, plane_info));
    }
  }
  for(auto iter: surfaces) {
    ushort id = iter.first;
    if(match_num_table_[id] > 4 && id != 0) {
      curr_surfaces_.insert(std::make_pair(id, iter.second));
    }
  }

  std::unique_lock<std::mutex> lock2(surface_mutex);
  surface_flag = true;
}

void TSDFMapper::GetGroundPos(const Eigen::Matrix4d& Twc, unsigned short& plane_id, Eigen::Vector3d& ground_pos, std::vector<Eigen::Vector3d> &normals)
{
  float gx, gy, gz, nx, ny, nz;
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Twcr = Twc.cast<float>();
  /* std::cout << "Twcr: " << std::endl;
  std::cout << Twcr << std::endl; */
  volume->GetGroundPos((float*)Twcr.data(), plane_id, gx, gy, gz, nx, ny, nz);
  ground_pos = Eigen::Vector3d(gx, gy, gz);
  normals.push_back(Eigen::Vector3d(nx, ny, nz));
}

void TSDFMapper::RenderView(const Eigen::Matrix4d& Twc)
{
  std::unique_lock<std::mutex> lock(render_mutex);

  double start = cv::getTickCount();
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Twcr = Twc.cast<float>();

  const int width = volume->width, height = volume->height;
  host_depth_img = cv::Mat(height, width, CV_32F);
  host_normal_img = cv::Mat(height, width, CV_32FC3);
  host_vertex_img = cv::Mat(height, width, CV_32FC3);
  host_index_img =  cv::Mat(height, width, CV_16U);  
  volume->RayCasting(Twcr.data(), host_depth_img, host_normal_img, host_vertex_img, host_index_img);
  printf("RayCasting: %lf\n", (cv::getTickCount() - start) / cv::getTickFrequency());    



  // memcpy((void*)host_vertex_img.data, (void*)volume->virtual_vertex_map, height*width*sizeof(float3));
  // memcpy((void*)host_depth_img.data, (void*)volume->d_virtual_depth_, height*width*sizeof(float));
  // memcpy((void*)host_normal_img.data, (void*)volume->virtual_normal_map, height*width*sizeof(float3)); 
  // memcpy((void*)host_index_img.data, (void*)volume->virtual_index_map, height*width*sizeof(ushort));

  gpuErrchk(cudaMemcpy((void*)host_vertex_img.data, (void*)volume->virtual_vertex_map, height*width*sizeof(float3), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy((void*)host_depth_img.data, (void*)volume->d_virtual_depth_, height*width*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy((void*)host_normal_img.data, (void*)volume->virtual_normal_map, height*width*sizeof(float3), cudaMemcpyDeviceToHost)); 
  gpuErrchk(cudaMemcpy((void*)host_index_img.data, (void*)volume->virtual_index_map, height*width*sizeof(ushort), cudaMemcpyDeviceToHost));


}

std::map<ushort, ushort> TSDFMapper::FindMatch(const cv::Mat& host_index_img, const cv::Mat& index_img)
{
  /* std::cout << "host index img" << std::endl
            << host_index_img << std::endl; */
  std::map<ushort, std::map<ushort, int>> match_matrix;
  for(int i = 0; i < index_img.rows; i++) {
    for(int j = 0; j < index_img.cols; j++) {
      ushort index = index_img.at<ushort>(i, j);
      ushort host_index = host_index_img.at<ushort>(i, j);
      if(index == 0) continue;

      auto iter_first = match_matrix.find(index);
      if(iter_first == match_matrix.end()) {
        std::map<ushort, int> sub_map;
        sub_map.insert(std::make_pair(host_index, 1));
        match_matrix.insert(std::make_pair(index, sub_map));
      }
      else {
        auto &sub_map = match_matrix[index];
        auto iter_second = sub_map.find(host_index);
        if(iter_second == sub_map.end()) {
          sub_map.insert(std::make_pair(host_index, 1));
        }
        else {
          sub_map[host_index]++;
        }
      }
    }
  } 

  std::map<ushort, ushort> match_pairs;

  for(auto iter_first: match_matrix) {
    int max_num = 0;
    ushort max_id = 0;
    // std::cout << "Key1: " << iter_first.first << std::endl;
    for(auto iter_second: iter_first.second) {
      if(iter_second.second > max_num) {
        max_num = iter_second.second;
        max_id = iter_second.first;
      }
      // std::cout << "   Key2: " << iter_second.first << " Num: " << iter_second.second << std::endl;
    }
    match_pairs.insert(std::make_pair(iter_first.first, max_id));
  }

  /* for (auto iter : match_pairs) {
    std::cout << "iter.first: " << iter.first << ", " << "iter.second: " << iter.second << std::endl;
  } */

  return match_pairs;
}

void TSDFMapper::AddNewSegmentImage(cv::Mat& index_img, const cv::Mat& depth_img, const Eigen::Matrix4d& Twc,
                                const std::vector<cv::Vec3b>& colors, const std::vector<Eigen::Vector3d>& centers, 
                                std::vector<Eigen::Vector3d>& normals)
{
  assert(colors.size() == centers.size());
  assert(colors.size() == normals.size());

  double start = cv::getTickCount();
  std::map<ushort, ushort> match_pairs = FindMatch(host_index_img, index_img);
  printf("[FindMatch] CostTime: %lf\n", (cv::getTickCount() - start) / cv::getTickFrequency());

  std::vector<std::pair<ushort, ushort>> match_vec = std::vector<std::pair<ushort, ushort>>(match_pairs.begin(), match_pairs.end());

  for(auto &pair: match_pairs) {
    assert(pair.first != 0);
    
    if(pair.second == 0) {
      pair.second = factory_plane_id++;
    }
  }

  for(size_t i = 0; i < match_vec.size(); i++) {
    for(size_t j = i+1; j < match_vec.size(); j++) {
      if(match_vec[i].second == match_vec[j].second) {
        //match_pairs[match_vec[i].first] = factory_plane_id;
        //match_vec[i].second = factory_plane_id;
        //factory_plane_id++;
        match_pairs[match_vec[j].first] = factory_plane_id;
        match_vec[j].second = factory_plane_id;
        factory_plane_id++;
      }
    }
  }    

  // for(auto pair: match_vec) {
  //   curr_normals_[pair.second] = 
  // }

  std::set<ushort> matched_ids;
    
  host_segment_img = cv::Mat(index_img.rows, index_img.cols, CV_8UC3, cv::Scalar(0, 0, 0));

  for(int i = 0; i < index_img.rows; i++) {
    for(int j = 0; j < index_img.cols; j++) {
      ushort index = index_img.at<ushort>(i, j);
      if(index == 0) continue;
      index_img.at<ushort>(i, j) = match_pairs[index];
      host_segment_img.at<cv::Vec3b>(i, j) = color_table_[match_pairs[index]];
      matched_ids.insert(match_pairs[index]);
    }
  }
  for(auto id: matched_ids) {
    match_num_table_[id]++;
  }
  
  // cv::imshow("new segment img", new_segment_img);
  // cv::waitKey(2);
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Twcr = Twc.cast<float>();
  volume->UpdateSegment((ushort*)index_img.data, (float*)depth_img.data, Twcr.data());

}

// void TSDFMapper::GetRegionBound(const cv::Mat& segment_img, const std::vector<cv::Vec3b> colors, 
//                                 const std::vector<Eigen::Vector3d>& centers, const std::vector<Eigen::Vector3d>& normals)
// {
//   assert(colors.size() == centers.size());
//   assert(colors.size() == normals.size());
  
//   printf("[GetRegionBound]\n");
//   uchar3 *d_segment_img;
//   uchar  *d_binary_img;
//   int num_pixels = segment_img.rows * segment_img.cols;
//   cv::Mat binary_img = cv::Mat(segment_img.rows, segment_img.cols, CV_8U, cv::Scalar(0));


//   gpuErrchk(cudaMalloc((void **)&d_segment_img, sizeof(uchar3) * num_pixels));
//   gpuErrchk(cudaMalloc((void **)&d_binary_img, sizeof(uchar) * num_pixels));
//   gpuErrchk(cudaMemcpy(d_segment_img, segment_img.data, sizeof(uchar3) * num_pixels, cudaMemcpyHostToDevice));
//   gpuErrchk(cudaDeviceSynchronize());

//   printf("[GetRegionBound]\n");
//   for(int i = 0; i < 1; i++) {
//     uchar3 target = make_uchar3(colors[i][0], colors[i][1], colors[i][2]);
//     SelectRegionKernel(d_segment_img, d_binary_img, target, segment_img.rows, segment_img.cols);
//     gpuErrchk(cudaMemcpy(binary_img.data, d_binary_img, sizeof(uchar) * num_pixels, cudaMemcpyDeviceToHost));
//     gpuErrchk(cudaDeviceSynchronize());
    
//     // std::vector<std::vector<cv::Point>> contours;
//     // std::vector<cv::Vec4i> hierarcy;
//     // cv::findContours(binary_img, contours, hierarcy, 0, cv::CHAIN_APPROX_NONE);

//     // std::vector<std::vector<cv::Point>> polygen(contours.size());//用于存放折线点集
//     // cv::Mat show = cv::Mat(segment_img.rows, segment_img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
//     // for (int i = 0; i<contours.size(); i++) {
//     //   cv::approxPolyDP(cv::Mat(contours[i]), polygen[i], 15, true);
//     //   cv::drawContours(show, polygen, i, cv::Scalar(0, 255, 255), 2, 8); 
//     // }
//     // cv::imshow("approx", show);
//   }
//   gpuErrchk(cudaDeviceSynchronize());
//   cv::imshow("region", binary_img);
//   cv::waitKey(2);
// } 

bool TSDFMapper::SurfaceOK() {
  std::unique_lock<std::mutex> lock(surface_mutex);
  return surface_flag;
}