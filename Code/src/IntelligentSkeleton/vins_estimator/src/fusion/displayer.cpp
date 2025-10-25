#include "displayer.h"

Displayer::Displayer(TSDFMapper::Ptr mapper)
: mapper_(mapper), pose_ok_(false), stop_(false), image_ok_(false), register_ok_(false), 
  match_ok_(false), host_data_ok_(false), sliding_window_ok_(false), plane_seg_ok_(false), terrain_seg_ok_(false), ground_state_ok_(false)
{
  surface = 0;
  vertex = 0;
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);

  width = mapper->volume->width;
  height = mapper->volume->height;
  max_depth = mapper->volume->max_depth;
  min_depth = mapper->volume->min_depth;

  color_img_ = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  depth_img_ = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  vis_depth_img_ = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  vis_normal_img_ = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
  segmentation_img_ = cv::Mat(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

  host_vertex_img_ = cv::Mat(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
  host_normal_img_ = cv::Mat(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
}

void Displayer::Run()
{  
  
  const int WIDTH = 640;
  const int HEIGHT = 480;
  const int UI_WIDTH = 180;
  pangolin::CreateWindowAndBind("MainWindow", WIDTH*2+UI_WIDTH, HEIGHT*2);
  glewInit(); 

  // 3D Mouse handler requires depth testing to be enabled  
  glEnable(GL_DEPTH_TEST); 

  // Issue specific OpenGl we might need
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


  pangolin::CreatePanel("options").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(UI_WIDTH));
  pangolin::Var<bool> bShowSurface("options.Surface",true,true);
  pangolin::Var<bool> bShowTrajectory("options.ShowTrajectory",true,true);
  pangolin::Var<bool> bShowBbx("options.ShowBoundingBox",false,true);
  pangolin::Var<bool> bShowVertex("options.ShowVertex",false,true);
  pangolin::Var<bool> bShowVirtualVertex("options.ShowVirtualVertex",false,true);
  pangolin::Var<bool> bFollowCamera("options.FollowCamera",true,true);
  pangolin::Var<bool> bReset("options.Reset",false,false);

  // Define Camera Render Object (for view / scene browsing)
  pangolin::OpenGlRenderState s_cam(
    pangolin::ProjectionMatrix(WIDTH,HEIGHT,400,400,WIDTH/2,HEIGHT/2,0.1,1000),
    pangolin::ModelViewLookAt(-10,0,0,0,0,0, pangolin::AxisZ)
  );


  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& d_cam = pangolin::Display("cam")
    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -WIDTH/(float)HEIGHT)
    .SetHandler(new pangolin::Handler3D(s_cam));
    

  pangolin::View& d_color = pangolin::Display("Color")
    .SetAspect(WIDTH/(float)HEIGHT);

  pangolin::View& d_depth = pangolin::Display("Depth")
    .SetAspect(WIDTH/(float)HEIGHT);

  pangolin::View& d_virtual_depth = pangolin::Display("VirtualDepth")
    .SetAspect(WIDTH/(float)HEIGHT);
  
  pangolin::View& d_virtual_normal = pangolin::Display("VirtualNormal")
    .SetAspect(WIDTH/(float)HEIGHT);

  pangolin::View& d_segmentation = pangolin::Display("Segmentation")
    .SetAspect(WIDTH/(float)HEIGHT);

  pangolin::CreateDisplay()
		  .SetBounds(0.0, 0.2, pangolin::Attach::Pix(UI_WIDTH), 1.0)
		  .SetLayout(pangolin::LayoutEqual)
		  .AddDisplay(d_color)
		  .AddDisplay(d_depth)
      .AddDisplay(d_virtual_depth)
      .AddDisplay(d_segmentation)
      .AddDisplay(d_virtual_normal);

    
  bool curFollowFlag = false;

  while(!pangolin::ShouldQuit()) {
    {
      std::unique_lock<std::mutex> lock(stop_mutex);
      if(stop_) {
        break;
      }
    }

    if(bReset) {
      surface = 0;
      vertex = 0;
      virtual_vertex = 0;
      terrain.clear();
      mapper_->Reset();
      reset_flag_ = true;
      bReset = false;
      // assert(1==0);
    }

    double start = cv::getTickCount();
    // Clear screen and activate view to render into
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);

    auto Twc = GetPGLCameraPose();

    if(bFollowCamera && curFollowFlag) {
      s_cam.Follow(Twc);
    }
    else if(bFollowCamera && !curFollowFlag) {
      s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,-0.7,-1.8, 0,0,0,0.0,-1.0, 0.0));
      s_cam.Follow(Twc);
      curFollowFlag = true;
    }
    else if(!bFollowCamera && curFollowFlag) {
      curFollowFlag = false;
    }

    DrawAxis(bShowBbx);
    DrawCamera(Rwc_, twc_);
    DrawSurface(bShowSurface, bShowVertex, bShowVirtualVertex);

    // DrawPlaneSeg();
    DrawTerrainSeg();
    DrawSlidingWindow();
    DrawGroundPos();

    // DrawNormals();
    // DrawResiduals();
    DrawImage();
  
    // DrawRegister(Twc_raw_, Twc_cur_);

    // Swap frames and Process Events
    pangolin::FinishFrame();
    // printf("Pangolin Render Time: %lf\n", (cv::getTickCount() - start) / cv::getTickFrequency());

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  // // unset the current context from the main thread
  pangolin::GetBoundWindow()->RemoveCurrent();

  printf("Distroy Window\n");
}

void Displayer::DrawSurface(const pangolin::Var<bool>& bShowSurface, const pangolin::Var<bool>& bShowVertex, const pangolin::Var<bool>& bShowVirtualVertex)
{
  if(mapper_->SurfaceOK()) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_surface = mapper_->GetSurfaceCloud();
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_vertex = mapper_->GetVertexCloud();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp_virtual_vertex = mapper_->GetVirtualVertexCloud();
    // if(tmp_surface->empty() || tmp_virtual_vertex->empty()) {
    if(tmp_surface->empty()) {
      return;
    }
    surface = PangoCloud::Ptr(new PangoCloud(tmp_surface.get()));
    // vertex = PangoCloud::Ptr(new PangoCloud(tmp_vertex.get()));
    virtual_vertex = PangoCloud::Ptr(new PangoCloud(tmp_virtual_vertex.get()));
  }

  glPointSize(4);                                                          
  if(surface && bShowSurface) {
    surface->drawPoints(); 
  }                                                           
                                            
                                                          
  // glPointSize(1);                     
  // if(vertex && bShowVertex)
  //   vertex->drawPoints();

  glPointSize(1);
  if(virtual_vertex && bShowVirtualVertex)
    virtual_vertex->drawPoints();
}

pangolin::OpenGlMatrix Displayer::GetPGLCameraPose()
{
  pangolin::OpenGlMatrix T;
  T.SetIdentity();
  if(pose_ok_) {
    T.m[0] = Rwc_(0,0);
    T.m[1] = Rwc_(1,0);
    T.m[2] = Rwc_(2,0);
    T.m[3]  = 0.0;

    T.m[4] = Rwc_(0,1);
    T.m[5] = Rwc_(1,1);
    T.m[6] = Rwc_(2,1);
    T.m[7]  = 0.0;

    T.m[8] = Rwc_(0,2);
    T.m[9] = Rwc_(1,2);
    T.m[10] = Rwc_(2,2);
    T.m[11]  = 0.0;

    T.m[12] = twc_(0);
    T.m[13] = twc_(1);
    T.m[14] = twc_(2);
    T.m[15]  = 1.0;

  }
  return T;
}

void Displayer::DrawAxis(const pangolin::Var<bool> bShowBbx)
{
  glLineWidth(3);
  glBegin(GL_LINES);
  glColor3f(0.8f,0.f,0.f);
  glVertex3f(0,0,0);
  glVertex3f(1,0,0);
  glColor3f(0.f,0.8f,0.f);
  glVertex3f(0,0,0);
  glVertex3f(0,1,0);
  glColor3f(0.2f,0.2f,1.f);
  glVertex3f(0,0,0);
  glVertex3f(0,0,1);
  glEnd();
  
  if(bShowBbx) {
    float xb = mapper_->volume->grid_origin_x + mapper_->volume->cur_warp_x * mapper_->volume->voxel_size;
    float yb = mapper_->volume->grid_origin_y + mapper_->volume->cur_warp_y * mapper_->volume->voxel_size;
    float zb = mapper_->volume->grid_origin_z + mapper_->volume->cur_warp_z * mapper_->volume->voxel_size;
    float xt = xb + mapper_->volume->grid_dim_x * mapper_->volume->voxel_size;
    float yt = yb + mapper_->volume->grid_dim_y * mapper_->volume->voxel_size;
    float zt = zb + mapper_->volume->grid_dim_z * mapper_->volume->voxel_size;

    glLineWidth(1);
    glBegin(GL_LINES);
    glColor3f(0.9f,0.9f,0.9f);
    glVertex3f(xb,yb,zb);
    glVertex3f(xt,yb,zb);
    glVertex3f(xb,yt,zb);
    glVertex3f(xt,yt,zb);
    glVertex3f(xb,yb,zt);
    glVertex3f(xt,yb,zt);
    glVertex3f(xb,yt,zt);
    glVertex3f(xt,yt,zt);

    glVertex3f(xb,yb,zb);
    glVertex3f(xb,yt,zb);
    glVertex3f(xt,yb,zb);
    glVertex3f(xt,yt,zb);
    glVertex3f(xb,yb,zt);
    glVertex3f(xb,yt,zt);
    glVertex3f(xt,yb,zt);
    glVertex3f(xt,yt,zt);

    glVertex3f(xb,yb,zb);
    glVertex3f(xb,yb,zt);
    glVertex3f(xt,yb,zb);
    glVertex3f(xt,yb,zt);  
    glVertex3f(xb,yt,zb);
    glVertex3f(xb,yt,zt);  
    glVertex3f(xt,yt,zb);
    glVertex3f(xt,yt,zt);
    glEnd();
  }

}

void Displayer::AddCamera(const Eigen::Matrix4d& Twc, float r, float g, float b, float lw, float scale)
{
  glPushMatrix();

  pangolin::OpenGlMatrix T;
  T.SetIdentity();
  
  T.m[0] = Twc(0,0);
  T.m[1] = Twc(1,0);
  T.m[2] = Twc(2,0);
  T.m[3]  = 0.0;

  T.m[4] = Twc(0,1);
  T.m[5] = Twc(1,1);
  T.m[6] = Twc(2,1);
  T.m[7]  = 0.0;

  T.m[8] = Twc(0,2);
  T.m[9] = Twc(1,2);
  T.m[10] = Twc(2,2);
  T.m[11]  = 0.0;

  T.m[12] = Twc(0,3);
  T.m[13] = Twc(1,3);
  T.m[14] = Twc(2,3);
  T.m[15]  = 1.0;

  glMultMatrixd(T.m);
  const float w = 0.2 * scale;
  const float h = w * 0.75 * scale;
  const float z = w * scale;

  glLineWidth(lw); 
  glBegin(GL_LINES);
  glColor3f(r,g,b);
  glVertex3f(0,0,0);		glVertex3f(w,h,z);
  glVertex3f(0,0,0);		glVertex3f(w,-h,z);
  glVertex3f(0,0,0);		glVertex3f(-w,-h,z);
  glVertex3f(0,0,0);		glVertex3f(-w,h,z);
  glVertex3f(w,h,z);		glVertex3f(w,-h,z);
  glVertex3f(-w,h,z);		glVertex3f(-w,-h,z);
  glVertex3f(-w,h,z);		glVertex3f(w,h,z);
  glVertex3f(-w,-h,z);    glVertex3f(w,-h,z);
  glEnd();
  glPopMatrix();
}

void Displayer::DrawCamera(const Eigen::Matrix3d& Rwc, const Eigen::Vector3d& twc)
{
  Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
  Twc.topLeftCorner(3, 3) = Rwc;
  Twc.topRightCorner(3, 1) = twc;

  AddCamera(Twc, 0, 1, 1, 2);

  if(register_ok_) {
    AddCamera(Twc_raw_, 1, 0, 0, 2);
    AddCamera(Twc_cur_, 0, 1, 0, 2);
  }

  trajectory.push_back(twc);

  if(trajectory.size() <= 300)
    return;

  glLineWidth(5); 
  glBegin(GL_LINES);
  for (size_t i = 290; i < trajectory.size() - 1; i++) {
    glColor3f(0.0f,1.0f,0.0f);
    glVertex3f(trajectory[i].x(),trajectory[i].y(),trajectory[i].z());
    glVertex3f(trajectory[i+1].x(),trajectory[i+1].y(),trajectory[i+1].z());
  }
  glEnd();
}

void Displayer::DrawSlidingWindow()
{
  if(sliding_window_ok_) {
    for(auto &pose: window_poses) {
      AddCamera(pose, 0, 0, 1, 2, 0.5);
    }

    glPointSize(8);
    glBegin(GL_POINTS);
    for(auto &iter: window_landmarks) {
      auto point = iter.first;
      auto color = iter.second;
      glColor3f(color.x(),color.y(),color.z());
      glVertex3f(point.x(),point.y(),point.z());
    }
    glEnd();
  }
}

void Displayer::DrawRegister(const Eigen::Matrix4d& Twc_raw, const Eigen::Matrix4d& Twc_cur)
{
  if(!register_ok_) return;
  // printf("[DrawRegister]\n");
  // std::cout << Twc_cur_ << std::endl;
  // std::cout << Twc_raw_ << std::endl;
  {
    glPushMatrix();

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();
    
    Twc.m[0] = Twc_raw(0,0);
    Twc.m[1] = Twc_raw(1,0);
    Twc.m[2] = Twc_raw(2,0);
    Twc.m[3]  = 0.0;

    Twc.m[4] = Twc_raw(0,1);
    Twc.m[5] = Twc_raw(1,1);
    Twc.m[6] = Twc_raw(2,1);
    Twc.m[7]  = 0.0;

    Twc.m[8] = Twc_raw(0,2);
    Twc.m[9] = Twc_raw(1,2);
    Twc.m[10] = Twc_raw(2,2);
    Twc.m[11]  = 0.0;

    Twc.m[12] = Twc_raw(0,3);
    Twc.m[13] = Twc_raw(1,3);
    Twc.m[14] = Twc_raw(2,3);
    Twc.m[15]  = 1.0;

    glMultMatrixd(Twc.m);
    const float w = 0.2;
    const float h = w * 0.75;
    const float z = w;

    glLineWidth(2); 
    glBegin(GL_LINES);
    glColor3f(1.0f,0.0f,0.0f);
    glVertex3f(0,0,0);		glVertex3f(w,h,z);
    glVertex3f(0,0,0);		glVertex3f(w,-h,z);
    glVertex3f(0,0,0);		glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);		glVertex3f(-w,h,z);
    glVertex3f(w,h,z);		glVertex3f(w,-h,z);
    glVertex3f(-w,h,z);		glVertex3f(-w,-h,z);
    glVertex3f(-w,h,z);		glVertex3f(w,h,z);
    glVertex3f(-w,-h,z);    glVertex3f(w,-h,z);
    glEnd();
    glPopMatrix();
  }

  {
    glPushMatrix();

    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();
    
    Twc.m[0] = Twc_cur(0,0);
    Twc.m[1] = Twc_cur(1,0);
    Twc.m[2] = Twc_cur(2,0);
    Twc.m[3]  = 0.0;

    Twc.m[4] = Twc_cur(0,1);
    Twc.m[5] = Twc_cur(1,1);
    Twc.m[6] = Twc_cur(2,1);
    Twc.m[7]  = 0.0;

    Twc.m[8] = Twc_cur(0,2);
    Twc.m[9] = Twc_cur(1,2);
    Twc.m[10] = Twc_cur(2,2);
    Twc.m[11]  = 0.0;

    Twc.m[12] = Twc_cur(0,3);
    Twc.m[13] = Twc_cur(1,3);
    Twc.m[14] = Twc_cur(2,3);
    Twc.m[15]  = 1.0;

    glMultMatrixd(Twc.m);
    const float w = 0.2;
    const float h = w * 0.75;
    const float z = w;

    glLineWidth(2); 
    glBegin(GL_LINES);
    glColor3f(0.0f,1.0f,0.0f);
    glVertex3f(0,0,0);		glVertex3f(w,h,z);
    glVertex3f(0,0,0);		glVertex3f(w,-h,z);
    glVertex3f(0,0,0);		glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);		glVertex3f(-w,h,z);
    glVertex3f(w,h,z);		glVertex3f(w,-h,z);
    glVertex3f(-w,h,z);		glVertex3f(-w,-h,z);
    glVertex3f(-w,h,z);		glVertex3f(w,h,z);
    glVertex3f(-w,-h,z);    glVertex3f(w,-h,z);
    glEnd();
    glPopMatrix();
  }
}

void Displayer::DrawImage()
{

  if(!image_ok_) return;

  pangolin::GlTexture rgbTex(width,height,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
  pangolin::GlTexture depthTex(width,height,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
  pangolin::GlTexture virtualDepthTex(width,height,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
  pangolin::GlTexture virtualNormalTex(width,height,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);
  pangolin::GlTexture segmentationTex(width,height,GL_RGB,false,0,GL_RGB,GL_UNSIGNED_BYTE);

  rgbTex.Upload(color_img_.data,GL_RGB,GL_UNSIGNED_BYTE);
  depthTex.Upload(depth_img_.data,GL_RGB,GL_UNSIGNED_BYTE);
  virtualDepthTex.Upload(vis_depth_img_.data,GL_RGB,GL_UNSIGNED_BYTE);
  virtualNormalTex.Upload(vis_normal_img_.data,GL_RGB,GL_UNSIGNED_BYTE);
  segmentationTex.Upload(segmentation_img_.data,GL_RGB,GL_UNSIGNED_BYTE);

  pangolin::Display("Color").Activate();
  glColor3f(1.0,1.0,1.0);
  rgbTex.RenderToViewport(true);

  pangolin::Display("Depth").Activate();
  glColor3f(1.0,1.0,1.0);
  depthTex.RenderToViewport(true);

  pangolin::Display("VirtualDepth").Activate();
  glColor3f(1.0,1.0,1.0);
  virtualDepthTex.RenderToViewport(true);

  pangolin::Display("VirtualNormal").Activate();
  glColor3f(1.0,1.0,1.0);
  virtualNormalTex.RenderToViewport(true);

  pangolin::Display("Segmentation").Activate();
  glColor3f(1.0,1.0,1.0);
  segmentationTex.RenderToViewport(true);

}

void Displayer::DrawNormals()
{
  if(!host_data_ok_)
    return;

  int rows = host_vertex_img_.rows, cols = host_vertex_img_.cols;
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      cv::Vec3f vertex = host_vertex_img_.at<cv::Vec3f>(i, j);
      cv::Vec3f normal = host_normal_img_.at<cv::Vec3f>(i, j);

      if(vertex[0] == 0.0f && vertex[1] == 0.0f && vertex[2] == 0.0f)
        continue;
      
      glLineWidth(2);
      glBegin(GL_LINES);    
      glColor3f(1, 0, 1);
      glVertex3f(vertex[0], vertex[1], vertex[2]);
      glVertex3f(vertex[0] + 0.03 * normal[0], vertex[1] + 0.03 * normal[1], vertex[2] + 0.03 * normal[2]);
      glEnd();
    }
  }
}

void Displayer::DrawResiduals()
{

  if(!match_ok_)
    return;

  for(size_t i = 0; i < residuals_.size(); i++) {
    auto residual = residuals_[i];
    Eigen::Vector3d vertex = cur_Twc_.block<3, 3>(0, 0) * residual.vertex + cur_Twc_.block<3, 1>(0, 3);

    Eigen::Vector3d host_vertex = residual.host_vertex;
    glPointSize(7);
    glBegin(GL_POINTS);    
    glColor3f(1, 0, 1);
    glVertex3f(vertex.x(), vertex.y(), vertex.z());
    glColor3f(1, 1, 0);
    glVertex3f(host_vertex.x(), host_vertex.y(), host_vertex.z());
    glEnd();
  }

  for(size_t i = 0; i < residuals_.size(); i++) {
    auto residual = residuals_[i];
    glLineWidth(2);
    glBegin(GL_LINES);
    glColor3f(0, 1, 1);
    Eigen::Vector3d vertex = cur_Twc_.block<3, 3>(0, 0) * residual.vertex + cur_Twc_.block<3, 1>(0, 3);
    Eigen::Vector3d host_vertex = residual.host_vertex;
    Eigen::Vector3d host_normal = residual.host_normal;
    glVertex3f(host_vertex.x(), host_vertex.y(), host_vertex.z());
    glVertex3f(host_vertex.x() + 0.1 * host_normal.x(), host_vertex.y() + 0.1 * host_normal.y(), host_vertex.z() + 0.1 * host_normal.z());

    glColor3f(0, 0, 1);
    glVertex3f(host_vertex.x(), host_vertex.y(), host_vertex.z());
    glVertex3f(vertex.x(), vertex.y(), vertex.z());
    glEnd();
  }
}

void Displayer::DrawPlaneSeg()
{
  if(!plane_seg_ok_)
    return;
  
  // printf("DrawPlaneSeg %ld, %ld, %ld\n", centers.size(), colors.size(), normals.size());
  for(size_t i = 0; i < centers.size(); i++) {
    auto center = centers[i];
    auto color = colors[i];
    glPointSize(7);
    glBegin(GL_POINTS);    
    // glColor3f((float)color[0] / 255, (float)color[1] / 255, (float)color[2] / 255);
    glColor3f((float)color[2] / 255, (float)color[1] / 255, (float)color[0] / 255);
    glVertex3f(center.x(), center.y(), center.z());
    glEnd();
  }

  for(size_t i = 0; i < normals.size(); i++) {
    auto center = centers[i];
    auto color = colors[i];
    auto normal = normals[i];
    glLineWidth(10);
    glBegin(GL_LINES);  
    // glColor3f((float)color[0] / 255, (float)color[1] / 255, (float)color[2] / 255);
    glColor3f((float)color[2] / 255, (float)color[1] / 255, (float)color[0] / 255);
    glVertex3f(center.x(), center.y(), center.z());
    glVertex3f(center.x() + 0.2 * normal.x(), center.y() + 0.2 * normal.y(), center.z() + 0.2 * normal.z());
    glEnd();
  }

}

void Displayer::DrawTerrainSeg()
{
  if(!terrain_seg_ok_)
    return;
  
  for(size_t i = 0; i < terrain.size(); i++) {
    for(size_t j = 0; j < terrain[i].size(); j++) {
      
      auto center = terrain[i][j].first;
      auto color = terrain[i][j].second;

      glPointSize(20);
      glBegin(GL_POINTS);    
      glColor3f((float)color[2] / 255, (float)color[1] / 255, (float)color[0] / 255);
      glVertex3f(center.x(), center.y(), center.z());
      glEnd();     

      if(j > 0) {
        auto prev_center = terrain[i][j-1].first;
        glLineWidth(10);
        glBegin(GL_LINES);  
        // glColor3f((float)color[0] / 255, (float)color[1] / 255, (float)color[2] / 255);
        glColor3f(1, 1, 1);
        glVertex3f(center.x(), center.y(), center.z());
        glVertex3f(prev_center.x(), prev_center.y(), prev_center.z());
        glEnd();
      } 
    }

  }
}

void Displayer::DrawGroundPos()
{
  if(ground_state_ok_) {
    if(plane_id_ > 0) {
      glPointSize(20);
      glBegin(GL_POINTS);    
      glColor3f(0, 1, 0);
      glVertex3f(ground_pos_.x(), ground_pos_.y(), ground_pos_.z());
      glEnd();     
    }
  }
}

void Displayer::SetPose(const Eigen::Matrix3d& Rwc, const Eigen::Vector3d& twc)
{
  std::unique_lock<std::mutex> lock(pose_mutex);
  Rwc_ = Rwc;
  twc_ = twc;
  pose_ok_ = true;
}

void Displayer::SetImage(const cv::Mat& color_img, const cv::Mat& depth_img)
{

  assert(color_img.type() == CV_8UC3 && depth_img.type() == CV_32F);



  cv::Mat depth_f(height, width, CV_32F);
  memcpy((void*)depth_f.data, (void*)depth_img.data, width * height * sizeof(float));

  depth_f.setTo(cv::Scalar(0), depth_img > max_depth);
  depth_f.setTo(cv::Scalar(0), depth_img < min_depth);

  cv::Mat depth_8u;
  depth_f.convertTo(depth_8u, CV_8U, 255. / max_depth);

  {
    std::unique_lock<std::mutex> lock(image_mutex);
    image_ok_ = false;

    cv::cvtColor(color_img, color_img_, cv::COLOR_BGR2RGB);  
    cv::applyColorMap(depth_8u, depth_img_, cv::COLORMAP_JET);
    cv::cvtColor(depth_img_, depth_img_, cv::COLOR_BGR2RGB);
    depth_img_.setTo(cv::Scalar(0, 0, 0), depth_8u == 0);
    image_ok_ = true;
  }

  cv::putText(color_img_, "Color", cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(255, 255, 255), 3, 8);
  cv::putText(depth_img_, "Depth", cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(255, 255, 255), 3, 8);
}

void Displayer::SetHostImage(const cv::Mat& host_vertex, const cv::Mat& host_normal)
{
  std::unique_lock<std::mutex> lock(image_mutex);
  host_vertex_img_ = host_vertex;
  host_normal_img_ = host_normal;
  host_data_ok_ = true;
}

void Displayer::SetRenderedImage(const cv::Mat& depth_img, const cv::Mat& normal_img)
{
  std::unique_lock<std::mutex> lock(image_mutex);
  cv::Mat host_depth_img = depth_img.clone();
  host_depth_img.setTo(0, depth_img > 6.0);
  cv::Mat vis_depth_img_8u;
  host_depth_img.convertTo(vis_depth_img_8u, CV_8U, 255. / 6.0);

  cv::applyColorMap(vis_depth_img_8u, vis_depth_img_, cv::COLORMAP_JET);
  vis_depth_img_.setTo(cv::Scalar(0, 0, 0), vis_depth_img_8u == 0);

  cv::cvtColor(vis_depth_img_, vis_depth_img_, cv::COLOR_RGB2BGR);
  cv::putText(vis_depth_img_, "VirtualDepth", cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(255, 255, 255), 3, 8);
  
  cv::Mat host_normal_img = normal_img.clone();
  host_normal_img.convertTo(vis_normal_img_, CV_8U, 128, 128);
  // vis_normal_img_.setTo(cv::Scalar(0, 0, 0), vis_normal_img_ == cv::Vec3f(128, 128, 128));

  // cv::cvtColor(vis_normal_img_, vis_normal_img_, cv::COLOR_BGR2RGB);
  cv::putText(vis_normal_img_, "VirtualNormal", cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(255, 255, 255), 3, 8);
}
void Displayer::SetSegmentationImage(const cv::Mat& segmentation_img)
{
  std::unique_lock<std::mutex> lock(image_mutex);
  memcpy((void*)segmentation_img_.data, (void*)segmentation_img.data, width * height * sizeof(uchar3));
  cv::cvtColor(segmentation_img_, segmentation_img_, cv::COLOR_BGR2RGB);
  cv::putText(segmentation_img_, "Segmentation", cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1.3, cv::Scalar(255, 255, 255), 3, 8);
}

void Displayer::SetRegisterResult(const Eigen::Matrix4d &Twc_raw, const Eigen::Matrix4d& Twc_cur)
{
  std::unique_lock<std::mutex> lock(pose_mutex);
  Twc_raw_ = Twc_raw;
  Twc_cur_ = Twc_cur;
  register_ok_ = true;
}

void Displayer::SetAssociateResult(const std::vector<SurfaceResidual>& residuals, const Eigen::Matrix4d& Twc_host, const Eigen::Matrix4d& Twc_cur)
{
  std::unique_lock<std::mutex> lock(pose_mutex);
  residuals_ = residuals;
  host_Twc_ = Twc_host;
  cur_Twc_ = Twc_cur;
  match_ok_ = true;
}

void Displayer::SetSlidingWindow(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3f>> &landmarks, const std::vector<Eigen::Matrix4d> &poses)
{
  std::unique_lock<std::mutex> lock(window_mutex);
  window_landmarks = landmarks;
  window_poses = poses;
  sliding_window_ok_ = true;
}

void Displayer::SetPlaneSegResult(const std::vector<cv::Vec3b>& _colors, const std::vector<Eigen::Vector3d>& _centers, const std::vector<Eigen::Vector3d>& _normals)
{
  std::unique_lock<std::mutex> lock(plane_seg_mutex);
  colors = _colors;
  centers = _centers;
  normals = _normals;
  assert(colors.size() == centers.size());
  assert(colors.size() == normals.size());
  plane_seg_ok_ = true;
}

void Displayer::SetTerrainSegResult(const std::vector<std::vector<std::pair<Eigen::Vector3d, cv::Vec3b>>>& result)
{
  std::unique_lock<std::mutex> lock(terrain_seg_mutex);
  terrain = result;
  terrain_seg_ok_ = true;
}

void Displayer::SetGroundPos(const unsigned short plane_id, const Eigen::Vector3d& ground_pos)
{
  std::unique_lock<std::mutex> lock(ground_state_mutex);
  plane_id_ = plane_id;
  ground_pos_ = ground_pos;
  ground_state_ok_ = true;
}

void Displayer::Stop()
{
  std::unique_lock<std::mutex> lock(stop_mutex);
  stop_ = true;
}