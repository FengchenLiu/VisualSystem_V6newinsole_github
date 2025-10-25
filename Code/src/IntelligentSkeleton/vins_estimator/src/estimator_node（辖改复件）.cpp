#include <stdio.h>
#include <queue>
#include <map>
#include <ctime>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <queue>
#include <deque>
#include <memory>
#include <filesystem>
#include <unordered_map>
#include <shared_mutex>

// include pcl io header files
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"

#include "fusion/tsdf_mapper.h"
#include "fusion/displayer.h"
#include "scene_graph/json.hpp"
#include "scene_graph/utils_control.h"

#include <opencv2/opencv.hpp>

using json = nlohmann::json;

#include "segmentation/SurfaceExtraction.h"

#include <bigai_robotic_msgs/Press.h>

// for publishing
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <cv_bridge/cv_bridge.h>
#include <thread>
#include <mutex>
#include <std_msgs/Float32MultiArray.h> // 用于发布多个浮点数的消息类型

// lfc add
#include <iostream>
#include <opencv2/core.hpp> // OpenCV 用于加载模型

// lfc 4.13
#include <fstream>
#include <chrono>
#include <iomanip>
//  lfc 5.28
#include <pcl/visualization/pcl_visualizer.h>

Estimator estimator;
Displayer::Ptr displayer;
TSDFMapper::Ptr mapper;

// std::shared_mutex rw_mutex;

std::mutex thread_mutex;
std::mutex thread_mutex_2;
std::condition_variable backend_cv;
Eigen::Matrix4d Twc;    // newest frame T
Eigen::Matrix4d Twc_kf; // keyframe as host frame T
cv::Mat cur_color_map;
Eigen::Quaterniond cur_pose_quater;
Eigen::Vector3d cur_pos_vec;
cv::Mat cur_depth_map;

ros::Publisher image_pub_;
ros::Publisher quat_pub_;
ros::Publisher pwc_pub_;
ros::Publisher pubSVM; // lfc
ros::Publisher pubtime; // lfc zwd

ros::Publisher pub_currPwc; // lfc zwd
ros::Publisher pub_currVwc; // lfc zwd
ros::Publisher pub_currRwc; // lfc zwd
ros::Publisher pub_next_dist ; // lfc zwd

ros::Time hs_TimeStamp ; 

bool pressFlag = false;

std::condition_variable con;
double current_time = -1;
std::queue<sensor_msgs::ImuConstPtr> imu_buf;
std::queue<bigai_robotic_msgs::PressConstPtr> press_buf;
std::queue<sensor_msgs::PointCloudConstPtr> feature_buf;
std::queue<sensor_msgs::PointCloudConstPtr> relo_buf;
std::queue<sensor_msgs::ImageConstPtr> color_msg_buf;
std::queue<sensor_msgs::ImageConstPtr> depth_msg_buf;
std::map<double, std::pair<sensor_msgs::ImageConstPtr, sensor_msgs::ImageConstPtr>> img_table;

std::deque<Eigen::Vector3d> deq_Velo;
unsigned long LENGTH_SAMPLE = 3 ;
float img_resolu = 0.02;

ushort stair_id = 0, level_ground_id = 0;
std::map<ushort, cv::Vec3b> stair_color_table, level_ground_color_table;

int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;
double hs_TS = 0;

// boost::array<double, 3> savedStairHeight{15, 20, 25};
/**********Scene Graph Declaration Area**********/
#include <fstream>
Eigen::Vector3d currPwc;
Eigen::Vector3d currVwc;
Eigen::Vector3d currEulerAngle;
Eigen::Matrix3d currRwc;
double currTimeStamp;

//zwd lfc
// static bool firstTime = true;
// static double firstTimeStamp;
float pub_time;

// vwc pwc and euler angle saving path
std::string cfgPath("/home/lfc/lfc/VisualSystem_V6newinsole/Code/src/IntelligentSkeleton/vins_estimator/src/scene_graph/scene_graph_config.json");

std::string output_Path_dir("./Special_Output");

json scene_graph_cfg;

std::fstream vwcSavingFileObj;
std::fstream pwcSavingFileObj;
std::fstream eulerSavingFileObj;
std::fstream landingSavingFileObj;
std::fstream backTimeSavingFileObj;
std::fstream allSavingFileObj;

bigai_robotic_msgs::PressConstPtr currPressMessage;

// the char string will be written to
char *p_char_writtings;

std::string raw_depth_path, host_depth_path, raw_color_path, pcd_path;

cv::Mat savingFlagImage;


struct Measurement
{
    std::vector<sensor_msgs::ImuConstPtr> imu_msg_vec;
    sensor_msgs::PointCloudConstPtr feature_msg;
    sensor_msgs::ImageConstPtr color_msg;
    sensor_msgs::ImageConstPtr depth_msg;

    Measurement(const std::vector<sensor_msgs::ImuConstPtr> &_imu_msg_vec,
                const sensor_msgs::PointCloudConstPtr &_feature_msg,
                const sensor_msgs::ImageConstPtr &_color_msg,
                const sensor_msgs::ImageConstPtr &_depth_msg)
        : imu_msg_vec(_imu_msg_vec), feature_msg(_feature_msg), color_msg(_color_msg), depth_msg(_depth_msg) {}
};

void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }
    double dt = t - latest_time;
    latest_time = t;

    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;
    tmp_V = tmp_V + dt * un_acc;

    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());
}

std::vector<Measurement> getMeasurements()
{
    std::vector<Measurement> measurements;

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            // ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr feature_msg = feature_buf.front();
        double timestamp = feature_msg->header.stamp.toSec();
        if (img_table.find(timestamp) == img_table.end())
            return measurements;

        feature_buf.pop();

        std::vector<sensor_msgs::ImuConstPtr> imu_msg_vec;
        while (imu_buf.front()->header.stamp.toSec() < feature_msg->header.stamp.toSec() + estimator.td)
        {
            imu_msg_vec.emplace_back(imu_buf.front());
            imu_buf.pop();
        }

        imu_msg_vec.emplace_back(imu_buf.front());
        if (imu_msg_vec.empty())
            ROS_WARN("no imu between two image");

        auto image_msg = img_table[feature_msg->header.stamp.toSec()];
        // currTimeStamp = feature_msg->header.stamp.toSec();
        // measurements.emplace_back(imu_msg_vec, feature_msg);
        measurements.emplace_back(Measurement(imu_msg_vec, feature_msg, image_msg.first, image_msg.second));
    }
    return measurements;
}

void imuCallback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    // std::cout << "imuCallback" << std::endl;
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder! %f", imu_msg->header.stamp.toSec());
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();

    last_imu_t = imu_msg->header.stamp.toSec();

    {
        std::lock_guard<std::mutex> lg(m_state);
        // predict imu (no residual error)
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
        {
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
            Eigen::Matrix3d Rwc = tmp_Q.toRotationMatrix() * RIC[0];
            currRwc = Rwc;
            currEulerAngle = Rwc.eulerAngles(0, 1, 2);
            Eigen::Vector3d Pwc = tmp_Q.toRotationMatrix() * TIC[0] + tmp_P;
            currPwc = Pwc;
            Eigen::Vector3d Vwc = tmp_V;

            if (deq_Velo.size() < 800)
            {
                deq_Velo.push_back(Vwc);
            }
            else
            {
                deq_Velo.pop_front();
                deq_Velo.push_back(Vwc);
            }

            currVwc = Vwc;
            currTimeStamp = last_imu_t;

            if (true)
            {
                pwcSavingFileObj << std::setprecision(15) << currPwc[0] << " " << currPwc[1] << " " << currPwc[2] << " " << currTimeStamp << std::endl;
                vwcSavingFileObj << std::setprecision(15) << currVwc[0] << " " << currVwc[1] << " " << currVwc[2] << " " << currTimeStamp << std::endl;
                eulerSavingFileObj << std::setprecision(15) << currEulerAngle[0] << " " << currEulerAngle[1] << " " << currEulerAngle[2] << " " << currTimeStamp << std::endl;
            }

            displayer->SetPose(Rwc, Pwc);
        }
    }
}

void pressCallback(const bigai_robotic_msgs::PressConstPtr &press_msg)
{
    m_buf.lock();
    press_buf.push(press_msg);
    currPressMessage = press_msg;
    m_buf.unlock();
}
/**************************************lfc add , insole callback function**************************************/

#include <ros/ros.h>
#include <std_msgs/UInt16MultiArray.h>
#include <chrono>
#include <iomanip>

bool switch1_L = true; // 定义一个标志位
bool switch2_L = true;
bool switch1_R = true; // 定义一个标志位
bool switch2_R = true;

bool heel_strike_detected = true; // 新标志位1：表示是否检测到足跟触底
bool left_heel_strike = false;     // 新标志位2：表示是否为左脚足跟触底
bool right_heel_strike = false;    // 新标志位3：表示是否为右脚足跟触底
int countHS = 0;

// 新加 lfc6.8 
int left_foot  =10 ;   //左右脚标志位，左脚10 表示 在地面 ，20表示在空中
int right_foot =10 ;
std::deque<int> left_foot_history;  // 存储 left_foot 的最近3个状态
std::deque<int> right_foot_history;  // 存储 left_foot 的最近3个状态

//  声明一个txt路径.
std::fstream hs_file_obj("/home/lfc/lfc/VisualSystem_V6newinsole/Code/logs/hs.txt", std::ios::out);

int threshold = 2500;   // 基础值合 应该 不超过1000    2500
int all_press_L =0;        // 初始化 两侧两个 压力值
int all_press_R =0;
  
// int init_TL = scene_graph_cfg["TL"] ;   //  用ofa测量的鞋垫平均初始值
// int init_TR = scene_graph_cfg["TR"] ;  

//  足跟触底 改回阈值。  
void hsCallback(const std_msgs::UInt16MultiArray::ConstPtr &msg)   
{
    int allhs=0 ;
    // 遍历 msg->data 数组，计算所有元素的和
    for (int i = 0; i < 32; ++i) {
        allhs += msg->data[i];
    }                    

    if (msg->data[32] == 1)
    {

        all_press_L=allhs-32*600;

        if ( all_press_L > threshold ) 
        {
            left_foot = 10;  // 10在地面
        }
        else
        {
            left_foot = 20;  // 20在空中
        }

        if (left_foot_history.size() > 3)
        {
            left_foot_history.pop_front();  // 移除最旧的状态
        }
        // 检查是否满足 [20, 20] → 10 的条件
        if (left_foot_history[0] == 20 && 
            left_foot_history[1] == 20 && 
            left_foot_history[2] == 20 &&   
            left_foot == 10)  // 当前状态是10
        {
            heel_strike_detected = true;  // 触地事件发生
            left_heel_strike=true ;
            countHS++; // 记录足跟触底次数
        }

        left_foot_history.push_back(left_foot);
        //将鞋垫数据写入txt
        hs_file_obj << std::setprecision(6) <<","<< threshold <<","<< allhs <<","<< all_press_L <<","<<left_foot<<","<<left_heel_strike<<"" << msg->data[32] <<","<< "/n"<< std::endl;
        // 去掉多的数值
        
    }
    else if (msg->data[32] == 2)
    {   
        all_press_R =allhs-32*600;

        if ( all_press_R > threshold  )   //21800 //21000
        {
            right_foot = 10;  // 10在地面
        }
        else
        {
            right_foot = 20;  // 20在空中
        }

        // 去掉多的数值
        if (right_foot_history.size() > 3)
        {
            right_foot_history.pop_front();  // 移除最旧的状态
        }
        // 检查是否满足 [20, 20, 20] → 10 的条件
        if (right_foot_history[0] == 20 && 
            right_foot_history[1] == 20 &&
            right_foot_history[2] == 20 && 
            right_foot == 10)  // 当前状态是10
        {
            heel_strike_detected = true;  // 触地事件发生
            right_heel_strike=true ;
            countHS++; // 记录足跟触底次数
        }
        //  填充
        right_foot_history.push_back(right_foot);

    }

}

/***************************************************************************/

void imageCallback(const sensor_msgs::ImageConstPtr &color_msg, const sensor_msgs::ImageConstPtr &depth_msg)
{
    m_buf.lock();
    double timestamp = color_msg->header.stamp.toSec();
    color_msg_buf.push(color_msg);
    depth_msg_buf.push(depth_msg);
    img_table.insert(std::make_pair(timestamp, std::make_pair(color_msg, depth_msg)));
    m_buf.unlock();
}

void featureCallback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        // skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one();
}

void restartCallback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while (!feature_buf.empty())
            feature_buf.pop();
        while (!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

void relocalizationCallback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    // printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
void FrontendLoop()
{
    while (true)
    {
        // std::cout << "CODE 运行到这里了 FrontendLoop 1 " << std::endl;
        // std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;
        std::vector<Measurement> measurements;
        std::unique_lock<std::mutex> lk(m_buf);
        con.wait(lk, [&]
                 { return (measurements = getMeasurements()).size() != 0; });
        lk.unlock();
        m_estimator.lock();
        // std::cout << "CODE 运行到这里了 FrontendLoop 2 " << std::endl;
        for (auto &measurement : measurements)
        {
            // std::cout << "CODE 运行到这里了 FrontendLoop 3 " << std::endl;
            auto feature_msg = measurement.feature_msg;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.imu_msg_vec)
            {
                double t = imu_msg->header.stamp.toSec();
                double img_t = feature_msg->header.stamp.toSec() + estimator.td;
                if (t <= img_t)
                {
                    if (current_time < 0)
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                }
                else
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    // printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame
            // std::cout << "CODE 运行到这里了 FrontendLoop 4" << std::endl;
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;
            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);
            }

            ROS_DEBUG("processing vision data with stamp %f \n", feature_msg->header.stamp.toSec());

            TicToc t_s;
            map<int, vector<pair<int, Eigen::Matrix<double, 8, 1>>>> feature;

            for (unsigned int i = 0; i < feature_msg->points.size(); i++)
            {

                int v = feature_msg->channels[0].values[i] + 0.5;

                int feature_id = v / NUM_OF_CAM;

                int camera_id = v % NUM_OF_CAM;
                double x = feature_msg->points[i].x;
                double y = feature_msg->points[i].y;
                double z = feature_msg->points[i].z;
                double p_u = feature_msg->channels[1].values[i];
                double p_v = feature_msg->channels[2].values[i];
                double velocity_x = feature_msg->channels[3].values[i];
                double velocity_y = feature_msg->channels[4].values[i];
                double depth = feature_msg->channels[5].values[i] / 1000.0;

                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 8, 1> xyz_uv_velocity_depth;
                xyz_uv_velocity_depth << x, y, z, p_u, p_v, velocity_x, velocity_y, depth;
                feature[feature_id].emplace_back(camera_id, xyz_uv_velocity_depth);
            }

            ROS_ASSERT(measurement.color_msg != nullptr);
            ROS_ASSERT(measurement.depth_msg != nullptr);
            // process the inputted color image
            cv_bridge::CvImageConstPtr color_rt_ptr;
            color_rt_ptr = cv_bridge::toCvShare(measurement.color_msg, sensor_msgs::image_encodings::BGR8);
            cv::Mat cur_color_img = color_rt_ptr->image;
            // cur_color_img.copyTo(cur_color_map);
            cur_color_map = cur_color_img;
            cur_pose_quater = tmp_Q;
            cur_pos_vec = currPwc;

            // process the inputted depth image
            cv_bridge::CvImageConstPtr depth_rt_ptr;
            depth_rt_ptr = cv_bridge::toCvShare(measurement.depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
            cv::Mat cur_depth_img = depth_rt_ptr->image;

            cur_depth_img.convertTo(cur_depth_img, CV_32F, 0.001);

            displayer->SetImage(cur_color_img, cur_depth_img);

            TicToc process_img;
            estimator.processImage(feature, cur_depth_img, feature_msg->header);
            //  lfc 刚注释掉
            // printf("[processImage] cost time: %lf\n", process_img.toc());

            m_buf.lock();
            m_state.lock();
            if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
                update();
            m_state.unlock();
            m_buf.unlock();

            double whole_t = t_s.toc();
            //  lfc 刚注释掉
            // printf("[Full Process] cost time: %lf\n", whole_t);

            printStatistics(estimator, whole_t);
            std_msgs::Header header = feature_msg->header;
            header.frame_id = "world";
            // utility/visualization.cpp
            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubOldestCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);

            if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR && estimator.marginalization_flag == 0)
            {

                thread_mutex.lock();
                int integration_id = WINDOW_SIZE - 2;
                double integration_time = estimator.Headers[integration_id].stamp.toSec();
                auto image_msg = img_table[integration_time];

                cv_bridge::CvImageConstPtr color_ptr;
                color_ptr = cv_bridge::toCvShare(image_msg.first, sensor_msgs::image_encodings::BGR8);

                cv_bridge::CvImageConstPtr depth_ptr;
                depth_ptr = cv_bridge::toCvShare(image_msg.second, sensor_msgs::image_encodings::TYPE_16UC1);

                cur_color_map = color_ptr->image;
                cur_depth_map = depth_ptr->image;
                cur_depth_map.convertTo(cur_depth_map, CV_32F, 0.001f);

                Eigen::Matrix4d Twi = Eigen::Matrix4d::Identity();
                Twi.block<3, 3>(0, 0) = estimator.Rs[integration_id];
                Twi.block<3, 1>(0, 3) = estimator.Ps[integration_id];
                Eigen::Matrix4d Tic = Eigen::Matrix4d::Identity();
                Tic.block<3, 3>(0, 0) = RIC[0];
                Tic.block<3, 1>(0, 3) = TIC[0];
                Twc = Eigen::Matrix4d::Identity();
                Twc = Twi * Tic;

                // 1nd newest keyframe when marginalize old
                int host_id = WINDOW_SIZE - 1;
                Twi.block<3, 3>(0, 0) = estimator.Rs[host_id];
                Twi.block<3, 1>(0, 3) = estimator.Ps[host_id];
                Twc_kf = Eigen::Matrix4d::Identity();
                Twc_kf = Twi * Tic;

                thread_mutex.unlock();
                backend_cv.notify_all();
            }
        }
        m_estimator.unlock();
    }
}

void BackendLoop()
{
    while (1)
    {
        std::unique_lock<std::mutex> lock(thread_mutex);
        backend_cv.wait(lock);
        TicToc backend_time;

        TicToc integrate_time;
        mapper->UpdateTSDF(cur_color_map, cur_depth_map, Twc);
        mapper->MoveVolume(Twc);
        printf("[Integrate] Run time: %lf ms\n", integrate_time.toc());

        TicToc render_time;
        mapper->RenderView(Twc_kf);
        printf("[RenderView] Run time: %lf ms\n", render_time.toc());

        double start = cv::getTickCount();
        ImagePlaneFeature surf_feature = extractSurfaces(mapper->host_depth_img, K, Twc_kf, 3.0, 0.0);
        printf("[extractSurfaces] Run time: %lf ms\n", (cv::getTickCount() - start) / cv::getTickFrequency() * 1000);

        cv::Mat index_img = surf_feature.getIndexImg();
        std::vector<Eigen::Vector3d> centers, normals;
        std::vector<cv::Vec3b> colors;
        std::vector<PlaneFeature> feature_vec = surf_feature.getFeatureVec();
        for (size_t i = 0; i < feature_vec.size(); i++)
        {
            centers.push_back(feature_vec[i].global_center);
            normals.push_back(feature_vec[i].global_normal);
            colors.push_back(feature_vec[i].color);
        }

        mapper->AddNewSegmentImage(index_img, mapper->host_depth_img, Twc_kf, colors, centers, normals);

        TicToc extract_time;
        mapper->ExtractSurface();
        printf("[ExtractSurface] Run time: %lf ms\n", extract_time.toc());

        auto surfaces = mapper->curr_surfaces_;
        centers.clear();
        normals.clear();
        colors.clear();
        if (displayer->GetResetFlag())
        {
            pcl::io::savePCDFileASCII("/home/lfc/lfc/VisualSystem_V6newinsole/Code/VIO/saved_data/whole_point_cloud.pcd", *(mapper->surface));
            std::fstream currPwcFileObj;
            currPwcFileObj.open("/home/lfc/lfc/VisualSystem_V6newinsole/Code/VIO/saved_data/currPwc.txt", std::ios::out);
            currPwcFileObj << currPwc[0] << " "
                           << currPwc[1] << " "
                           << currPwc[2] << " "
                           << std::endl;
            currPwcFileObj.close();

            std::fstream deq_Velo_FileObj;
            deq_Velo_FileObj.open("/home/lfc/lfc/VisualSystem_V6newinsole/Code/VIO/saved_data/deq_Velo.txt", std::ios::out);
            for (std::deque<Eigen::Vector3d>::iterator iter = deq_Velo.begin(); iter != deq_Velo.end(); ++iter)
            {
                deq_Velo_FileObj << (*(iter))[0] << " "
                                 << (*(iter))[1] << " "
                                 << (*(iter))[2] << " "
                                 << std::endl;
            }
            deq_Velo_FileObj.close();
        }

        displayer->SetRenderedImage(mapper->host_depth_img, mapper->host_normal_img);
        displayer->SetSegmentationImage(mapper->host_segment_img);
        displayer->SetPlaneSegResult(colors, centers, normals);
        printf("[Backend] Run time: %lf ms\n", backend_time.toc());

        // displayer->SetGroundPos(plane_id, ground_pos);
    }
}

// lfc 挪到外面了
float s_height = 0.00;
float next_dist = 0.00;
// float manheight = 188.0;   //新加 人身高
// lfc 新定义一个结构体，来接
struct sendDataStorage
{
    double newtimestamp;
    float stairheight;
    float t2mdist;
    float velocity;
    float manheight;
    float position;  
    // 默认构造函数，初始化为零
    sendDataStorage() : newtimestamp(0), stairheight(0.00f), t2mdist(0.00f), velocity(0.00f), manheight(1880.00) ,position(0.00f) {}
};
sendDataStorage globalData;


// lfc
// std::deque<float> bufferV;

// +++ 双缓存结构体 +++   lfc 改双缓存
struct SharedData {
    std::vector<std::string> eff_vec_color_key;
    std::unordered_map<std::string, std::vector<Eigen::Vector3d>> map_planes;
};

SharedData shared_data_front;  // 前台数据（消费者读取）
SharedData shared_data_back;   // 后台数据（生产者更新）
std::mutex swap_mutex;         // 缓冲区交换锁


void cal_Ess_info_and_send()
{   

    // std::cout << "CODE 运行到这里了 cal_Ess_info_and_send 1 " << std::endl;
    // const double interval = 4.80;  // 控制单次 发送的 总时间的 ms   1000/5=200Hz
    // currPwc[0] 右手边是正方向， currPwc[1]是前进方向是正方向 
    const double interval = 9.60;

    // lfc 5.5
    std::deque<float> bufferD(5, 0.0f); // 初始化为5个0.0    6.7 给左侧用
    std::deque<float> bufferD2(5, 0.0f); // 初始化为5个0.0   6.7 给右侧用
    std::deque<float> left_bufferV;     // 左脚速度缓冲区
    std::deque<float> right_bufferV;    // 右脚速度缓冲区
    float aveV = 0.0f;                  // 平均步速，
    float sizeofstride = 0.0f;          // 平均步长

    int hs_label=0;  // hs计数
    int side_label=0;

    p_char_writtings = (char *)malloc(sizeof(char) * 40);

    std::fstream res_file_obj("/home/lfc/lfc/VisualSystem_V6newinsole/Code/logs/res.txt", std::ios::out);
    // lfc
    std::fstream keymess_file_obj("/home/lfc/lfc/VisualSystem_V6newinsole/Code/logs/keymess.txt", std::ios::out);
    // std::cout << "CODE 运行到这里了 cal_Ess_info_and_send 2 " << std::endl;
    while (true)
    {
        // std::cout << "CODE 运行到这里了 cal_Ess_info_and_send 3 " << std::endl;
        // lfc 注释掉
        // std::cout << "******************** In the while loop of calculate essential info and sending ********************" << std::endl;
        auto cal_start = std::chrono::high_resolution_clock::now();
        if (mapper->surface != NULL && !(mapper->surface->empty()))
        {
            // std::unique_lock<std::mutex> lock(thread_mutex_2);  //注释锁 
           // === 修改点1：替换原锁，复制前台数据 ===
            SharedData local_data;
            {
                std::lock_guard<std::mutex> lock(swap_mutex);
                local_data = shared_data_front; // 复制前台数据
            }
            
            std::cout << "\n\n\n\n******************** try to calculate ess info and send ********************" << std::endl;
            auto start = std::chrono::high_resolution_clock::now();

            Eigen::Vector3d copy_curr_Pwc;
            copy_curr_Pwc << currPwc[0], currPwc[1], currPwc[2];

            //  lfc 注释掉
            // std::cout << "Copy Curr Pwc " << std::endl
            //           << copy_curr_Pwc << std::endl;

            Eigen::Vector3d target_Point, hori_vec;
            if (deq_Velo.size() == 800)
            {
                Eigen::Vector2d avg_velo_xy = c_comp::cal_avg_velo_xy(deq_Velo);

                copy_curr_Pwc[0] = copy_curr_Pwc[0] - 0.2 * avg_velo_xy[0];
                copy_curr_Pwc[1] = copy_curr_Pwc[1] - 0.2 * avg_velo_xy[1];

                target_Point << copy_curr_Pwc[0] + LENGTH_SAMPLE * avg_velo_xy[0],
                    copy_curr_Pwc[1] + LENGTH_SAMPLE * avg_velo_xy[1],
                    copy_curr_Pwc[2];

                hori_vec << avg_velo_xy[0], avg_velo_xy[1], 0.0;
            }
            else
            {
                continue;
            }


            std::vector<float> vec_min_dist;
            std::vector<double> vec_mean_z;
            std::vector<std::pair<float, double>> vec_pair_min_dist_mean_z;

            c_comp::cal_min_values(local_data.eff_vec_color_key,  // 使用副本
                                    local_data.map_planes,         // 使用副本
                                    vec_min_dist,
                                    copy_curr_Pwc,
                                    vec_mean_z,
                                    vec_pair_min_dist_mean_z);

            // get the height and next distance
            if (vec_mean_z.size() == 0)
            {
                continue;
            }
            if (vec_mean_z.size() == 1)
            {
                s_height = -0.2;
                next_dist = -0.2;
            }
            else
            {
                // set the s_height and min distance
                s_height = abs(vec_pair_min_dist_mean_z[0].second - vec_pair_min_dist_mean_z[1].second);
                next_dist = abs(vec_pair_min_dist_mean_z[0].first - vec_pair_min_dist_mean_z[1].first);
                // 后面这5行是新加的看看能不能避免误识别 lfc 6.17
                if (s_height>0.5 || s_height <-0.5 || (s_height <0.03 && s_height <0.03  ))  
                {
                    s_height = abs(vec_pair_min_dist_mean_z[0].second - vec_pair_min_dist_mean_z[2].second);
                    next_dist = abs(vec_pair_min_dist_mean_z[0].first - vec_pair_min_dist_mean_z[2].first);
                }

            }

            // lfc 赋值
            globalData.velocity = std::sqrt(currVwc[0] * currVwc[0] + currVwc[1] * currVwc[1]);
            globalData.newtimestamp = currTimeStamp;
            globalData.stairheight = s_height;

            if (next_dist <0 ) {
                globalData.t2mdist  = 2.5;
            } else {
                globalData.t2mdist = next_dist;
            }
           
            globalData.position = std::sqrt(currPwc[0] * currPwc[0] + currPwc[1] * currPwc[1]);


            // 同步添加数据到左右缓冲区
            left_bufferV.push_back(globalData.velocity);
            right_bufferV.push_back(globalData.velocity);
            // 控制缓冲区大小（可选）
            if (left_bufferV.size() > 400)
            {
                left_bufferV.pop_front();
            }
            if (right_bufferV.size() > 400)
            {
                right_bufferV.pop_front();
            }


            std::cout << "HSlabel= " << heel_strike_detected
                      << ", right_heel_strike= " << right_heel_strike
                      << ", countHS= " << countHS
                      << ", velocity= " << globalData.velocity
                      << ", stairheight= " << globalData.stairheight
                      << ", t2mdist= " << globalData.t2mdist
                      << std::endl;

            hs_TimeStamp= ros::Time::now();

            if (heel_strike_detected)
            {
                hs_TimeStamp= ros::Time::now();
                                
                if (left_heel_strike)
                {
                    hs_label=1;
                    side_label=11;   // side label  1表示左侧。
                    float left_aveV = 0; // 左侧的平均速度
                    int valid_count = 0;
                    for (float v : left_bufferV)
                    {
                        if (v != 0)
                        { // 忽略接近零的值
                            left_aveV += v;
                            valid_count++; // 记录有效数据个数
                        }
                    }
                    aveV = (valid_count > 0) ? (left_aveV / valid_count) : 0; // aveV = left_aveV / left_bufferV.size(); //
                    left_bufferV.clear();                                     // 清空左速度buffer

                    // lfc6.8
                    bufferD.push_back(globalData.position);   // lfc6.8
                    if (bufferD.size() > 1)
                    {
                        sizeofstride = bufferD[bufferD.size() - 1] - bufferD[bufferD.size() - 2]; // stride
                    }
                    

                }
                else if (right_heel_strike)
                {
                    hs_label=1;
                    side_label=22;   // side label  2表示右侧。
                    float right_aveV = 0; // 左侧的平均速度
                    int valid_count = 0;
                    for (float v : right_bufferV)
                    {
                        if (v != 0)
                        { // 忽略接近零的值
                            right_aveV += v;
                            valid_count++;
                        }
                    }
                    aveV = (valid_count > 0) ? (right_aveV / valid_count) : 0;
                    right_bufferV.clear(); // 清空左速度buffer

                    // lfc6.8
                    bufferD2.push_back(globalData.position);   // lfc6.8
                    if (bufferD2.size() > 1)
                    {
                        sizeofstride = bufferD2[bufferD2.size() - 1] - bufferD2[bufferD2.size() - 2]; // stride
                    }
                }

                if (bufferD.size() > 5)
                {
                    bufferD.pop_front(); // 使用 pop_front() 来移除最前面的元素
                }

                if (bufferD2.size() > 5)
                {
                    bufferD2.pop_front(); // 使用 pop_front() 来移除最前面的元素
                }
                // if (bufferD.size() > 1)
                // {
                //     sizeofstride = bufferD[bufferD.size() - 1] - bufferD[bufferD.size() - 3]; // stride
                // }

                // 重置标志位
                heel_strike_detected = false;
                left_heel_strike = false;
                right_heel_strike = false;

            }
            // 创建并填充消息   发布出去  //lfc6.6新修改，之前pub过程是在 heel 函数里面的，现在放到外面，变成了实时发送，并不只是hs的时候发送
            std_msgs::Float32MultiArray svmmsg;
            svmmsg.data.push_back(globalData.manheight);               // 添加manheight    毫米
            svmmsg.data.push_back(globalData.stairheight * 100);       // 添加stairheight // [cm] 转换为厘米
            svmmsg.data.push_back(globalData.t2mdist * 1000);          // 添加t2mdist    [mm] 转换为毫米
            svmmsg.data.push_back(sizeofstride * 1000);                // 添加sizeofstride   [mm] 转换为毫米
            svmmsg.data.push_back(aveV * 1000);                        // 添加aveV     [mm] 转换为毫米
            svmmsg.data.push_back(hs_label); // 6.6新加  hs标志位
            svmmsg.data.push_back(side_label); // 6.6新加  左右侧 标志位，是左脚足跟触底，还是右脚足跟触底
            svmmsg.data.push_back(left_foot );      // 左脚是在空中还是在地面上
            svmmsg.data.push_back(right_foot );      // 右脚是在空中还是在地面上

            svmmsg.data.push_back(all_press_L );      // 把左右脚的压力 放上
            svmmsg.data.push_back(all_press_R );      


            // 修改发布代码
            double hs_TS = hs_TimeStamp.toSec();
            uint32_t sec_part = static_cast<uint32_t>(hs_TS);      // 整数秒
            uint32_t msec_part = static_cast<uint32_t>((hs_TS - sec_part) * 1000000);  // 微秒部分

            svmmsg.data.push_back(static_cast<float>(sec_part));   // 整数秒
            svmmsg.data.push_back(static_cast<float>(msec_part));  // 微秒部分


            // hs_TS= hs_TimeStamp.toSec();
            // svmmsg.data.push_back(hs_TS);
            std::cout << ", hs_TimeStamp!!!!!!!!!!!!!!!!!= " << std::fixed << std::setprecision(6) <<  static_cast<float>(sec_part) << std::endl;
            std::cout << ", hs_TimeStamp!!!!!!!!!!!!!!!!!= " << std::fixed << std::setprecision(6) <<  static_cast<float>(msec_part) << std::endl;

            hs_label=0;   //hs标志位归零

            auto end = std::chrono::high_resolution_clock::now();

            // std::chrono::duration<double, std::milli> elapsed = end - start;
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            int sleepDuration = interval * 1000 - static_cast<int>(elapsed.count());

            // svmmsg.data.push_back(sleepDuration );
            pubSVM.publish(svmmsg);   //改双缓存后新加。
            
            std::cout << "******************** sleep duration " << sleepDuration << " ********************\n\n" << std::endl;
            if (sleepDuration > 0)
            {
                std::this_thread::sleep_for(std::chrono::microseconds(sleepDuration));
            }
        }
        

        auto cal_end = std::chrono::high_resolution_clock::now();
        auto cal_duration = std::chrono::duration_cast<std::chrono::microseconds>(cal_end - cal_start);

    }
}

void cut_The_PointCloud()
{
    const int interval = 300;
    // const int interval = 100;
    
    // 创建PCL可视化对象
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Cloud Viewer"));
    viewer->setBackgroundColor(0.0, 0.0, 0.0); // 设置背景色为黑色
    viewer->addCoordinateSystem(1.0); // 添加坐标系
    viewer->initCameraParameters(); // 初始化相机参数
    
    // [修改] 设置一个固定的视角位置，只在开始时设置一次
    // 这将提供一个俯视整个场景的视角
    viewer->setCameraPosition(
        0.0, 0.0, 10.0,  // 相机位置：在原点上方10个单位
        0.0, 0.0, 0.0,   // 视点位置：指向原点
        0.0, 1.0, 0.0);  // 上方向：y轴方向为上
    
    // [新增] 为显示t2mdist创建文本ID
    const std::string t2mdist_text_id = "t2mdist_text";
    
    Eigen::Vector3d gravity_vec;
    gravity_vec << 0.0, 0.0, -1.0;
    
    while (true)
    {
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "-------------------- In the while loop of cut the pointcloud --------------------" << std::endl;
        
        Eigen::Vector3d target_Point, hori_vec;
        
        if (mapper->surface != NULL && !(mapper->surface->empty()))
        {
            viewer->removeAllShapes();
            viewer->removeAllPointClouds();
            
            // [新增] 格式化并添加t2mdist文本显示
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2);
            ss << "T2M Distance: " << globalData.t2mdist* 1000;
            viewer->addText(ss.str(), 10, 30, 18, 1.0, 1.0, 1.0, t2mdist_text_id);
            
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> white_color(mapper->surface, 255, 255, 255);
            viewer->addPointCloud<pcl::PointXYZRGB>(mapper->surface, white_color, "original_cloud");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original_cloud");
            
            Eigen::Vector3d copy_curr_Pwc;
            copy_curr_Pwc << currPwc[0], currPwc[1], currPwc[2];

            if (deq_Velo.size() == 800)
            {
                Eigen::Vector2d avg_velo_xy = c_comp::cal_avg_velo_xy(deq_Velo);
                copy_curr_Pwc[0] = copy_curr_Pwc[0] - 0.2 * avg_velo_xy[0];
                copy_curr_Pwc[1] = copy_curr_Pwc[1] - 0.2 * avg_velo_xy[1];

                target_Point << copy_curr_Pwc[0] + LENGTH_SAMPLE * avg_velo_xy[0],
                    copy_curr_Pwc[1] + LENGTH_SAMPLE * avg_velo_xy[1],
                    copy_curr_Pwc[2];

                hori_vec << avg_velo_xy[0], avg_velo_xy[1], 0.0;
                
                // 添加当前位置和目标点的可视化
                viewer->addSphere(pcl::PointXYZ(copy_curr_Pwc[0], copy_curr_Pwc[1], copy_curr_Pwc[2]), 0.1, 0.0, 1.0, 0.0, "current_position");
                viewer->addSphere(pcl::PointXYZ(target_Point[0], target_Point[1], target_Point[2]), 0.1, 1.0, 0.0, 0.0, "target_position");
                
                // 添加速度向量可视化
                viewer->addArrow(
                    pcl::PointXYZ(copy_curr_Pwc[0], copy_curr_Pwc[1], copy_curr_Pwc[2]),
                    pcl::PointXYZ(copy_curr_Pwc[0] + avg_velo_xy[0], copy_curr_Pwc[1] + avg_velo_xy[1], copy_curr_Pwc[2]),
                    0.0, 0.0, 1.0, "velocity_vector");
            }
            else
            {
                continue;
            }

            Eigen::Vector4f box_min_point, box_max_point;

            box_min_point << min(copy_curr_Pwc[0], target_Point[0]),
                min(copy_curr_Pwc[1], target_Point[1]),
                min(copy_curr_Pwc[2], target_Point[2]) - 2,
                1.0;

            box_max_point << max(copy_curr_Pwc[0], target_Point[0]),
                max(copy_curr_Pwc[1], target_Point[1]),
                max(copy_curr_Pwc[2], target_Point[2]),
                1.0;
                
            // 添加裁剪盒可视化
            viewer->addCube(
                box_min_point[0], box_max_point[0], 
                box_min_point[1], box_max_point[1], 
                box_min_point[2], box_max_point[2], 
                0.0, 1.0, 1.0, "crop_box");
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, 
                                              pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, 
                                              "crop_box");
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 1.0, "crop_box");
            viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "crop_box");

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr crop_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

            pcl::CropBox<pcl::PointXYZRGB> crop;
            crop.setInputCloud(mapper->surface);
            crop.setMin(box_min_point);
            crop.setMax(box_max_point);
            crop.filter(*crop_cloud);

            if (!crop_cloud->empty())
            {
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> red_color(crop_cloud, 255, 0, 0);
                viewer->addPointCloud<pcl::PointXYZRGB>(crop_cloud, red_color, "cropped_cloud");
                viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cropped_cloud");
            }
            else
            {
                std::cout << "The cropped point cloud is empty!----------" << std::endl;
            }

            // 更新可视化窗口，但不更改视角
            viewer->spinOnce(1);

            Eigen::Vector3d plane_norm = hori_vec.cross(gravity_vec);
            
            // 添加平面法向量可视化
            viewer->addArrow(
                pcl::PointXYZ(copy_curr_Pwc[0], copy_curr_Pwc[1], copy_curr_Pwc[2]),
                pcl::PointXYZ(copy_curr_Pwc[0] + plane_norm[0], copy_curr_Pwc[1] + plane_norm[1], copy_curr_Pwc[2] + plane_norm[2]),
                1.0, 0.0, 1.0, "plane_normal");

            // 以下代码保持不变
            std::vector<std::string> vec_color_key;
            std::unordered_map<std::string, pcl::PointCloud<pcl::PointXYZRGB>::Ptr> map_planes_in_pointclouds;
            cv::Mat vis_img = cv::Mat::zeros(cv::Size(360, 360), CV_8UC3);
            
            c_comp::filter_and_project(crop_cloud,
                                    0.1,           
                                    copy_curr_Pwc,
                                    plane_norm,
                                    vec_color_key,
                                    shared_data_back.map_planes,
                                    map_planes_in_pointclouds,
                                    img_resolu,
                                    vis_img,
                                    hori_vec);

            cv::imshow("visualization result of projection", vis_img);
            cv::waitKey(1);

            c_comp::fit_3D_plane(vec_color_key,
                                map_planes_in_pointclouds,
                                shared_data_back.eff_vec_color_key);
            
            {
                std::lock_guard<std::mutex> lock(swap_mutex);
                std::swap(shared_data_front, shared_data_back);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        int sleepDuration = interval - static_cast<int>(elapsed.count());
        std::cout << "The pcd updating sleep duration is " << sleepDuration << std::endl;
        if (sleepDuration > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(sleepDuration));
        }
    }
}


void my_kf_publisher()
{
    while (true)
    {
        // printf("-------------------- publish the image and relative pose 1 --------------------\n");
        if (!cur_color_map.empty())
        {
            // printf("-------------------- publish the image and relative pose 2 --------------------\n");
            sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", cur_color_map).toImageMsg();
            img_msg->header.stamp = ros::Time::now();

            geometry_msgs::QuaternionStamped quater_msg;

            quater_msg.header.stamp = ros::Time::now();
            quater_msg.quaternion.w = cur_pose_quater.w();
            quater_msg.quaternion.x = cur_pose_quater.x();
            quater_msg.quaternion.y = cur_pose_quater.y();
            quater_msg.quaternion.z = cur_pose_quater.z();

            geometry_msgs::PointStamped pwc_msg;

            pwc_msg.header.stamp = ros::Time::now();
            pwc_msg.point.x = cur_pos_vec.x();
            pwc_msg.point.y = cur_pos_vec.y();
            pwc_msg.point.z = cur_pos_vec.z();

            image_pub_.publish(img_msg);
            quat_pub_.publish(quater_msg);
            pwc_pub_.publish(pwc_msg);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    registerPub(n);
    TSDFMapper::TSDFMappingOptions options;
    options.height = (int)ROW;
    options.width = (int)COL;
    options.grid_dim_x = GRID_DIM_X;
    options.grid_dim_y = GRID_DIM_Y;
    options.grid_dim_z = GRID_DIM_Z;
    options.voxel_size = VOXEL_SIZE;
    options.K = K;
    options.max_depth = MAX_DEPTH;
    options.min_depth = MIN_DEPTH;

    srand((unsigned int)time(NULL));
    for (unsigned short i = 1; i < 100; ++i)
    {
        cv::Vec3b color;
        while (color[0] < 10 || color[1] < 10 || color[2] < 10)
        {
            color[0] = rand() % 256;
            color[1] = rand() % 256;
            color[2] = rand() % 256;
            if (color[0] > 10 && color[1] > 10 && color[2] > 10)
            {
                stair_color_table.insert(std::make_pair(stair_id, color));
                stair_id++;
                break;
            }
        }
    }

    for (unsigned short i = 1; i < 100; ++i)
    {
        cv::Vec3b color;
        while (color[0] < 10 || color[1] < 10 || color[2] < 10)
        {
            color[0] = rand() % 256;
            color[1] = rand() % 256;
            color[2] = rand() % 256;
            if (color[0] > 10 && color[1] > 10 && color[2] > 10)
            {
                level_ground_color_table.insert(std::make_pair(level_ground_id, color));
                level_ground_id++;
                break;
            }
        }
    }

    // create directory with std::filesystem
    if (!std::filesystem::exists(output_Path_dir))
    {
        if (std::filesystem::create_directory(output_Path_dir))
        {
            std::cout << "Directory Create Success!: " << output_Path_dir << std::endl;
        }
        else
        {
            std::cout << "Directory Create Fail!: " << output_Path_dir << std::endl;
        }
    }

    // read configurs from cfg file
    std::ifstream sg_cfg(cfgPath);
    scene_graph_cfg = json::parse(sg_cfg);

    /*****opening saving files in this main function*****/
    pwcSavingFileObj.open(scene_graph_cfg["strPwcSavingPath"], std::ios::out);
    vwcSavingFileObj.open(scene_graph_cfg["strVwcSavingPath"], std::ios::out);
    eulerSavingFileObj.open(scene_graph_cfg["strEulerSavingPath"], std::ios::out);
    landingSavingFileObj.open(scene_graph_cfg["strLandingRealTime"], std::ios::out);
    backTimeSavingFileObj.open(scene_graph_cfg["strBackendTimeComsumption"], std::ios::out);
    allSavingFileObj.open(scene_graph_cfg["strAllRecordedData"], std::ios::out);

    if (!pwcSavingFileObj.is_open())
    {
        std::cerr << "can not open pwc saving file!" << std::endl;
        exit(1);
    }

    if (!vwcSavingFileObj.is_open())
    {
        std::cerr << "can not open vwc saving file!" << std::endl;
        exit(1);
    }

    if (!eulerSavingFileObj.is_open())
    {
        std::cerr << "can not open euler angles saving file!" << std::endl;
        exit(1);
    }

    if (!landingSavingFileObj.is_open())
    {
        std::cerr << "can not open landing postion saving file!" << std::endl;
        exit(1);
    }

    if (!backTimeSavingFileObj.is_open())
    {
        std::cerr << "can not open backend time comsumption saving file!" << std::endl;
        exit(1);
    }

    if (!allSavingFileObj.is_open())
    {
        std::cerr << "can not open all recorded data saving file!" << std::endl;
        exit(1);
    }

    raw_depth_path = scene_graph_cfg["raw_depth_path"];
    host_depth_path = scene_graph_cfg["host_depth_path"];
    raw_color_path = scene_graph_cfg["raw_color_path"];
    pcd_path = scene_graph_cfg["merged_pcd_path"];

    savingFlagImage = cv::imread("/home/lfc/lfc/VisualSystem_V6newinsole/Code/src/IntelligentSkeleton/SavingFlag.drawio.png");

    std::cout << "end of the open files area" << std::endl;

    mapper = TSDFMapper::Ptr(new TSDFMapper(options));
    displayer = Displayer::Ptr(new Displayer(mapper));
    estimator.setDisplayer(displayer);

    image_pub_ = n.advertise<sensor_msgs::Image>("estimator/rgb_kf_image", 1);
    quat_pub_ = n.advertise<geometry_msgs::QuaternionStamped>("estimator/rgb_kf_quaternion", 1);
    pwc_pub_ = n.advertise<geometry_msgs::PointStamped>("estimator/rgb_kf_position", 1);
    pubSVM = n.advertise<std_msgs::Float32MultiArray>("/svmneed", 10); // lfc
    pubtime = n.advertise<std_msgs::Float32>("/timestamp", 10);  //lfc zwd 6.3


    // 初始化发布器  zwd lfc 6.4
    pub_currPwc = n.advertise<geometry_msgs::PointStamped>("/currPwc", 10);
    pub_currVwc = n.advertise<geometry_msgs::PointStamped>("/currVwc", 10);
    pub_currRwc = n.advertise<geometry_msgs::QuaternionStamped>("/currRwc", 10);
    pub_next_dist = n.advertise<std_msgs::Float32>("/next_dist", 10);


    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imuCallback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_press = n.subscribe("/press", 2000, pressCallback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, featureCallback);
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restartCallback);
    // topic from pose_graph, notify if there's relocalization
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalizationCallback);


    // *****************************lfc add
    ros::Subscriber sub_hs = n.subscribe("/hs", 10, hsCallback);
    // *****************************lfc add

    message_filters::Subscriber<sensor_msgs::Image> sub_key_color_img(n, IMAGE_TOPIC, 1);
    message_filters::Subscriber<sensor_msgs::Image> sub_key_depth_img(n, DEPTH_TOPIC, 1);

    // fit fisheye camera
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> syncPolicy;
    message_filters::Synchronizer<syncPolicy> sync(syncPolicy(10), sub_key_color_img, sub_key_depth_img);
    sync.registerCallback(boost::bind(&imageCallback, _1, _2));

    std::thread frontend(FrontendLoop);    //1 
    std::thread backend(BackendLoop);      // 2
    std::thread cutPCD(cut_The_PointCloud);       // 3 待解决问题，通过双缓存，解决线程锁的问题，
    std::thread send_info(cal_Ess_info_and_send); // 4 interval = 4.80;  // 控制单次 发送的 总时间的 ms   1000/5=200Hz，控制频率

    std::thread visualization = std::thread(&Displayer::Run, displayer);   //5
 
    ros::spin();

    return 0;
}
