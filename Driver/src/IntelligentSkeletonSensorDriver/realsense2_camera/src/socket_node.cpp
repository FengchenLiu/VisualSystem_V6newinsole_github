// include the std cpp header files
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <realsense2_camera/Press.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

// include other header files
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <ctime>

#include "imu_data.h"

using std::cout;
using std::cerr;
using std::cin;
using std::endl;

#define DEST_IP "192.168.1.1"
#define DEST_PORT 8899
#define RECEIVE_SIZE 100000
#define EFFECTIVE_DATA_LENGTH 1020
#define SINGLE_BAG_LENGTH 340

#define BEGIN_FLAG_STRING_LENGTH 8
#define FLOAT_LENGTH 4

#define SKELETON_IMU_NUM 12

// begin index area
#define IMU_1_BEGIN 0
#define IMU_2_BEGIN 24
#define IMU_3_BEGIN 48
#define IMU_4_BEGIN 72
#define IMU_5_BEGIN 96
#define IMU_6_BEGIN 120
#define IMU_7_BEGIN 144
#define IMU_8_BEGIN 168
#define IMU_9_BEGIN 192
#define IMU_10_BEGIN 216
#define IMU_11_BEGIN 240
#define IMU_12_BEGIN 264

#define LEFT_PRESS_BEGIN 304
#define RIGHT_PRESS_BEGIN 320

void printCharArrayLength(unsigned char *array, int length);
std::vector<int> checkNoneZeroIndexes(unsigned char *array, int length);
std::vector<unsigned char *> divideToThreeVectors(unsigned char *array);

ros::Publisher skeleton_imu_pub[SKELETON_IMU_NUM];
ros::Publisher press_pub;

void Buffer2Info(unsigned char *data, sensor_msgs::Imu *imu_msg, realsense2_camera::Press *press_msg, std_msgs::Header header)
{
  float imu_i[6];

  for (int i = 0; i < SKELETON_IMU_NUM; ++i) {
    int start_addr = i * 6 * FLOAT_LENGTH;
    memcpy((void*)&imu_i, data + start_addr, 6 * FLOAT_LENGTH);

    imu_msg[i].header = header;
    imu_msg[i].orientation.z = imu_i[0];
    imu_msg[i].orientation.y = imu_i[1];
    imu_msg[i].orientation.x = imu_i[2];
    imu_msg[i].linear_acceleration.x = imu_i[3];
    imu_msg[i].linear_acceleration.y = imu_i[4];
    imu_msg[i].linear_acceleration.z = imu_i[5];
  }

  memcpy((void*)&press_msg->left, data + 304, 16);
  memcpy((void*)&press_msg->right, data + 320, 16);
  press_msg->header = header;

  // std::cout << "press_left: " << std::endl;
  // for(int i = 0; i < 16; i++) {
  //   std::cout << (int)press_msg->left[i] << " ";
  // }
  // std::cout << std::endl;
  // std::cout << "press_right: " << std::endl;
  // for(int i = 0; i < 16; i++) {
  //   std::cout << (int)press_msg->right[i] << " ";
  // }
  // std::cout << std::endl;
}

int main(int argc, char **argv) {

  ros::init(argc, argv, "socket_node");
  ROS_INFO("\033[1;32m---->\033[0m Socket Node.");
  ros::NodeHandle nh("~");
  time_t now = time(0);

  for(int i = 0; i < SKELETON_IMU_NUM; i++) {
    std::string topic_name = "/skelenton_imu_" + std::to_string(i);
    skeleton_imu_pub[i] = nh.advertise<sensor_msgs::Imu>(topic_name, 100);
  }

  press_pub = nh.advertise<realsense2_camera::Press>("/press", 100);

  // convert now to string
  char *dt = ctime(&now);
  std::string currT(dt);
  //cout << "time: " << dt << endl;
  cout << "time: " << currT << endl;

  // init the socket
  int client = socket(AF_INET, SOCK_STREAM, 0);
  if (client == -1){
    cout << "Error: socket init failed!" << std::endl;
    return 0;
  }

  // connect
  struct sockaddr_in serverAddr;
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_port = htons(DEST_PORT);
  serverAddr.sin_addr.s_addr = inet_addr(DEST_IP);

  cout << "...connect" << endl;

  if (connect(client, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0){
    cout << "Error: can not connect to dest!" << endl;
    return 0;
  }

  cout << "success!" << endl;

  std::string strBeginFlag("view=1\n\r");
  std::string strReceiveBuff("");
  send(client, strBeginFlag.c_str(), BEGIN_FLAG_STRING_LENGTH, 0);

  unsigned char *receivedData = new unsigned char[RECEIVE_SIZE];

  for(int i = 0;i < RECEIVE_SIZE; ++i){
    receivedData[i] = 'z';
  }

  std::fstream dataFile("data" + currT + ".txt", std::ios::out);

  std::fstream dataFile1("data1.txt", std::ios::out);
  std::fstream dataFile2("data2.txt", std::ios::out);
  std::fstream dataFile3("data3.txt", std::ios::out);
  std::fstream dataFile4("data4.txt", std::ios::out);
  std::fstream dataFile5("data5.txt", std::ios::out);
  std::fstream dataFile6("data6.txt", std::ios::out);
  std::fstream dataFile7("data7.txt", std::ios::out);

  sensor_msgs::Imu imu_msg[SKELETON_IMU_NUM]; 
  realsense2_camera::Press press_msg;
  // while (!cin.eof()){
  while (ros::ok()){
    int recvLength = recv(client, receivedData, RECEIVE_SIZE, 0);
    std_msgs::Header header;
    header.frame_id = "body";
    header.stamp = ros::Time::now();
    // cout << "the length of received data: " << recvLength << endl;
    //exit(1);

    if (recvLength == EFFECTIVE_DATA_LENGTH){
      //printCharArrayLength(receivedData, EFFECTIVE_DATA_LENGTH);
      //std::vector<int> effIndexes = checkNoneZeroIndexes(receivedData, EFFECTIVE_DATA_LENGTH);
      //for (auto const &c : effIndexes) cout << c << " ";
      // cout << endl;

      std::vector<unsigned char *> vecThreeBags = divideToThreeVectors(receivedData);
      // double cur_time = header.stamp.toSec();
      // for(int i = 0; i < 3; i++) {
      //   header.stamp.fromSec(cur_time - (2-i) * 0.01);
      //   Buffer2Info(vecThreeBags[i], imu_msg, &press_msg, header);
      //   for(int i = 0; i < SKELETON_IMU_NUM; i++) {
      //     skeleton_imu_pub[i].publish(imu_msg[i]);
      //   }
      //   press_pub.publish(press_msg);
      // }

      Buffer2Info(vecThreeBags[2], imu_msg, &press_msg, header);
      for(int i = 0; i < SKELETON_IMU_NUM; i++) {
        skeleton_imu_pub[i].publish(imu_msg[i]);
      }
      press_pub.publish(press_msg);
      

      // // print the results of these three bag
      // for (int i = 0; i < vecThreeBags.size(); ++i){
      //   cout << "bag index: " << i << endl;
      //   printCharArrayLength(vecThreeBags[i], SINGLE_BAG_LENGTH);
      //   std::vector<int> effIndexes = checkNoneZeroIndexes(vecThreeBags[i], SINGLE_BAG_LENGTH);
      //   for (auto const &c : effIndexes) cout << c << " ";
      //   cout << endl;

      //   IMUData currData4;
      //   currData4.convertFromArray(vecThreeBags[i], IMU_4_BEGIN);
      //   currData4.writeToFile(dataFile4);

      //   // write to One file
      //   writeOneLineToFile(vecThreeBags[i], dataFile);
      // }
    }
    ros::spinOnce();
  }

  delete[] receivedData;

  return 0;
}

void printCharArrayLength(unsigned char *array, int length){
  for(int i = 0; i < length; ++i){
    cout << (int )*(array + i) << " ";
  }
  cout << endl;
}

std::vector<int> checkNoneZeroIndexes(unsigned char *array, int length){
  std::vector<int> retIndexes;

  for (int i = 0; i < length; ++i){
    if ((int)*(array + i) != 0){
      retIndexes.push_back(i);
    }
  }

  return retIndexes;
}

std::vector<unsigned char *> divideToThreeVectors(unsigned char *array){
  unsigned char *data1 = new unsigned char[SINGLE_BAG_LENGTH];
  unsigned char *data2 = new unsigned char[SINGLE_BAG_LENGTH];
  unsigned char *data3 = new unsigned char[SINGLE_BAG_LENGTH];

  memcpy(data1, array, SINGLE_BAG_LENGTH);
  memcpy(data2, array + SINGLE_BAG_LENGTH, SINGLE_BAG_LENGTH);
  memcpy(data3, array + SINGLE_BAG_LENGTH * 2, SINGLE_BAG_LENGTH);

  return std::vector<unsigned char *>{data1, data2, data3};
}