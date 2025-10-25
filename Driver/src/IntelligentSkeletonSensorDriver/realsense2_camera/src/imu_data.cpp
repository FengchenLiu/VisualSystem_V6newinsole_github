//
// Created by shunyi on 2022/3/15.
//

#include "imu_data.h"
#include <iostream>
#include <cstring>

#define IMU_4_BEGIN 72

#define FLOAT_LENGTH 4
#define INT_16_LENGTH 2
#define INT_8_LENGTH 1

using std::cout;
using std::cerr;
using std::endl;

// constructor
IMUData::IMUData() {
  yaw = new float;
  pitch = new float;
  roll = new float;

  accX = new float;
  accY = new float;
  accZ = new float;
}

// getter
float IMUData::getYaw() const {
  return *(this -> yaw);
}

float IMUData::getPitch() const {
  return *(this -> pitch);
}

float IMUData::getRoll() const {
  return *(this -> roll);
}

float IMUData::getAccX() const {
  return *(this -> accX);
}

float IMUData::getAccY() const {
  return *(this -> accY);
}

float IMUData::getAccZ() const {
  return *(this -> accZ);
}

// mehods
void IMUData::convertFromArray(unsigned char *data, int beginIndex) {
  memcpy(this -> yaw, data + beginIndex + FLOAT_LENGTH * 0, FLOAT_LENGTH);
  //cout << "value of yaw: " << *(this -> yaw) << endl;
  memcpy(this -> pitch, data + beginIndex + FLOAT_LENGTH * 1, FLOAT_LENGTH);
  memcpy(this -> roll, data + beginIndex + FLOAT_LENGTH * 2, FLOAT_LENGTH);
  memcpy(this -> accX, data + beginIndex + FLOAT_LENGTH * 3, FLOAT_LENGTH);
  memcpy(this -> accY, data + beginIndex + FLOAT_LENGTH * 4, FLOAT_LENGTH);
  memcpy(this -> accZ, data + beginIndex + FLOAT_LENGTH * 5, FLOAT_LENGTH);
}

void IMUData::writeToFile(std::fstream &file) {
  file << *yaw << " "
        << *pitch << " "
        << *roll << " "
        << *accX << " "
        << *accY << " "
        << *accZ << " "
        << endl;
}

// functions
void writeOneLineToFile(unsigned char *data, std::fstream &file){
  float *value = new float;
  for (int i = 0; i * FLOAT_LENGTH <= 296; ++i) {
    memcpy(value, data + i * FLOAT_LENGTH, FLOAT_LENGTH);
    //cout << "value: " << *value << endl;
    file << *value << " ";
  }
  delete value;

  int16_t *value_int16 = new int16_t;
  for (int i = 300; i <= 302; i += INT_16_LENGTH){
    memcpy(value_int16, data + i, INT_16_LENGTH);
    file << *value_int16 << " ";
  }
  delete value_int16;

  int8_t *value_int8 = new int8_t;
  for (int i = 304; i < 336; ++i){
    memcpy(value_int8, data + i, INT_8_LENGTH);
    file << (int)*value_int8 << " ";
    cout << (int)*value_int8 << " ";
  }
  delete value_int8;

  int32_t *value_int32 = new int32_t;
  memcpy(value_int32, data + 336, FLOAT_LENGTH);
  file << *value_int32 << " ";
  delete value_int32;

  file << endl;
};