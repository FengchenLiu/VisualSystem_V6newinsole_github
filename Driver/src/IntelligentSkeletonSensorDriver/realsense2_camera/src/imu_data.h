//
// Created by shunyi on 2022/3/15.
//

#ifndef IMUREADER_IMUDATA_H
#define IMUREADER_IMUDATA_H
#include <fstream>

class IMUData{
private:
  float *yaw, *pitch, *roll;
  float *accX, *accY, *accZ;
public:
  // constructor
  IMUData();

  // getter
  float getYaw() const;
  float getPitch() const;
  float getRoll() const;

  float getAccX() const;
  float getAccY() const;
  float getAccZ() const;

  // METHODS
  void convertFromArray(unsigned char *data, int beginIndex);
  void writeToFile(std::fstream &file);
};

// functions
void writeOneLineToFile(unsigned char *data, std::fstream &file);
#endif //IMUREADER_IMUDATA_H