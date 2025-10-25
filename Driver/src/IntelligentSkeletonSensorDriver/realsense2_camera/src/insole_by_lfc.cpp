#include <ros/ros.h>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <vector>
#include <thread>
#include <std_msgs/Float32MultiArray.h>

// 打印时间戳
void printTimestamp()
{
    auto now = std::chrono::system_clock::now();
    std::time_t currentTime = std::chrono::system_clock::to_time_t(now);
    std::cout << std::put_time(std::localtime(&currentTime), "%Y-%m-%d %H:%M:%S") << " - ";
}

// 串口配置函数
int configureSerialPort(const char *port)
{
    // 打开串口
    int serialPort = open(port, O_RDWR | O_NOCTTY | O_SYNC);
    if (serialPort == -1)
    {
        std::cerr << "Error opening serial port: " << port << std::endl;
        return -1;
    }

    // 配置串口
    struct termios tty;
    memset(&tty, 0, sizeof tty);
    if (tcgetattr(serialPort, &tty) != 0)
    {
        std::cerr << "Error from tcgetattr" << std::endl;
        return -1;
    }

    // 设置波特率为921600
    cfsetospeed(&tty, B921600);
    cfsetispeed(&tty, B921600);

    tty.c_cflag &= ~PARENB; // 无奇偶校验
    tty.c_cflag &= ~CSTOPB; // 1位停止位
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;            // 8数据位
    tty.c_cflag &= ~CRTSCTS;       // 禁用硬件流控制
    tty.c_cflag |= CREAD | CLOCAL; // 启用接收器和本地连接

    tty.c_iflag &= ~(IXON | IXOFF | IXANY);         // 禁用软件流控制
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG); // 禁用输入处理  (修正:c_iflag改为c_lflag)
    tty.c_oflag &= ~OPOST;                          // 禁用输出处理

    if (tcsetattr(serialPort, TCSANOW, &tty) != 0)
    {
        std::cerr << "Error from tcsetattr" << std::endl;
        return -1;
    }

    return serialPort;
}

// JustFloat协议解析函数
// 每帧数据: 7个float(28字节) + 帧尾(4字节) = 32字节
void decodeAndPublishData(std::vector<unsigned char> &buffer, ros::Publisher &pub)
{
    // JustFloat帧尾标识 (4字节): 00 00 80 7F
    const unsigned char FRAME_TAIL[4] = {0x00, 0x00, 0x80, 0x7F};
    const int FRAME_SIZE = 32; // 7个float(28字节) + 帧尾(4字节)
    
    // 打印当前缓冲区大小和前几个字节(调试用)
    if (buffer.size() >= 10)
    {
        ROS_INFO("Buffer size: %zu, First bytes: %02X %02X %02X %02X %02X %02X %02X %02X", 
                 buffer.size(), buffer[0], buffer[1], buffer[2], buffer[3],
                 buffer[4], buffer[5], buffer[6], buffer[7]);
    }
    
    while (buffer.size() >= FRAME_SIZE)
    {
        // 查找帧尾
        bool foundTail = false;
        size_t tailPos = 0;
        
        for (size_t i = 0; i <= buffer.size() - 4; i++)
        {
            if (buffer[i] == FRAME_TAIL[0] && 
                buffer[i+1] == FRAME_TAIL[1] && 
                buffer[i+2] == FRAME_TAIL[2] && 
                buffer[i+3] == FRAME_TAIL[3])
            {
                // 检查这个帧尾位置是否合理(应该在32的倍数位置-4)
                // 或者至少前面有28个字节的数据
                if (i >= 28)
                {
                    ROS_INFO("Found frame tail at position: %zu", i);
                    foundTail = true;
                    tailPos = i;
                    break;
                }
            }
        }
        
        if (!foundTail)
        {   
            ROS_INFO("No frame tail found in buffer of size %zu", buffer.size());
            
            // 打印整个缓冲区内容以供调试
            if (buffer.size() <= 100)
            {
                std::cout << "Buffer content: ";
                for (size_t i = 0; i < buffer.size(); i++)
                {
                    printf("%02X ", buffer[i]);
                }
                std::cout << std::endl;
            }
            
            // 没找到帧尾,保留最后31个字节(一帧数据-1),删除其他数据
            if (buffer.size() > 31)
            {
                buffer.erase(buffer.begin(), buffer.end() - 31);
            }
            break;
        }
        
        // 提取7个float数据 (从帧尾往前数28字节)
        size_t dataStart = tailPos - 28;
        float floatData[7];
        
        ROS_INFO("Extracting data from position %zu to %zu", dataStart, tailPos - 1);
        
        for (int i = 0; i < 7; i++)
        {
            // 将4个字节转换为float (小端格式)
            unsigned char* ptr = &buffer[dataStart + i * 4];
            memcpy(&floatData[i], ptr, 4);
            
            // 打印每个float的原始字节
            // printf("Float[%d] bytes: %02X %02X %02X %02X = %f\n", 
            //        i, ptr[0], ptr[1], ptr[2], ptr[3], floatData[i]);
        }
        
        // 创建并发布消息
        std_msgs::Float32MultiArray msg;
        msg.data.resize(7);
        
        for (int i = 0; i < 7; i++)
        {
            msg.data[i] = floatData[i];
        }
        
        // 发布消息
        pub.publish(msg);
        
        // 打印调试信息
        printTimestamp();
        std::cout << "Frame: " << floatData[6] 
                  << " | L: [" << floatData[0] << ", " << floatData[1] << ", " << floatData[2] << "]"
                  << " | R: [" << floatData[3] << ", " << floatData[4] << ", " << floatData[5] << "]"
                  << std::endl;
        
        // 删除已处理的数据(包括帧尾)
        buffer.erase(buffer.begin(), buffer.begin() + tailPos + 4);
        
        ROS_INFO("Successfully processed one frame, remaining buffer size: %zu", buffer.size());
    }
}

// 串口读取函数
void readFromSerial(int serialPort, const char *portName, ros::Publisher &pub)
{
    std::vector<unsigned char> buffer;
    unsigned char receivedData[256]; // 增大缓冲区

    while (ros::ok())
    {   
        int bytesRead = read(serialPort, receivedData, sizeof(receivedData));
        if (bytesRead > 0)
        {
            // ROS_INFO("Read %d bytes from serial port", bytesRead);
            
            // 打印读取到的原始数据(前32字节)
            // std::cout << "Raw data: ";
            // for (int i = 0; i < std::min(bytesRead, 32); i++)
            // {
            //     printf("%02X ", receivedData[i]);
            // }
            // std::cout << std::endl;
            
            // 将接收到的数据添加到缓存 
            buffer.insert(buffer.end(), receivedData, receivedData + bytesRead);

            // 尝试解析缓存中的数据包
            decodeAndPublishData(buffer, pub);
        }
        else
        {
            // 短暂延时,避免CPU占用过高
            usleep(1000); // 1ms
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "serial_publisher");
    ros::NodeHandle nh;

    const char *port = "/dev/ttyUSB3"; // 新的串口设备

    // 配置串口
    int serialPort = configureSerialPort(port);
    if (serialPort == -1)
    {
        return -1;
    }

    // 创建ROS发布者 (使用Float32MultiArray因为数据是float类型)
    ros::Publisher pub = nh.advertise<std_msgs::Float32MultiArray>("/hs", 10);
    
    std::cout << "开始接收鞋垫数据 (JustFloat协议, 波特率921600)" << std::endl;
    ROS_INFO("Serial port opened successfully: %s", port);

    // 读取串口数据
    readFromSerial(serialPort, port, pub);

    // 关闭串口
    close(serialPort);

    return 0;
}