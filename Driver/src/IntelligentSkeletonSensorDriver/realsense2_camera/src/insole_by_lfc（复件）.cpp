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
#include <std_msgs/UInt16MultiArray.h>

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

    // 设置波特率为460800
    cfsetospeed(&tty, B460800);
    cfsetispeed(&tty, B460800);

    tty.c_cflag &= ~PARENB; // 无奇偶校验
    tty.c_cflag &= ~CSTOPB; // 1位停止位
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;            // 8数据位
    tty.c_cflag &= ~CRTSCTS;       // 禁用硬件流控制
    tty.c_cflag |= CREAD | CLOCAL; // 启用接收器和本地连接

    tty.c_iflag &= ~(IXON | IXOFF | IXANY);         // 禁用软件流控制
    tty.c_iflag &= ~(ICANON | ECHO | ECHOE | ISIG); // 禁用输入处理
    tty.c_oflag &= ~OPOST;                          // 禁用输出处理

    if (tcsetattr(serialPort, TCSANOW, &tty) != 0)
    {
        std::cerr << "Error from tcsetattr" << std::endl;
        return -1;
    }

    return serialPort;
}


// 数据解析函数
void decodeAndPublishData(std::vector<unsigned char> &buffer, char portType, ros::Publisher &pub)
{
    while (buffer.size() >= 68)
    {
        if (buffer[0] == 0xFF && buffer[1] == 0xFE)
        {
            if (buffer.size() >= 66)
            {
                uint16_t frameNumber = buffer[2] | (buffer[3] << 8); // 小端格式
                
                // 定义要发送的 数据 msg
                std_msgs::UInt16MultiArray msg;
                msg.data.resize(32); // 32个通道的数据
                


                for (int i = 4; i < 68; i += 2)
                {
                    uint16_t channelData = buffer[i] | (buffer[i + 1] << 8); // 小端格式
                    msg.data[(i / 2) - 2] = channelData; //  把数据赋值给msg
                    // std::cout << "Channel " << (i / 2) - 2 << ": " << channelData << " ";
                }
                
                // 在消息的最后添加标识符
                if (portType == 'L')
                {
                    msg.data.push_back(1); // 如果是左串口，添加 1
                }
                else if (portType == 'R')
                {
                    msg.data.push_back(2); // 如果是右串口，添加 2
                }

                // 发布消息
                pub.publish(msg);

            // lfc注释掉了
                // 打印 左右通道，时间戳，帧号
                std::cout << (portType == 'L' ? "L: " : "R: ");
                printTimestamp();  //d打印时间戳
                std::cout << "Frame Number: " << frameNumber << std::endl;  // 打印帧数
                // 使用 ROS_INFO 打印 msg 的内容
                ROS_INFO("Published message:");
                for (size_t i = 0; i < msg.data.size(); ++i)
                {
                    ROS_INFO("  Channel %zu: %u", i, msg.data[i]);
                }

                // 移除已处理的数据包
                buffer.erase(buffer.begin(), buffer.begin() + 66);
            }
        }
        else
        {
            // 如果包头不匹配，移除一个字节并继续查找
            buffer.erase(buffer.begin());
        }
    }
}



// 串口读取函数
void readFromSerial(int serialPort, const char *portName, char portType, ros::Publisher &pub)
{
    std::vector<unsigned char> buffer;
    unsigned char receivedData[66]; // 数据包大小：2字节包头 + 2字节帧号 + 64字节数据 + 2字节包尾

    while (ros::ok())
    {
        // std::cout << "@@@@@@@@@@" <<serialPort << "@@@@@@@@@@"<< "@@@@@@@@@@" <<serialPort<< std::endl;//   lfc看线程挂没？6.10
        int bytesRead = read(serialPort, receivedData, sizeof(receivedData));
        if (bytesRead > 0)
        {
            // 将接收到的数据添加到缓存 
            buffer.insert(buffer.end(), receivedData, receivedData + bytesRead);

            // 尝试解析缓存中的数据包
            decodeAndPublishData(buffer, portType, pub);
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "serial_publisher");
    ros::NodeHandle nh;

    const char *port0 = "/dev/ttyACM0"; // ttyACM1  是左侧 0是右侧
    const char *port1 = "/dev/ttyACM1";

    // const char *port0 = "/dev/ttyACM2"; // ttyACM1  是左侧 0是右侧
    // const char *port1 = "/dev/ttyACM3";


    // 配置两个串口
    int serialPort0 = configureSerialPort(port0);

    if (serialPort0 == -1)
    {
        return -1;
    }
    int serialPort1 = configureSerialPort(port1);
    if (serialPort1 == -1)
    {

        return -1;
    }

    // 创建ROS发布者
    ros::Publisher pub0 = nh.advertise<std_msgs::UInt16MultiArray>("/hs", 10); // 话题名称是 hs
    ros::Publisher pub1 = nh.advertise<std_msgs::UInt16MultiArray>("/hs", 10);
    
    // std::cout << "运行到这里了 4" << std::endl; 

    // 发送一次数据 (0x06)
    unsigned char dataToSend = 0x06;
    write(serialPort0, &dataToSend, 1);
    write(serialPort1, &dataToSend, 1);
    std::cout << "Sent data: 0x06" << std::endl;
    std::cout << "开始接收鞋垫数据" << std::endl;


    // 创建线程来同时读取两个串口数据
    std::thread thread0(readFromSerial, serialPort0, port0, 'R', std::ref(pub0));
    std::thread thread1(readFromSerial, serialPort1, port1, 'L', std::ref(pub1));


    // 等待线程完成
    thread0.join();
    thread1.join();

    // 关闭串口
    close(serialPort0);
    close(serialPort1);
    // std::cout << "关闭关闭关闭关闭关闭关闭关闭关闭关闭关闭关闭关闭"<< std::endl;//   lfc看线程挂没？6.10


    return 0;
}






