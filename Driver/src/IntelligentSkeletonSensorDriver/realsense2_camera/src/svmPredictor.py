#!/usr/bin/env python
import rospy
import numpy as np
import os
import joblib
from rospkg import RosPack
from std_msgs.msg import Float32MultiArray, Int32

import time
from datetime import datetime


class SVMPredictor:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('svm_predictor', anonymous=True)
        
        # 初始化计数器
        self.prediction_count = 0

        # 获取包路径
        rp = RosPack()
        try:
            pkg_path = rp.get_path('realsense2_camera')  # 请确认实际包名
        except:
            rospy.logerr("Package 'realsense2_camera' not found!")
            rospy.signal_shutdown("Package not found")
            return

        # 构建模型文件绝对路径
        model_dir = os.path.join(pkg_path, "src")
        self.model_path = os.path.join(model_dir, "svm_model.pkl")
        self.scaler_path = os.path.join(model_dir, "scaler.pkl")

        # lfc 5.5 ========== 新增日志文件路径 ==========
        self.log_file_path = os.path.join(model_dir, "svm_predictions.log")
        # lfc 5.5 ========== 初始化日志文件（写入标题行） ==========
        with open(self.log_file_path, 'w') as f:
            f.write("timestamp,count,human_height,stair_height,distance,avg_velocity,stride_size,prediction\n")


        # 加载模型和标准化器
        try:
            self.svm_model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            rospy.loginfo("成功加载SVM模型和标准化器")
        except Exception as e:
            rospy.logerr(f"加载模型失败: {str(e)}")
            rospy.signal_shutdown("Model loading failed")
            return

        # 初始化发布/订阅
        self.sub = rospy.Subscriber("/svmneed", Float32MultiArray, self.svm_callback)
        self.pred_pub = rospy.Publisher("/prediction", Int32, queue_size=10)
        rospy.loginfo("Python节点已启动，等待SVM数据...")

    def svm_callback(self, msg):
        try:
            # 验证输入数据
            if len(msg.data) != 5:
                rospy.logwarn(f"无效数据长度: 期望5个特征，1計數，收到{len(msg.data)}")
                return


            # 獲取當前時間戳
            # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  #到毫秒

            # 提取特征并转换为numpy数组
            features = np.array(msg.data).reshape(1, -1)
            
            # 打印原始数据
            rospy.loginfo("\n===== 接收特征数据 =====")
            rospy.loginfo("\n===== 接收特徵數據 [%s] =====", current_time)
            rospy.loginfo("人体高度: %.3f m", features[0, 0])
            rospy.loginfo("台阶高度: %.3f cm", features[0, 1])
            rospy.loginfo("下一地形距离: %.3f mm", features[0, 2])
            rospy.loginfo("平均速度: %.3f mm/s", features[0, 3])
            rospy.loginfo("步长: %.3f mm", features[0, 4])


            # 数据标准化
            scaled_features = self.scaler.transform(features)
            
            # 模型预测
            prediction = self.svm_model.predict(scaled_features)
            rospy.loginfo("!!!预测结果: %d", prediction[0])

            # 发布预测结果
            self.prediction_count += 1
            rospy.loginfo(">>>> 第 %d 次預測 <<<<", self.prediction_count)
            self.pred_pub.publish(prediction[0])
            # lfc 5.5 保存到csv
            with open(self.log_file_path, 'a') as f:
                f.write(f"{current_time},{self.prediction_count},{features[0,0]:.3f},{features[0,1]:.3f},")
                f.write(f"{features[0,2]:.3f},{features[0,3]:.3f},{features[0,4]:.3f},{prediction[0]}\n")


        except Exception as e:
            rospy.logerr(f"处理数据时发生错误: {str(e)}")

if __name__ == '__main__':
    try:
        predictor = SVMPredictor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

    



