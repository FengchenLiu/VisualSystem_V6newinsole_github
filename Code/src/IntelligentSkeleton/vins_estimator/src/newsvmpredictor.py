#!/usr/bin/env python3
import rospy
import numpy as np
import os
import joblib
from rospkg import RosPack
from std_msgs.msg import Float32MultiArray, Int32

import time
# from datetime import datetime
from datetime import datetime, timezone

class SVMPredictor:
    def __init__(self):
        # 初始化ROS节点
        # rospy.init_node('svm_predictor', anonymous=True)
        rospy.init_node('newsvmpredictor', anonymous=True)
        
        # 构建模型文件路径
        # 获取包路径
        rp = RosPack()
        try:
            pkg_path = rp.get_path('vins_estimator')  # 请确认实际包名
        except:
            # rospy.logerr("Package 'realsense2_camera' not found!")
            rospy.logerr("Package 'vins_estimator' not found!")
            rospy.signal_shutdown("Package not found")
            return      
        model_dir = os.path.join(pkg_path, "src")
        
        # 直接写 绝对路径
        # model_dir ="/home/lfc/lfc/VisualSystem_V6newinsole/Code/src/IntelligentSkeleton/vins_estimator/record_csv_LFC"  
        self.model_path = os.path.join(model_dir, "svm_model.pkl")
        self.scaler_path = os.path.join(model_dir, "scaler.pkl")

        # # lfc 5.5 ========== 新增日志文件路径 ==========
        # self.log_file_path = os.path.join(model_dir, "new_svm_predictions.log")
        # lfc 5.5 ========== 新增日志文件路径（带时间戳） ==========
        timename = datetime.now().strftime("%m%d-%H-%M-%S")  
        self.log_file_path = os.path.join(model_dir, f"new_svm_predictions_{timename}.csv")
        # lfc 5.5 ========== 初始化日志文件（写入标题行） ==========
        with open(self.log_file_path, 'w') as f:
            f.write("predict_TimeStamp,count_hs,heel_strike,side_strike,left_state,right_state,Press_L,Press_R,human_height,stair_height,distance,stride_size,avg_velocity,prediction,number_of_plane,hs_TS_Nsec,LM,\n")


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
        # self.pred_pub = rospy.Publisher("/prediction", Int32, queue_size=10)
        rospy.loginfo("Python节点已启动，等待SVM数据...")


            # 添加模态相关变量
        self.current_mode = 1  # 初始模式：平地行走
        self.hscount_after_predict = 0  # 预测后的足跟触地计数
        self.prev_hs_state = 0  # 上一次的足跟触地状态
        self.count_hs=0      #足跟hs计数

    def svm_callback(self, msg):
        try:
            # 验证输入数据
            # 6.6新加
            if len(msg.data) != 14:  # 
                rospy.logwarn(f"无效数据长度: 期望14个字段（5特征+4个hs标志位+2个压力+2*hs时间戳），收到{len(msg.data)}")
                return

            # 提取特征并转换为numpy数组
            features = np.array(msg.data).reshape(1, -1)
            
            # # 打印原始数据
            # rospy.loginfo("\n===== 接收特征数据 =====")
            # rospy.loginfo("\n===== 接收特徵數據 [%s] =====", current_time)
            # rospy.loginfo("人体高度: %.3f m", features[0, 0])
            # rospy.loginfo("台阶高度: %.3f cm", features[0, 1])
            # rospy.loginfo("下一地形距离: %.3f mm", features[0, 2])
            # rospy.loginfo("步长: %.3f mm", features[0, 3])
            # rospy.loginfo("平均速度: %.3f mm/s", features[0, 4])

            # 选择前5个特征
            features_to_scale = features[:, :5]  # 假设 features 是一个 2D 数组，每行是一个样本，每列是一个特征

            # 检查第2个特征是否小于0，或者第3个特征是否等于2500
            if features[0, 1] < 3 or features[0, 1] > 50 or features[0, 2] == 2500 or features[0, 3] < 500:
                prediction = np.array([0.0])  # 统一为numpy数组格式
                # 程序运行当前时间
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f") #到毫秒
            else:
                # 标准化这些特征
                scaled_features = self.scaler.transform(features_to_scale)
                # 模型预测
                prediction = self.svm_model.predict(scaled_features)
                # 程序运行当前时间
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")  #到毫秒

                # self.pred_pub.publish(prediction[0])   发布结果

            rospy.loginfo("!!!预测结果: %d", prediction[0])


                        # 提取事件时间戳（秒和微秒）
            event_sec = features[0, 11]   # 秒（整数部分）
            event_usec = features[0, 12]  # 微秒部分
            
            # 将秒和微秒组合成完整时间戳
            event_timestamp = event_sec + event_usec * 1e-6
            
            # 使用本地时区转换时间戳（不指定时区参数）
            event_dt = datetime.fromtimestamp(event_timestamp)  # 移除 tz=timezone.utc
            event_time_str = event_dt.strftime("%m-%d %H:%M:%S") + f".{event_dt.microsecond:06d}"

            
            if features[0,5]==1:
                self.count_hs+=1

            # 统一的日志写入逻辑
            with open(self.log_file_path, 'a') as f:
                f.write(f"{current_time},")
                f.write(f"{self.count_hs},")
                f.write(f"{int(features[0,5])},")                # heel_strike  是不是触底了
                f.write(f"{int(features[0,6])},")                # side_strike    哪一侧触底了
                f.write(f"{int(features[0,7])},")                # 左脚状态
                f.write(f"{int(features[0,8])},")                # 右脚状态
                f.write(f"{int(features[0,9])},")                # 左脚压力
                f.write(f"{int(features[0,10])},")               # 右脚压力
                f.write(f"{features[0,0]:.3f},{features[0,1]:.3f},")  # 前5个特征
                f.write(f"{features[0,2]:.3f},{features[0,3]:.3f},{features[0,4]:.3f},")
                f.write(f"{prediction[0]},")                        # 预测结果
                f.write(f"{int(features[0,13])},")                  # 平面数量
                f.write(f"{event_time_str},")                        # 预测结果

            #新加入模态label
            # AllMode=1 ; Label 1: 平地行走  ,2: 过渡1  ,3: 上楼梯   4:过渡2   1:平地行走.    
            # AllMode=2 ; Label 1: 平地行走  ,2: 过渡3  ,3: 下楼梯   4:过渡4   1:平地行走.  

                            # 提取需要的特征
                hs_detected = int(features[0, 5])  # 足跟触地标志
                num_planes = int(features[0, 13])  # 平面数量
                
                # 模态判断逻辑
                # 模态说明: 
                # Mode=1: 平地行走
                # Mode=2: 过渡1 (平地->上楼梯) 或 过渡3 (平地->下楼梯)
                # Mode=3: 上楼梯或下楼梯
                # Mode=4: 过渡2 (上楼梯->平地) 或 过渡4 (下楼梯->平地)
                
                # 检测足跟触地的新事件(从0变为1)
                new_hs_event = (hs_detected == 1 and self.prev_hs_state == 0)
                
                # 更新前一状态
                self.prev_hs_state = hs_detected
                
                # 状态机逻辑
                if self.current_mode == 1:  # 当前是平地行走
                    if prediction[0] == 1.0 and new_hs_event:
                        # 检测到楼梯，且有新的足跟触地事件
                        self.current_mode = 2  # 进入过渡状态
                        self.hscount_after_predict = 0 # 记录第一次足跟触地

                        
                elif self.current_mode == 2:  # 当前是过渡状态
                    if new_hs_event:
                        self.hscount_after_predict += 1
                        if self.hscount_after_predict == 2:
                            self.current_mode = 3  # 进入楼梯状态
                            self.hscount_after_predict = 0  # 重置计数

                            
                elif self.current_mode == 3:  # 当前是楼梯状态 
                    if new_hs_event:
                        self.hscount_after_predict += 1
                    if self.hscount_after_predict>=3:           # 在楼梯上至少发生了3次足跟触地
                        if num_planes == 3 and new_hs_event:  # 平面数量减少，可能接近地
                            self.current_mode = 4  # 进入第二个过渡状态
                            self.hscount_after_predict = 0  # 记录第一次足跟触地

                        
                elif self.current_mode == 4:  # 当前是第二个过渡状态
                    if new_hs_event:
                        self.hscount_after_predict += 1
                        if self.hscount_after_predict == 2:
                            self.current_mode = 1  # 回到平地行走
                            self.hscount_after_predict = 0  # 重置计数


                f.write(f"{self.current_mode},,\n")                    # 当前模式
            

        except Exception as e:
            rospy.logerr(f"处理数据时发生错误 lfc: {str(e)}")

if __name__ == '__main__':
    try:
        predictor = SVMPredictor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass



