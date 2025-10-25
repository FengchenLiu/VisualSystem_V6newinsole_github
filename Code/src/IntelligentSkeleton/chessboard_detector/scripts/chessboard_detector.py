#!/usr/bin/env python3
import rospy
import message_filters
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import QuaternionStamped
from geometry_msgs.msg import PointStamped
import tf2_ros
import tf.transformations

from scipy.spatial.transform import Rotation

import traceback

def create_transform_matrix(position, quaternion):
    """
    根据位置和四元数创建4x4变换矩阵
    position: [x, y, z]
    quaternion: [x, y, z, w]
    """
    # 创建旋转矩阵
    rot_matrix = Rotation.from_quat(quaternion).as_matrix()
    
    # 创建4x4变换矩阵
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = position.flatten()
    
    return transform

def transform_point(point, T):
    """
    使用变换矩阵转换点
    point: [x, y, z]
    T: 4x4变换矩阵
    """
    # 转换为齐次坐标
    point_homogeneous = np.append(point, 1)
    # 应用变换
    transformed_point = T @ point_homogeneous
    # 返回3D坐标
    return transformed_point[:3]

def world_to_checkerboard(point_world, camera_in_world_pos, camera_in_world_quat,
                         checkerboard_in_camera_pos, checkerboard_in_camera_quat):
    """
    将世界坐标系中的点转换到棋盘格坐标系
    
    参数：
    point_world: 世界坐标系中的点 [x, y, z]
    camera_in_world_pos: 相机在世界坐标系中的位置 [x, y, z]
    camera_in_world_quat: 相机在世界坐标系中的旋转（四元数）[x, y, z, w]
    checkerboard_in_camera_pos: 棋盘格在相机坐标系中的位置 [x, y, z]
    checkerboard_in_camera_quat: 棋盘格在相机坐标系中的旋转（四元数）[x, y, z, w]
    """
    # 1. 创建世界到相机的变换矩阵
    T_world_to_camera = create_transform_matrix(camera_in_world_pos, camera_in_world_quat)
    
    # 2. 创建相机到棋盘格的变换矩阵
    T_camera_to_checkerboard = create_transform_matrix(
        checkerboard_in_camera_pos, 
        checkerboard_in_camera_quat
    )
    
    # 3. 计算世界到棋盘格的完整转换
    # 注意：需要求逆，因为我们给定的是相机在世界中的位姿
    T_world_to_camera_inv = np.linalg.inv(T_world_to_camera)
    
    # 注意：需要求逆，因为我们要从相机坐标系转换到棋盘格坐标系
    T_camera_to_checkerboard_inv = np.linalg.inv(T_camera_to_checkerboard)
    
    # 4. 组合变换
    T_world_to_checkerboard = T_camera_to_checkerboard_inv @ T_world_to_camera_inv
    
    # 5. 转换点
    point_checkerboard = transform_point(point_world, T_world_to_checkerboard)
    
    return point_checkerboard, T_world_to_checkerboard

class ChessboardDetector:
    def __init__(self):
        rospy.init_node('chessboard_detector')
        
        # 从参数服务器获取参数
        self.chess_rows = rospy.get_param('/chessboard_detector/chess_rows', 6)
        self.chess_cols = rospy.get_param('/chessboard_detector/chess_cols', 9)
        self.square_size = rospy.get_param('/chessboard_detector/square_size', 0.0254)
        self.image_topic = rospy.get_param('/chessboard_detector/image_topic', '/camera/rgb/image_raw')
        self.camera_frame = rospy.get_param('/chessboard_detector/camera_frame', 'camera_link')
        self.chessboard_frame = rospy.get_param('/chessboard_detector/chessboard_frame', 'chessboard')

        # 初始化相机参数
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False

        self.whole_T = None
        self.chessboard_flag = False

        # 初始化CV Bridge
        self.bridge = CvBridge()

        # 创建TF广播器
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # 订阅相机信息
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', 
                                              CameraInfo, 
                                              self.camera_info_callback)
        
        # 等待接收相机参数
        rospy.loginfo("Waiting for camera info...")
        while not self.camera_info_received and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # 订阅RGB图像
        self.image_sub = message_filters.Subscriber(self.image_topic, Image)
        self.quat_sub = message_filters.Subscriber('/vins_estimator/estimator/rgb_kf_quaternion', QuaternionStamped)
        self.vector_sub = message_filters.Subscriber("/vins_estimator/estimator/rgb_kf_position", PointStamped)

        sync = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.quat_sub, self.vector_sub],
            queue_size=10,
            slop=0.1
        )

        sync.registerCallback(self.image_callback)

        # 在初始化函数中添加publisher
        self.transform_pub = rospy.Publisher('chessboard_transform', TransformStamped, queue_size=10)
        self.position_pub = rospy.Publisher("rotated_position", PointStamped, queue_size=10)
        
        rospy.loginfo("Chessboard detector initialized")

    def camera_info_callback(self, msg):
        if not self.camera_info_received:
            # 从CameraInfo消息中提取相机矩阵和畸变系数
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D)
            self.camera_info_received = True
            rospy.loginfo("Camera parameters received")
            # 取消订阅，因为相机参数通常是静态的
            self.camera_info_sub.unregister()

    def image_callback(self, msg, quat_msg, pwc_msg):
        if not self.camera_info_received:
            rospy.loginfo("The camera is not initialized now!")
            return

        try:
            rospy.loginfo("Try to find chessboard now!")
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            quat = np.array([quat_msg.quaternion.x, quat_msg.quaternion.y, quat_msg.quaternion.z, quat_msg.quaternion.w])
            pwc = np.array([pwc_msg.point.x, pwc_msg.point.y, pwc_msg.point.z])

            ret, corners = cv2.findChessboardCorners(cv_image, 
                                                   (self.chess_cols, self.chess_rows), 
                                                   None)

            if ret:
                print("********** Find a chessboard **********")
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

                objp = np.zeros((self.chess_rows*self.chess_cols,3), np.float32)
                objp[:,:2] = np.mgrid[0:self.chess_cols,0:self.chess_rows].T.reshape(-1,2)
                objp = objp * self.square_size

                _, rvec, tvec = cv2.solvePnP(objp, corners2, 
                                            self.camera_matrix,
                                            self.dist_coeffs)

                transform = TransformStamped()
                transform.header.stamp = msg.header.stamp  # 使用图像的时间戳
                transform.header.frame_id = self.camera_frame
                transform.child_frame_id = self.chessboard_frame

                transform.transform.translation.x = tvec[0]
                transform.transform.translation.y = tvec[1]
                transform.transform.translation.z = tvec[2]

                rotation_matrix, _ = cv2.Rodrigues(rvec)
                T = np.eye(4)
                T[:3, :3] = rotation_matrix
                T[:3, 3] = tvec.reshape(-1)

                # 转换为右手系
                conversion = np.eye(4)
                conversion[1, 1] = -1  # Y轴翻转
                T_rhs = T @ conversion

                print("T: \n", T)
                print("conversion: \n", conversion)
                print("T_rhs: \n", T_rhs)

                # rotation_matrix_4x4 = np.eye(4)
                # rotation_matrix_4x4[:3, :3] = rotation_matrix
                quat = tf.transformations.quaternion_from_matrix(T_rhs)

                transform.transform.rotation.x = quat[0]
                transform.transform.rotation.y = quat[1]
                transform.transform.rotation.z = quat[2]
                transform.transform.rotation.w = quat[3]

                transform.transform.translation.y = - transform.transform.translation.y

                camera_in_world_pos = pwc
                camera_in_world_quat = quat

                checkerboard_in_camera_pos = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
                checkerboard_in_camera_quat = np.array([transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w])

                pos_T, whole_T = world_to_checkerboard(camera_in_world_pos, 
                                                       camera_in_world_pos,
                                                       camera_in_world_quat,
                                                       checkerboard_in_camera_pos,
                                                       checkerboard_in_camera_quat)

                self.whole_T = whole_T
                self.chessboard_flag = True

                transform_2 = TransformStamped()
                transform_2.header.stamp = msg.header.stamp  # 使用图像的时间戳
                transform.header.frame_id = self.camera_frame
                transform.child_frame_id = self.chessboard_frame

                rotated_position = PointStamped()
                rotated_position.header.stamp = msg.header.stamp
                rotated_position.header.frame_id = self.camera_frame
                rotated_position.point.x = pos_T[0]
                rotated_position.point.y = pos_T[1]
                rotated_position.point.z = pos_T[2]

                # self.tf_broadcaster.sendTransform(transform)
                self.transform_pub.publish(transform)
                self.position_pub.publish(rotated_position)
                rospy.loginfo_throttle(1, "Chessboard detected and transform published")
            else:
                print("********** chessboard not find **********")
                if self.chessboard_flag:
                    pos_T = transform_point(pwc, self.whole_T)

                    rotated_position = PointStamped()
                    rotated_position.header.stamp = msg.header.stamp
                    rotated_position.header.frame_id = self.camera_frame
                    rotated_position.point.x = pos_T[0]
                    rotated_position.point.y = pos_T[1]
                    rotated_position.point.z = pos_T[2]

                    self.position_pub.publish(rotated_position)


        except Exception as e:
            error_msg = traceback.format_exc()
            print(error_msg)
            rospy.logerr(f"Error processing image: {str(e)}")

if __name__ == '__main__':
    try:
        detector = ChessboardDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass