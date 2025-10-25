import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime
from matplotlib.ticker import MultipleLocator

# 清除变量和关闭所有图形窗口 (Python不需要clear all和clc)
plt.close('all')

# # 设置文件路径和文件名
file_path = '/home/lfc/lfc/VisualSystem_V6newinsole/Code/src/IntelligentSkeleton/vins_estimator/src'
# # file_name = 'new_svm_predictions_0923-16-05-01.csv'# 室内 一楼 下楼梯

# file_name = 'new_svm_predictions_1008-18-17-47.csv'     #7

# # file_name = 'new_svm_predictions_0926-18-27-02.csv'

# full_path = os.path.join(file_path, file_name)

# 获取最新的CSV文件
def get_latest_csv_file(directory):
    # 查找所有符合模式的CSV文件
    pattern = os.path.join(directory, 'new_svm_predictions_*.csv')
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        raise FileNotFoundError("No CSV files matching the pattern found in directory")
    
    # 从文件名解析日期时间
    def parse_datetime(filename):
        # 提取日期时间部分 (例如 '1008-18-17-47')
        basename = os.path.basename(filename)
        date_part = basename.replace('new_svm_predictions_', '').replace('.csv', '')
        
        # 解析日期时间
        try:
            # 假设格式为 'MMDD-HH-MM-SS'
            return datetime.strptime(date_part, '%m%d-%H-%M-%S')
        except ValueError:
            # 如果解析失败，返回一个很早的日期
            return datetime(1900, 1, 1)
    
    # 按日期时间排序并返回最新的文件
    latest_file = max(csv_files, key=parse_datetime)
    print(f"Latest CSV file: {os.path.basename(latest_file)}")
    return latest_file

# 获取最新的CSV文件路径
full_path = get_latest_csv_file(file_path)
file_name = os.path.basename(full_path)


# 读取CSV文件数据
try:
    # 使用pandas读取CSV文件
    data = pd.read_csv(full_path)
    # 如果CSV文件没有标题，使用以下方式
    # data = pd.read_csv(full_path, header=None)
    print('Successfully read the CSV file.')
except Exception as e:
    print(f'Failed to read the CSV file. Error: {e}')
    raise

# 找出count_hs等于特定值的第一帧位置
n1 = 10  # 设置需要标记的count_hs值
# n2 = 7
# n3 = 8

# 定义列索引（根据实际数据调整
left_state_col = 4  #3           #这里 3和 4 互换一下，因为 左右脚的 state好像 反了  
right_state_col = 3   #4
distance_col = 10-1
stair_height_col = 9-1
number_of_plane_col = 14-1
LM_col = 16-1
count_hs_col = 1-1
Press_L_col = 6-1
Press_R_col = 7-1

# 提取列数据 (假设数据是数值型数组)
# 左右侧数据似乎是反的，按照原代码的逻辑提取
right_state = data.iloc[:, left_state_col].values
left_state = data.iloc[:, right_state_col].values
distance = data.iloc[:, distance_col].values
stair_height = data.iloc[:, stair_height_col].values
number_of_plane = data.iloc[:, number_of_plane_col].values
LM = data.iloc[:, LM_col].values
count_hs = data.iloc[:, count_hs_col].values
Press_L = data.iloc[:, Press_L_col].values
Press_R = data.iloc[:, Press_R_col].values

# 找出count_hs等于特定值的第一帧位置
hs_marker1 = np.where(count_hs == n1)[0][0] if np.any(count_hs == n1) else None
# hs_marker2 = np.where(count_hs == n2)[0][0] if np.any(count_hs == n2) else None
# hs_marker3 = np.where(count_hs == n3)[0][0] if np.any(count_hs == n3) else None

# 基于100Hz采样率创建时间轴
num_samples = len(left_state)
time_axis = np.arange(num_samples) / 100  # 100Hz采样率，转换为秒

# 设置matplotlib样式为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 创建图形窗口 - 现在有4个子图
fig = plt.figure(figsize=(10, 12))  # 增加高度以容纳4个子图

# 第一个子图：Press_L和Press_R
ax1 = plt.subplot(4, 1, 1)
ax1.plot(time_axis, Press_L, 'b-', linewidth=1.5, label='Press_L')
ax1.plot(time_axis, Press_R, 'r-', linewidth=1.5, label='Press_R')

# 在count_hs=n的第一帧位置添加竖线
if hs_marker1 is not None:
    ax1.axvline(x=time_axis[hs_marker1], color='k', linestyle='--', linewidth=1)
# if hs_marker2 is not None:
#     ax1.axvline(x=time_axis[hs_marker2], color='k', linestyle='--', linewidth=1)
# if hs_marker3 is not None:
#     ax1.axvline(x=time_axis[hs_marker3], color='k', linestyle='--', linewidth=1)

ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Pressure', fontsize=12)
ax1.set_title('Left and Right Pressure', fontsize=14)
ax1.legend(loc='best')
ax1.grid(True)
ax1.set_xlim(min(time_axis), max(time_axis))
ax1.set_ylim( -50, 1600)

# 第二个子图：left_state和right_state
ax2 = plt.subplot(4, 1, 2)
ax2.plot(time_axis, left_state, 'b-', linewidth=1.5, label='Left State')
ax2.plot(time_axis, right_state, 'r-', linewidth=1.5, label='Right State')

# 在count_hs=n的第一帧位置添加竖线
if hs_marker1 is not None:
    ax2.axvline(x=time_axis[hs_marker1], color='k', linestyle='--', linewidth=1)
# if hs_marker2 is not None:
#     ax2.axvline(x=time_axis[hs_marker2], color='k', linestyle='--', linewidth=1)
# if hs_marker3 is not None:
#     ax2.axvline(x=time_axis[hs_marker3], color='k', linestyle='--', linewidth=1)

ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('State', fontsize=12)
ax2.set_title('Left and Right State', fontsize=14)
ax2.legend(loc='best')
ax2.grid(True)
ax2.set_xlim(min(time_axis), max(time_axis))
ax2.set_ylim(8, 25)

# 第三个子图：distance和stair_height
ax3 = plt.subplot(4, 1, 3)
ax3_twin = ax3.twinx()  # 创建双Y轴

# 左Y轴: distance
ax3.plot(time_axis, distance, 'b-', linewidth=1.5, label='Distance')
ax3.set_ylabel('Distance (mm)', fontsize=12, color='b')
ax3.set_ylim(min(distance) - 10, 2700)
ax3.tick_params(axis='y', colors='b')

# 右Y轴: stair_height
ax3_twin.plot(time_axis, stair_height, 'r-', linewidth=1.5, label='Stair Height')
ax3_twin.set_ylabel('Stair Height (cm)', fontsize=12, color='r')
ax3_twin.set_ylim(min(stair_height) - 1, 50)
ax3_twin.tick_params(axis='y', colors='r')

# 添加竖线
if hs_marker1 is not None:
    ax3.axvline(x=time_axis[hs_marker1], color='k', linestyle='--', linewidth=1)
# if hs_marker2 is not None:
#     ax3.axvline(x=time_axis[hs_marker2], color='k', linestyle='--', linewidth=1)
# if hs_marker3 is not None:
#     ax3.axvline(x=time_axis[hs_marker3], color='k', linestyle='--', linewidth=1)

ax3.set_xlabel('Time (s)', fontsize=12)
ax3.set_title('Distance and Stair Height', fontsize=14)
ax3.grid(True)
ax3.set_xlim(min(time_axis), max(time_axis))

# 合并两个Y轴的图例
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='best')

# 第四个子图：number_of_plane和LM
ax4 = plt.subplot(4, 1, 4)
ax4_twin = ax4.twinx()  # 创建双Y轴

# 左Y轴: number_of_plane
ax4.plot(time_axis, number_of_plane, 'b-', linewidth=1.5, label='Number of Plane')
ax4.set_ylabel('Number of Plane', fontsize=12, color='b')
ax4.set_ylim(0, 9)
ax4.yaxis.set_major_locator(MultipleLocator(3))  # 设置y轴刻度间隔为3
ax4.tick_params(axis='y', colors='b')

# 右Y轴: LM
ax4_twin.plot(time_axis, LM, 'r-', linewidth=1.5, label='Locomotion Mode')
ax4_twin.set_ylabel('Locomotion Mode', fontsize=12, color='r')
ax4_twin.set_ylim(0, 5)
ax4_twin.tick_params(axis='y', colors='r')

# 添加竖线
if hs_marker1 is not None:
    ax4.axvline(x=time_axis[hs_marker1], color='k', linestyle='--', linewidth=1)
# if hs_marker2 is not None:
#     ax4.axvline(x=time_axis[hs_marker2], color='k', linestyle='--', linewidth=1)
# if hs_marker3 is not None:
#     ax4.axvline(x=time_axis[hs_marker3], color='k', linestyle='--', linewidth=1)

ax4.set_xlabel('Time (s)', fontsize=12)
ax4.set_title('Number of Plane and Locomotion Mode', fontsize=14)
ax4.grid(True)
ax4.set_xlim(min(time_axis), max(time_axis))

# 合并两个Y轴的图例
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='best')

# 添加总标题
plt.suptitle('Data Analysis', fontsize=16)

# 调整子图之间的间距
plt.tight_layout(rect=[0, 0, 1, 0.95])  # rect参数留出顶部空间给suptitle

# 在图表右下角添加文件名
plt.figtext(0.98, 0.01, file_name, ha='right', fontsize=10)

# 显示图形
plt.show()

# 可选：保存图形
# plt.savefig(os.path.join(file_path, 'data_analysis_plot.png'))
print('Figures plotted successfully.')