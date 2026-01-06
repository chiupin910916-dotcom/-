# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 10:46:50 2021

@author: htchen
"""

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')
import math
import numpy as np
import numpy.linalg as la
#import cv2
import matplotlib.pyplot as plt
import pandas as pd
figdpi = 400

hw9_csv = pd.read_csv('hw9.csv').to_numpy(dtype = np.float64)
t = hw9_csv[:, 0] # 時間
flow_velocity = hw9_csv[:, 1] # 氣體流速
plt.figure(dpi=figdpi)
plt.plot(t, flow_velocity, 'r')
plt.title('Gas Flow Velocity')
plt.xlabel('time in seconds')
plt.ylabel('ml/sec')
plt.show()

# Integrating the gas flow velocity yields the net flow
net_vol = np.cumsum(flow_velocity) * 0.01
plt.figure(dpi=figdpi)
plt.plot(t, net_vol, 'r')
plt.title('Gas Net Flow')
plt.xlabel('time in seconds')
plt.ylabel('ml')
plt.show()

A = np.zeros((len(t), 3))
A[:, 0] = 1
A[:, 1] = t
A[:, 2] = t * t
y = net_vol
a = la.inv(A.T @ A) @ A.T @ y
trend_curve = a[0] + a[1] * t + a[2] * t * t

# write your code here
# find data trend line(找出資料趨勢線)
# 將 net_vol - trend_line 後做圖


# 1. 根據老師算出的係數 a，計算趨勢線 (Trend line)
# a[0] 是常數項, a[1] 是線性項, a[2] 是二次項
trend_line = a[0] + a[1] * t + a[2] * t**2

# 2. 移除趨勢：將原始積分結果減去趨勢線
corrected_net_vol = net_vol - trend_line

# 3. 繪製校正後的結果 (Gas Net Flow without drift)
plt.figure(dpi=figdpi)
plt.plot(t, corrected_net_vol, 'r')
plt.title('Gas Net Flow (Corrected)')
plt.xlabel('time in seconds')
plt.ylabel('ml')

# 為了美觀，建議加上一條 y=0 的水平線，方便觀察震盪
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.show()


