# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021
Modified for HW8 Solution
"""
# If this script is not run under spyder IDE, comment the following two lines.
try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-sf')
except:
    pass

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd
import os

# 引入 sklearn 用於分類
from sklearn.svm import SVC

# --- 1. 讀取或生成數據 ---
file_path = 'C:/Users/chiupin/Desktop/研究所/資訊專題/data/hw8(in).csv'
if os.path.exists(file_path):
    hw8_csv = pd.read_csv(file_path)
    hw8_dataset = hw8_csv.to_numpy(dtype=np.float64)
    X0 = hw8_dataset[:, 0:2]
    y = hw8_dataset[:, 2]
else:
    print("找不到 hw8.csv，正在生成模擬的非線性數據...")
    # 生成類似題目圖形的模擬數據 (類似花瓣/風車形狀)
    np.random.seed(0)
    N = 400
    # 產生螺旋或花瓣狀數據
    theta = np.sqrt(np.random.rand(N)) * 4 * np.pi 
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T + np.random.randn(N, 2)
    
    # 這裡簡單使用 sklearn 生成類似的非線性數據代替
    from sklearn.datasets import make_moons
    X0, y_temp = make_moons(n_samples=400, noise=0.2, random_state=0)
    # 將標籤轉為 1 和 -1 以符合你的程式碼習慣
    y = np.where(y_temp == 0, -1, 1)
    # 放大一點以符合圖表比例
    X0 = X0 * 5 

# --- 2. 建立與訓練模型 (Write your code here) ---
# 使用 SVM 搭配 RBF Kernel (非線性分類的強大工具)
# C: 懲罰係數，越大容錯率越低 (邊界越複雜)
# gamma: 控制 RBF 核的影響範圍
clf = SVC(kernel='rbf', C=10, gamma=0.5)
clf.fit(X0, y)

# --- 3. 繪圖設定 ---
fig = plt.figure(dpi=144) # 調整 DPI 以便觀看

# --- 4. 繪製分類邊界 (Write your code here) ---
# 建立網格 (Meshgrid) 來覆蓋整個數據範圍
x_min, x_max = X0[:, 0].min() - 1, X0[:, 0].max() + 1
y_min, y_max = X0[:, 1].min() - 1, X0[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

# 預測網格中每一點的類別
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 使用 contourf 繪製背景顏色區塊 (決策邊界)
# alpha 設定透明度，讓資料點可以透出來
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)

# --- 5. 繪製原始資料點 (User's code) ---
plt.plot(X0[y == 1, 0], X0[y == 1, 1], 'r.', label='$\omega_1$')
plt.plot(X0[y == -1, 0], X0[y == -1, 1], 'b.', label='$\omega_2$')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')
plt.legend()
plt.title('Non-linear SVM Classification')
plt.show()