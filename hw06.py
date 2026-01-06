# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021
Modified for HW6 Solution
"""

from IPython import get_ipython
# 若在非 Spyder 環境執行可能會報錯，可用 try-except 包覆或註解掉
try:
    get_ipython().run_line_magic('reset', '-sf')
except:
    pass

import math
import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt
import pandas as pd


# calculate the eigenvalues and eigenvectors of a squared matrix
# the eigenvalues are decreasing ordered
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    # lambdas, V may contain complex value
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1] 
    return lambdas[sorted_idx], V[:, sorted_idx]


# class 1
mean1 = np.array([0, 5])
sigma1 = np.array([[0.3, 0.2],[0.2, 1]])
N1 = 200
X1 = np.random.multivariate_normal(mean1, sigma1, N1)

# class 2
mean2 = np.array([3, 4])
sigma2 = np.array([[0.3, 0.2],[0.2, 1]])
N2 = 100
X2 = np.random.multivariate_normal(mean2, sigma2, N2)

# m1: mean of class 1
# m2: mean of class 2
m1 = np.mean(X1, axis = 0, keepdims=1)
m2 = np.mean(X2, axis = 0, keepdims=1)

# --- 題目要求實作部分 (計算 w) ---

# 1. 計算 Within-class scatter matrix (Sw)
# Sw = Sum((X1 - m1)(X1 - m1)^T) + Sum((X2 - m2)(X2 - m2)^T)
# 使用矩陣乘法加速計算: (X - m).T @ (X - m)
S1 = (X1 - m1).T @ (X1 - m1)
S2 = (X2 - m2).T @ (X2 - m2)
Sw = S1 + S2

# 2. 計算最佳投影向量 w
# 根據 LDA 公式: w = Sw^-1 * (m1 - m2)
# m1, m2 為 (1, 2) 向量，相減後轉置為 (2, 1) 以進行矩陣運算
mean_diff = (m1 - m2).T
w = np.linalg.inv(Sw) @ mean_diff

# 將 w 正規化為單位向量 (方便繪圖與投影計算)
w = w / np.linalg.norm(w)

# 3. 計算投影點 (Projected Points)
# 純量投影值 y = X @ w
y1 = X1 @ w
y2 = X2 @ w

# 將純量轉換回 2D 空間的座標點 P = y @ w.T
# 這些點會落在通過原點且方向為 w 的直線上
P1 = y1 @ w.T
P2 = y2 @ w.T

# 設定一個位移量，讓投影線顯示在資料下方 (模仿題目範例圖)
shift = np.array([-1, -2]) 
P1_plot = P1 + shift
P2_plot = P2 + shift

# ------------------------------

plt.figure(dpi=144) # 稍微調整 dpi 讓顯示適中

# 繪製原始資料
plt.plot(X1[:, 0], X1[:,1], 'r.', label='Class 1')
plt.plot(X2[:, 0], X2[:,1], 'g.', label='Class 2')

# --- 題目要求實作部分 (繪製投影) ---

# 繪製投影後的點 (顏色對應類別)
plt.plot(P1_plot[:, 0], P1_plot[:, 1], 'r.', markersize=3, alpha=0.6)
plt.plot(P2_plot[:, 0], P2_plot[:, 1], 'g.', markersize=3, alpha=0.6)

# (選用) 為了美觀，可以畫出那條投影線
min_p = np.min(np.vstack((P1_plot, P2_plot)), axis=0)
max_p = np.max(np.vstack((P1_plot, P2_plot)), axis=0)
# 畫一條淡淡的黑線當底
plt.plot([min_p[0], max_p[0]], [min_p[1], max_p[1]], 'k-', linewidth=0.5, alpha=0.3)

# ----------------------------------

plt.title(f'LDA Projection (w=[{w[0,0]:.2f}, {w[1,0]:.2f}])')
plt.axis('equal')  
plt.legend()
plt.show()