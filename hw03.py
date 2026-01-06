# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:38:53 2024

@author: htchen
"""

import numpy as np
import numpy.linalg as la
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

# SVD: A = U * Sigma * V^T
# V: eigenvector matrix of A^T * A; U: eigenvector matrix of A * A^T 
def mysvd(A):
    lambdas, V = myeig(A.T @ A, symmetric=True)
    lambdas, V = np.real(lambdas), np.real(V)
    # if A is full rank, no lambda value is less than 1e-6 
    # append a small value to stop rank check
    lambdas = np.append(lambdas, 1e-12)
    rank = np.argwhere(lambdas < 1e-6).min()
    lambdas, V = lambdas[0:rank], V[:, 0:rank]
    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))
    return U, Sigma, V


pts = 50
x = np.linspace(-2, 2, pts)
y = np.zeros(x.shape)

# square wave
pts2 = pts // 2
y[0:pts2] = -1
y[pts2:] = 1

# sort x
argidx = np.argsort(x)
x = x[argidx]
y = y[argidx]

T0 = np.max(x) - np.min(x)
f0 = 1.0 / T0
omega0 = 2.0 * np.pi * f0

# step1: generate X=[1 cos(omega0 x) cos(omega0 2x) ... cos(omega0 nx) sin(omega0 x) sin(omega0 2x) ... sin(omega0 nx)]
# step2: SVD of X => X=USV^T
# step3: a = U @ S^-1 @ V^T @ y
# write your code here
# n = 5 表示我們要取到第 5 個諧波
n = 5

# --- Step 1: 建立矩陣 X ---
# X 的每一列代表一個樣本，每一欄代表一個基底函數
# 欄位的順序：[1, cos(1w0x), cos(2w0x)...cos(nw0x), sin(1w0x), sin(2w0x)...sin(nw0x)]
X = np.zeros((pts, 2 * n + 1))

X[:, 0] = 1.0  # 第一欄是常數項 a0

for k in range(1, n + 1):
    X[:, k] = np.cos(k * omega0 * x)      # 餘弦項 a1 ~ an
    X[:, k + n] = np.sin(k * omega0 * x)  # 正弦項 b1 ~ bn

# --- Step 2: 對 X 進行 SVD 分解 ---
# 這裡使用你寫好的 mysvd 函數
U, Sigma, V = mysvd(X)

# --- Step 3: 計算係數 a ---
# 根據題目公式：a = V @ inv(Sigma) @ U.T @ y
# 注意：在矩陣運算中，a = V @ Sigma^-1 @ U^T @ y
# 因為 Sigma 是對角矩陣，Sigma^-1 就是對角線元素取倒數
inv_Sigma = np.linalg.inv(Sigma)
a = V @ inv_Sigma @ U.T @ y

y_bar = X @ a

plt.plot(x, y_bar, 'g-')

plt.plot(x, y, 'b-')

plt.xlabel('x')

plt.xlabel('y')

plt.show()


