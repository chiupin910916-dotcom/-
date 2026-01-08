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
    # find the cut-off point for non-zero singular values
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

# --- 這裡開始是補上的程式碼 ---

# 設定要使用多少個頻率成分 (n)
# n 越大，擬合越精確，但計算量越大。通常 n=10~20 效果就很明顯。
n = 5
m = len(x)

# step1: generate X=[1 cos(omega0 x) ... sin(omega0 nx)]
# X 的大小將會是 m x (2n + 1)
# 第一欄是常數項 (DC term)，接著是 n 個 cos 項，最後是 n 個 sin 項
X = np.zeros((m, 2 * n + 1))

X[:, 0] = 1.0  # Constant term (Bias)

for k in range(1, n + 1):
    # 填入 cos(k * omega0 * x)
    X[:, k] = np.cos(k * omega0 * x)
    # 填入 sin(k * omega0 * x)
    # 位置在 n + k
    X[:, n + k] = np.sin(k * omega0 * x)

# step2: SVD of X => X=USV^T
U, Sigma, V = mysvd(X)

# step3: a = U @ S^-1 @ V^T @ y (Pseudo-inverse solution)
# 公式推導: X a = y  =>  U Sigma V^T a = y
# => a = V * Sigma_inv * U^T * y
# 注意: mysvd 回傳的 Sigma 是對角矩陣，需使用 inv 求逆
Sigma_inv = la.inv(Sigma)
a = V @ Sigma_inv @ U.T @ y

# 計算預測值 y_bar
y_bar = X @ a

# --- 結束補上的程式碼 ---

plt.plot(x, y_bar, 'g-', label='predicted values') 
plt.plot(x, y, 'b-', label='true values')
plt.xlabel('x')
plt.ylabel('y') # 修正了原本程式碼重複的 xlabel
plt.legend()
plt.show()
