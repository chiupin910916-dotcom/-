# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""

# If this script is not run under spyder IDE, comment the following two lines.
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd

def scatter_pts_2d(x, y):
    # set plotting limits
    xmax = np.max(x)
    xmin = np.min(x)
    xgap = (xmax - xmin) * 0.2
    xmin -= xgap
    xmax += xgap

    ymax = np.max(y)
    ymin = np.min(y)
    ygap = (ymax - ymin) * 0.2
    ymin -= ygap
    ymax += ygap 

    return xmin,xmax,ymin,ymax

dataset = pd.read_csv(r"C:\Users\chiupin\Desktop\研究所\資訊專題\data\hw7.csv").to_numpy(dtype = np.float64)
x = dataset[:, 0]
y = dataset[:, 1]

# parameters for our two runs of gradient descent
w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])

alpha = 0.05
max_iters = 500
# cost function
#     J(w0, w1, w2, w3) = sum(y[i] - w0 - w1 * sin(w2 * x[i] + w3))^2
for _ in range(1, max_iters):

    y_pred = w[0] + w[1] * np.sin(w[2] * x + w[3])
    e = y - y_pred
    
    # 根據公式計算梯度 (Partial Derivatives)
    dw0 = -np.sum(2 * e)
    dw1 = -np.sum(2 * e * np.sin(w[2] * x + w[3]))
    dw2 = -np.sum(2 * e * x * w[1] * np.cos(w[2] * x + w[3]))
    dw3 = -np.sum(2 * e * w[1] * np.cos(w[2] * x + w[3]))
    
    gradient_of_cost = np.array([dw0, dw1, dw2, dw3])
    
    # 更新權重
    w = w - alpha * gradient_of_cost



xmin,xmax,ymin,ymax = scatter_pts_2d(x, y)
xt = np.linspace(xmin, xmax, 100)
yt1 = w[0] + w[1] * np.sin(w[2] * xt + w[3])

w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])
# --- 第二段：數值法 ---
eps = 1e-8 # 設定微小變化量
for _ in range(1, max_iters):
    # 先計算目前的 Cost
    y_p = w[0] + w[1] * np.sin(w[2] * x + w[3])
    current_cost = np.sum((y - y_p)**2)
    
    grad = np.zeros(4)
    # 分別對 w0, w1, w2, w3 加上 eps 來計算變化率
    for k in range(4):
        w_plus = w.copy()
        w_plus[k] += eps
        
        # 計算 J(w + eps)
        y_p_plus = w_plus[0] + w_plus[1] * np.sin(w_plus[2] * x + w_plus[3])
        cost_plus = np.sum((y - y_p_plus)**2)
        
        # 數值梯度公式: [J(w + eps) - J(w)] / eps
        grad[k] = (cost_plus - current_cost) / eps
    
    # 更新權重
    w = w - alpha * grad
    

xt = np.linspace(xmin, xmax, 100)
yt2 = w[0] + w[1] * np.sin(w[2] * xt + w[3])

# plot x vs y; xt vs yt1; xt vs yt2 
fig = plt.figure(dpi=288)
plt.scatter(x, y, color='k', edgecolor='w', linewidth=0.9, s=60, zorder=3)
plt.plot(xt, yt1, linewidth=4, c='b', zorder=0, label='Analytic method')
plt.plot(xt, yt2, linewidth=2, c='r', zorder=0, label='Numeric method')
plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()
