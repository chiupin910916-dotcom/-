import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 畫圖範圍設定輔助函式
def scatter_pts_2d(x, y):
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
    return xmin, xmax, ymin, ymax

# 讀取資料
# 請確認 'hw7.csv' 在你的工作目錄中
try:
    dataset = pd.read_csv('hw7.csv').to_numpy(dtype=np.float64)
except FileNotFoundError:
    dataset = pd.read_csv('data/hw7.csv').to_numpy(dtype=np.float64)

x = dataset[:, 0]
y = dataset[:, 1]

# ---------------------------------------------------------
# 第一部分：解析法 (Analytic Gradient Descent)
# 使用偏導數公式直接計算
# ---------------------------------------------------------
print("開始執行：解析法 (Analytic Method)")
w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])
alpha = 0.05
max_iters = 500

for _ in range(1, max_iters):
    # 預測值 y_pred
    # y_pred = w0 + w1 * sin(w2 * x + w3)
    arg = w[2] * x + w[3]
    y_pred = w[0] + w[1] * np.sin(arg)
    
    # 誤差 e = y - y_pred
    e = y - y_pred
    
    # 計算梯度 (Partial Derivatives)
    # J = sum(e^2) -> dJ/dw = sum(2*e * de/dw) = sum(-2*e * dy_pred/dw)
    
    # dJ/dw0 = sum(-2 * e * 1)
    grad_0 = -np.sum(2 * e)
    
    # dJ/dw1 = sum(-2 * e * sin(w2*x + w3))
    grad_1 = -np.sum(2 * e * np.sin(arg))
    
    # dJ/dw2 = sum(-2 * e * w1 * cos(w2*x + w3) * x)
    grad_2 = -np.sum(2 * e * w[1] * np.cos(arg) * x)
    
    # dJ/dw3 = sum(-2 * e * w1 * cos(w2*x + w3))
    grad_3 = -np.sum(2 * e * w[1] * np.cos(arg))
    
    gradient_of_cost = np.array([grad_0, grad_1, grad_2, grad_3])
    
    # 更新權重
    w = w - alpha * gradient_of_cost

# 記錄解析法的結果曲線
xmin, xmax, ymin, ymax = scatter_pts_2d(x, y)
xt = np.linspace(xmin, xmax, 100)
yt1 = w[0] + w[1] * np.sin(w[2] * xt + w[3])


# ---------------------------------------------------------
# 第二部分：數值法 (Numeric Gradient Descent)
# 使用微小變化量 epsilon 估算梯度
# ---------------------------------------------------------
print("開始執行：數值法 (Numeric Method)")
w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])
epsilon = 1e-8  # 微小變化量

for _ in range(1, max_iters):
    # 1. 計算當前的 Cost
    arg = w[2] * x + w[3]
    y_pred = w[0] + w[1] * np.sin(arg)
    error = y - y_pred
    current_cost = np.sum(error ** 2)
    
    gradient_of_cost = np.zeros(4)
    
    # 2. 對每個權重 w[k] 分別做微擾，計算斜率
    for k in range(4):
        # 建立一個暫時的權重向量，將第 k 個分量 + epsilon
        w_temp = w.copy()
        w_temp[k] += epsilon
        
        # 計算新的 Cost
        arg_temp = w_temp[2] * x + w_temp[3]
        y_pred_temp = w_temp[0] + w_temp[1] * np.sin(arg_temp)
        cost_temp = np.sum((y - y_pred_temp) ** 2)
        
        # 數值微分公式: (f(x+h) - f(x)) / h
        gradient_of_cost[k] = (cost_temp - current_cost) / epsilon
    
    # 更新權重
    w = w - alpha * gradient_of_cost

# 記錄數值法的結果曲線
yt2 = w[0] + w[1] * np.sin(w[2] * xt + w[3])


# ---------------------------------------------------------
# 繪圖比較
# ---------------------------------------------------------
plt.figure(figsize=(10, 6), dpi=100)
plt.scatter(x, y, color='k', edgecolor='w', s=60, label='Data points')
plt.plot(xt, yt1, linewidth=4, c='b', alpha=0.6, label='Analytic method')
plt.plot(xt, yt2, linewidth=2, c='r', linestyle='--', label='Numeric method')

plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Gradient Descent: Analytic vs Numeric')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
