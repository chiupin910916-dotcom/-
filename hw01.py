import numpy as np
import numpy.linalg as la

def gram_schmidt(S1: np.ndarray):
    """
    Parameters
    ----------
    S1 : np.ndarray
        A m x n matrix with columns that need to be orthogonalized using Gram-Schmidt process.
        It is assumed that vectors in S = [v1 v2 ... vn] are linear independent.

    Returns
    -------
    S2 : np.ndarray
        S2 = [e1 e2 ... en] is a mxn orthogonal matrix such that span(S1)=span(S2)
    """
    
    # 取得矩陣大小: m 是維度(列數), n 是向量數量(行數)
    (m, n) = S1.shape
    S2 = np.zeros(S1.shape)

    for i in range(n):
        # 1. 從 S1 取出第 i 個向量 v (作為初始的 u)
        # 使用 copy() 避免改動到原矩陣
        v = S1[:, i].copy()
        
        # 2. 減去該向量在之前所有已計算出的基底向量上的投影分量
        # Gram-Schmidt 公式: u_i = v_i - sum( proj_{u_j}(v_i) ) for j < i
        for j in range(i):
            e_prev = S2[:, j] # 取出之前算好的正交單位向量
            
            # 計算投影量: (v dot e_prev)
            # 注意：因為下面的步驟確保 e_prev 已經是「單位向量(Normalized)」，
            # 所以投影公式簡化為內積即可，不需要除以長度平方。
            projection = np.dot(v, e_prev)
            
            # 減去投影向量
            v = v - projection * e_prev
        
        # 3. 正規化 (Normalization)
        # 將向量除以自身的長度(Norm)，使其成為單位向量 e
        norm_v = la.norm(v)
        
        # 簡單防呆：避免除以 0 (雖然題目假設線性獨立，但在數值計算中常加上此檢查)
        if norm_v > 1e-12:
            e = v / norm_v
        else:
            e = np.zeros_like(v)
            
        # 4. 將結果存入 S2 的第 i 行
        S2[:, i] = e

    return S2

# --- 測試區塊 (執行時會驗證結果) ---
if __name__ == "__main__":
    # 建立一個測試矩陣 (3x3，行向量線性獨立)
    A = np.array([[1.0, 1.0, 0.0],
                  [1.0, 0.0, 1.0],
                  [0.0, 1.0, 1.0]])
    
    print("原始矩陣 S1:")
    print(A)
    
    # 執行 Gram-Schmidt
    Q = gram_schmidt(A)
    
    print("\n正交化後的矩陣 S2 (Q):")
    print(Q)
    
    # 驗證性質：Q 的轉置乘以 Q 應該要等於單位矩陣 (Q.T @ Q = I)
    print("\n驗證正交性 (Q.T @ Q):")
    check = Q.T @ Q
    # 使用 np.round 讓顯示更乾淨
    print(np.round(check, 5))
