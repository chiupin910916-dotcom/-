# -*- coding: utf-8 -*-
"""
Modified for HW4 Solution (Fixed TypeError)
"""
import numpy as np

def scale_to_range(X: np.ndarray, to_range=(0,1), byrow = False):
    """
    Parameters
    ----------
    X: 
        1D or 2D array
     
    to_range: default to (0,1).
        Desired range of transformed data.
        
    byrow: default to False
        When working with a 2D array, true to perform row mapping; 
        otherwise, column mapping. Ignore if X is 1D. 
     
    ----------
    """
    # 確保輸入是浮點數格式
    X = np.array(X, dtype=np.float64)
    a, b = to_range
    
    # 判斷輸入是 1D 還是 2D 陣列
    if X.ndim == 1:
        min_x = np.min(X)
        max_x = np.max(X)
    else:
        if byrow:
            # Row-wise: axis=1, 保持維度以便廣播
            min_x = np.min(X, axis=1, keepdims=True)
            max_x = np.max(X, axis=1, keepdims=True)
        else:
            # Column-wise: axis=0
            min_x = np.min(X, axis=0, keepdims=True)
            max_x = np.max(X, axis=0, keepdims=True)
    
    # 計算範圍 (分母)
    diff = max_x - min_x
    
    # [修正點] 使用 np.where 來避免除以 0
    # 如果 diff == 0，就設為 1e-9，否則保持原值 diff
    #這行能同時處理 Scalar (1D) 和 Array (2D) 的情況
    diff = np.where(diff == 0, 1e-9, diff)
    
    # 套用公式
    Y = a + ((X - min_x) / diff) * (b - a)
    
    return Y

# --- 測試案例 ---

print('test case 1:')
A = np.array([1, 2.5, 6, 4, 5])
print(f'A => \n{A}')
print(f'scale_to_range(A) => \n{scale_to_range(A)}\n\n')

print('test case 2:')
A = np.array([[1,12,3,7,8],
              [5,14,1,5,5],
              [4,11,4,1,2],
              [3,13,2,3,5],
              [2,15,6,3,2]])
print(f'A => \n{A}')
print(f'scale_to_range(A) => \n{scale_to_range(A)}\n\n')

print('test case 3:')
A = np.array([[1,2,3,4,5],
              [5,4,1,2,3],
              [3,5,4,1,2]])
print(f'A => \n{A}')
print(f'scale_to_range(A, byrow=True) => \n{scale_to_range(A, byrow=True)}\n\n')