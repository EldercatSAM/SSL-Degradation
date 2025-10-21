import pandas as pd
import os
import numpy as np

def select_top_T_local_maxima(P, T = 3):
    """
    Select the top T indices of local maxima from list P.
    
    A local maximum is defined as an index i where P[i] is the maximum
    in the window [i-2, i+2]. Edge indices are handled by adjusting the window.
    
    Parameters:
    - P: List[float], list of real number values.
    - T: int, number of top indices to select from local maxima.
    
    Returns:
    - List[int], list of selected indices sorted in descending order of P[i].
    """
    C = []  # List to store indices of local maxima
    
    n = len(P)
    for i in range(n):
        # Define window boundaries
        start = max(0, i - 2)
        end = min(n, i + 3)  # end is non-inclusive in Python slicing
        
        window = P[start:end]
        current = P[i]
        
        if current == max(window):
            C.append(i)
    
    C_sorted = sorted(C, key=lambda x: (-P[x], x))
    
    # Select top T indices
    top_T_indices = C_sorted[:T]
    
    return top_T_indices

def read_metrics(excel_file_path):
    df = pd.read_excel(excel_file_path)
    if 'esvit' in excel_file_path or 'mec' in excel_file_path or 'simsiam' in excel_file_path:
        key = 'Layer 3 DSE Metric'
    else:
        key = 'Layer 11 DSE Metric'
    metric_values = df[key].tolist()
    return metric_values


if __name__ == '__main__':
    metric_path = '/path/to/your/DSE/results/'
    methods = ['dino-ori-800']

    methods = [x + '.xlsx' for x in methods]
    
    all_miou_diffs = []
    
    for _method in methods:
        method = _method.strip().split('.xlsx')[0]
        metrics = read_metrics(os.path.join(metric_path, _method))
        max_indices = select_top_T_local_maxima(np.array(metrics))
        max_indices = [(x + 1) * 10 for x in max_indices]
        
        print(method, max_indices)
