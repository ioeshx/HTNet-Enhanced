import numpy as np
from sklearn.metrics import f1_score, recall_score

def calculate_uf1_uar(data):
    """
    计算 UF1 和 UAR 分数
    :param data: 字典格式的输入数据，包含 'pred' 和 'truth'
    :return: UF1 和 UAR 分数
    """
    all_f1_scores = []
    all_recall_scores = []
    
    # 获取所有类别
    all_classes = set()
    for subject in data.values():
        all_classes.update(subject['truth'])
    all_classes = sorted(all_classes)  # 确保类别有序

    # 针对每个类别计算 F1 和 Recall
    for cls in all_classes:
        y_true = []
        y_pred = []
        for subject in data.values():
            y_true.extend([1 if label == cls else 0 for label in subject['truth']])
            y_pred.extend([1 if label == cls else 0 for label in subject['pred']])
        
        # 计算 F1 和 Recall
        f1 = f1_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        all_f1_scores.append(f1)
        all_recall_scores.append(recall)
    
    # 计算 UF1 和 UAR
    uf1 = np.mean(all_f1_scores)
    uar = np.mean(all_recall_scores)
    
    return uf1, uar

# 示例数据
data = {'006': {'pred': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2]}, '007': {'pred': [0, 0, 1, 1, 1, 1, 0, 2], 'truth': [0, 1, 1, 1, 1, 1, 2, 2]}, '009': {'pred': [0, 0, 0, 1], 'truth': [0, 0, 0, 2]}, '010': {'pred': [0, 0, 0, 0], 'truth': [0, 0, 0, 0]}, '011': {'pred': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]}, '012': {'pred': [0, 0, 2], 'truth': [0, 0, 2]}, '013': {'pred': [0, 0, 0, 0, 0, 0], 'truth': [0, 0, 0, 0, 0, 0]}, '014': {'pred': [0, 0, 1, 1, 1, 1, 1, 1, 1, 2], 'truth': [0, 0, 1, 1, 1, 1, 1, 1, 1, 2]}, '015': {'pred': [0, 1, 2], 'truth': [0, 0, 2]}, '016': {'pred': [0, 0, 1, 2, 2], 'truth': [0, 0, 1, 2, 2]}, '017': {'pred': [0, 0, 0, 1], 'truth': [0, 0, 0, 2]}, '018': {'pred': [0, 0, 2], 'truth': [0, 0, 2]}, '019': {'pred': [1], 'truth': [1]}, '020': {'pred': [0, 0, 1, 0], 'truth': [0, 0, 1, 1]}, '021': {'pred': [0, 0], 'truth': [0, 0]}, '022': {'pred': [0, 0, 0, 1, 1], 'truth': [0, 0, 0, 1, 1]}, '023': {'pred': [0], 'truth': [0]}, '024': {'pred': [0], 'truth': [0]}, '026': {'pred': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 0]}, '028': {'pred': [0, 2, 2], 'truth': [0, 2, 2]}, '030': {'pred': [0, 0, 0], 'truth': [0, 0, 0]}, '031': {'pred': [0], 'truth': [0]}, '032': {'pred': [0, 0, 0, 0], 'truth': [0, 0, 0, 0]}, '033': {'pred': [0, 0, 0, 0, 2], 'truth': [0, 0, 0, 0, 1]}, '034': {'pred': [0, 0, 0], 'truth': [0, 0, 0]}, '035': {'pred': [0, 0, 0, 0, 0, 0, 0, 2], 'truth': [0, 0, 0, 0, 0, 0, 0, 2]}, '036': {'pred': [0], 'truth': [0]}, '037': {'pred': [0], 'truth': [0]}, 's01': {'pred': [0, 0, 0, 1, 1, 0], 'truth': [0, 0, 0, 1, 1, 2]}, 's02': {'pred': [1, 2, 2, 2, 2, 2], 'truth': [1, 2, 2, 2, 2, 2]}, 's03': {'pred': [1, 0, 2, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 2, 0, 1, 0, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 0, 0, 2, 2], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]}, 's04': {'pred': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]}, 's05': {'pred': [0, 0], 'truth': [0, 2]}, 's06': {'pred': [0, 1, 2, 2], 'truth': [0, 0, 2, 2]}, 's08': {'pred': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]}, 's09': {'pred': [1, 2, 2, 2], 'truth': [1, 2, 2, 2]}, 's11': {'pred': [0, 0, 0, 1, 1, 1, 2], 'truth': [0, 0, 0, 1, 1, 1, 2]}, 's12': {'pred': [1, 1, 1, 1, 1, 1, 1, 1, 2], 'truth': [1, 1, 1, 1, 1, 1, 1, 1, 2]}, 's13': {'pred': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'truth': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}, 's14': {'pred': [0, 0, 0, 1, 0, 2, 2, 2, 2, 2], 'truth': [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]}, 's15': {'pred': [0, 1, 0, 2], 'truth': [0, 1, 2, 2]}, 's18': {'pred': [0, 0, 2, 2, 2, 2, 2], 'truth': [0, 0, 2, 2, 2, 2, 2]}, 's19': {'pred': [1, 2], 'truth': [1, 2]}, 's20': {'pred': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 2, 0], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]}, 'sub01': {'pred': [0, 0, 1], 'truth': [0, 0, 1]}, 'sub02': {'pred': [0, 0, 0, 0, 0, 2, 2, 2, 2], 'truth': [0, 0, 0, 0, 0, 1, 2, 2, 2]}, 'sub03': {'pred': [0, 0, 0, 0, 2], 'truth': [0, 0, 0, 0, 2]}, 'sub04': {'pred': [0, 0], 'truth': [0, 0]}, 'sub05': {'pred': [1, 2, 2, 2, 2, 2], 'truth': [1, 2, 2, 2, 2, 2]}, 'sub06': {'pred': [0, 1, 2, 2], 'truth': [0, 1, 2, 2]}, 'sub07': {'pred': [0, 0, 0, 0, 0], 'truth': [0, 0, 0, 0, 0]}, 'sub08': {'pred': [0], 'truth': [0]}, 'sub09': {'pred': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 'truth': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]}, 'sub11': {'pred': [0, 0, 0, 0], 'truth': [0, 0, 0, 0]}, 'sub12': {'pred': [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2], 'truth': [0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2]}, 'sub13': {'pred': [1, 1], 'truth': [1, 1]}, 'sub14': {'pred': [1, 1, 1], 'truth': [1, 1, 1]}, 'sub15': {'pred': [0, 1, 2], 'truth': [0, 1, 2]}, 'sub16': {'pred': [0, 1, 1], 'truth': [0, 1, 1]}, 'sub17': {'pred': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1, 2], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2]}, 'sub19': {'pred': [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2], 'truth': [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]}, 'sub20': {'pred': [0, 0], 'truth': [0, 0]}, 'sub21': {'pred': [0], 'truth': [0]}, 'sub22': {'pred': [0, 0], 'truth': [0, 0]}, 'sub23': {'pred': [0, 0, 0, 0, 0, 0, 0, 1], 'truth': [0, 0, 0, 0, 0, 0, 0, 1]}, 'sub24': {'pred': [0, 0, 2], 'truth': [0, 0, 2]}, 'sub25': {'pred': [0, 0, 0, 2, 2], 'truth': [0, 0, 0, 2, 2]}, 'sub26': {'pred': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 'truth': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]}}


uf1, uar = calculate_uf1_uar(data)
print(f"UF1: {uf1:.4f}, UAR: {uar:.4f}")