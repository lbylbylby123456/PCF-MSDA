import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

file_paths = ["2probabilities.xlsx", "3probabilities.xlsx", "4probabilities.xlsx"]
methods = ['iMSDA', 'SSD', 'MWSDTN','PCF-MSDA']

fpr_dict = {}
tpr_dict = {}
roc_auc_dict = {}


df = pd.read_excel("probabilities.xlsx")
y_score = df['Class 0']
y_true = df['labels'].apply(lambda x: 1 if x == 0 else 0)
# 计算 ROC 曲线
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# 存储结果
fpr_dict_method = fpr
tpr_dict_method = tpr
roc_auc_dict_method = roc_auc


for i, file in enumerate(file_paths):
    df = pd.read_excel(file)

    y_score = df['Class 1']
    y_true = df['labels'].apply(lambda x: 1 if x == 0 else 0)


    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fpr_dict[i] = fpr
    tpr_dict[i] = tpr
    roc_auc_dict[i] = roc_auc

plt.figure(figsize=(10, 8))

for i in range(len(file_paths)):
    plt.plot(fpr_dict[i], tpr_dict[i], label=f'{methods[i]} (AUC = {roc_auc_dict[i]:.2f})')

plt.plot(fpr_dict_method, tpr_dict_method, label=f'{methods[-1]} (AUC = {roc_auc_dict_method:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.legend(loc='lower right', fontsize=18)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
plt.show()
