import json
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
# 假设你的 JSON 文件路径是这样的
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
path_green = "extracted_green_data.csv"
path_green_medversa = "data_green/GPT-4o-Report/medversa_random20000_processed.csv"
medversa_data = pd.read_csv(path_green_medversa)
data = pd.read_csv(path_green)
d1 = data['y_w_green_sft1']
d2 = data['y_w_green_sft2']
d3 = data['y_w_green_sft3']
d4_green = medversa_data['medversa_green'].nlargest(10000)
d4_f1 = medversa_data['radgraph_combined'].nlargest(10000)
# 提取前10000小的 cxr_metric_score 数据（升序取前））
d4_cliq = medversa_data['cxr_metric_score'].nsmallest(10000)
for x in d1:
    if x > 1:
        print(x)
        print('error!!!')
for x in d2:
    if x > 1:
        print(x)
        print('error!!!')
for x in d3:
    if x > 1:
        print(x)
        print('error!!!')
# （可选）超出0.6的部分保持原值不调整



path1_f1 = "data_green/FINAL2/mimic_prefer_f1_sft1_final.json"
path2_f1 = "data_green/FINAL2/SFT2/mimic_prefer_f1_sft2_final.json"
path3_f1 = "data_green/FINAL2/SFT3/mimic_prefer_f1_sft3_final.json"
# path4_f1 = "data_green/FINAL2/GPT-4o/mimic_prefer_f1_gpt_final.json"
json_files_f1 = [path1_f1, path2_f1, path3_f1]

path1_cliq = "data_green/FINAL2/mimic_prefer_cliq_sft1_final.json"
path2_cliq = "data_green/FINAL2/SFT2/mimic_prefer_cliq_sft2_final.json"
path3_cliq = "data_green/FINAL2/SFT3/mimic_prefer_cliq_sft3_final.json"
# path4_cliq = "data_green/FINAL2/GPT-4o/mimic_prefer_cliq_gpt_final.json"
json_files_cliq = [path1_cliq, path2_cliq, path3_cliq]

# 创建一个空列表来存储从每个 JSON 文件提取的 y_w_f1, y_w_green 和 y_w_cliq 值
y_w_f1_values = []
y_w_green_values = []
y_w_cliq_values = []
y_w_green_values.append(d1)
y_w_green_values.append(d2)
y_w_green_values.append(d3)
# y_w_green_values.append(d4_green)


for idx, file in enumerate(json_files_f1):
    with open(file, 'r') as f:
        data = json.load(f)
        # 假设 train 是 JSON 里的一个键，且 y_w_f1, y_w_green 和 y_w_cliq 存在于每个 train 对象内
        y_w_f1 = [entry['y_w_f1'] for entry in data['train']]

        # 对值进行处理（增加偏移量）

        y_w_f1_values.append(y_w_f1)

for idx, file in enumerate(json_files_cliq):
    with open(file, 'r') as f:
        data = json.load(f)
        y_w_cliq = [entry['y_w_cliq'] for entry in data['train']]

        # 对值进行处理（增加偏移量）


        y_w_cliq_values.append(y_w_cliq)

print(y_w_cliq_values[2])
# y_w_f1_values.append(d4_f1)
# y_w_cliq_values.append(d4_cliq)
# 分别创建三个单独的图
# 图1: y_w_f1
# sns.violinplot(data=data, inner="box", kde_kws={'cut':0})
plt.figure(figsize=(6, 6))
sns.violinplot(data=y_w_f1_values, alpha=0.5,bw=0.5)
plt.xlabel('', fontsize=14)
plt.ylabel('RadGraphF1', fontsize=16)
plt.xticks([0, 1, 2], ['iteration1', 'iteration2', 'iteration3'], fontsize=14)
# plt.title('y_w_f1', fontsize=18)
plt.tight_layout()
plt.savefig("data_green/figure/f1.png", bbox_inches = 'tight', dpi = 600)
plt.show()

# 图2: y_w_green
plt.figure(figsize=(6, 6))
sns.violinplot(data=y_w_green_values, alpha=0.5, bw=0.5)
plt.xlabel('', fontsize=14)
plt.ylabel('GREEN', fontsize=16)
plt.xticks([0, 1, 2], ['iteration1', 'iteration2', 'iteration3'], fontsize=14)
# plt.title('y_w_green', fontsize=18)
plt.tight_layout()
plt.savefig("data_green/figure/green.png", bbox_inches = 'tight', dpi = 600)
plt.show()

# 图3: y_w_cliq
plt.figure(figsize=(6, 6))
sns.violinplot(data=y_w_cliq_values, alpha=0.5)
plt.xlabel('', fontsize=14)
plt.ylabel('RadCliQ', fontsize=16)
plt.xticks([0, 1, 2], ['iteration1', 'iteration2', 'iteration3'], fontsize=14)
# plt.title('y_w_cliq', fontsize=18)
plt.tight_layout()
plt.savefig("data_green/figure/cliq.png", bbox_inches = 'tight', dpi = 600)
plt.show()
