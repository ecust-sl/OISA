import json
import numpy as np

# 读取 JSON 文件
with open('/home/shilei/project/PromptMRG/data/mimic_cxr/FINAL2/SFT3/mimic_prefer_cliq_sft3_final.json', 'r') as file:
    data = json.load(file)

# 初始化变量
y_w_word_counts = []
y_l_word_counts = []

# 遍历 train 中的每个条目
for item in data.get('train', []):
    # 统计 y_w 字段的单词数
    if 'y_w' in item:
        y_w_text = item['y_w']
        y_w_word_counts.append(len(y_w_text.split()))

    # 统计 y_l 字段的单词数
    if 'y_l' in item:
        y_l_text = item['y_l']
        y_l_word_counts.append(len(y_l_text.split()))

# 计算统计数据
def calculate_stats(word_counts):
    if word_counts:
        average = np.mean(word_counts)
        median = np.median(word_counts)
        std_dev = np.std(word_counts)
        return average, median, std_dev
    return None, None, None

# 计算 y_w 和 y_l 的统计数据
average_y_w, median_y_w, std_y_w = calculate_stats(y_w_word_counts)
average_y_l, median_y_l, std_y_l = calculate_stats(y_l_word_counts)

# 输出统计结果
print(f"y_w 字段统计数据:")
print(f"平均单词数: {average_y_w}")
print(f"中位数: {median_y_w}")
print(f"标准差: {std_y_w}")

print(f"\ny_l 字段统计数据:")
print(f"平均单词数: {average_y_l}")
print(f"中位数: {median_y_l}")
print(f"标准差: {std_y_l}")
