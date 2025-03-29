import pandas as pd
#
# # Reload and process the correct CSV data file (you'll need to provide the correct file path if available)
# csv_path = 'data_green/cleared_sft_2.csv'  # Placeholder path for the actual CSV file
# data = pd.read_csv(csv_path)
# print(len(data))
# Add columns for the sum of cmn and mrg values
# data['sum_cmn'] = data[['cmn_cliq']].sum(axis=1)
# data['sum_mrg'] = data[['mrg_cliq']].sum(axis=1)

# import pandas as pd
#
# # 读取CSV文件
# csv_path = 'data_green/final_top_all.csv'  # 替换为实际CSV路径
# data = pd.read_csv(csv_path)
#
# # 计算三列的两两差值
# def calculate_differences_cliq(row):
#     differences = {
#         'dpo_cmn': abs(row['dpo_cliq'] - row['cmn_cliq']),
#         'dpo_mrg': abs(row['dpo_cliq'] - row['mrg_cliq']),
#         'cmn_mrg': abs(row['cmn_cliq'] - row['mrg_cliq'])
#     }
#     # 找出差值最大的组合
#     max_pair = max(differences, key=differences.get)
#     max_diff = differences[max_pair]
#     cols = max_pair.split('_')
#     y_w_value = min(row[f"{cols[0]}_cliq"], row[f"{cols[1]}_cliq"])  # 较大值
#     y_l_value = max(row[f"{cols[0]}_cliq"], row[f"{cols[1]}_cliq"])  # 较小值
#     y_w_source = cols[0] if row[f"{cols[0]}_cliq"] < row[f"{cols[1]}_cliq"] else cols[1]
#     y_l_source = cols[1] if row[f"{cols[0]}_cliq"] < row[f"{cols[1]}_cliq"] else cols[0]
#     return max_pair, max_diff, y_w_value, y_l_value, y_w_source, y_l_source
#
# def calculate_differences_f1(row):
#     differences = {
#         'dpo_cmn': abs(row['dpo_f1'] - row['cmn_f1']),
#         'dpo_mrg': abs(row['dpo_f1'] - row['mrg_f1']),
#         'cmn_mrg': abs(row['cmn_f1'] - row['mrg_f1'])
#     }
#     # 找出差值最大的组合
#     max_pair = max(differences, key=differences.get)
#     max_diff = differences[max_pair]
#     cols = max_pair.split('_')
#     y_w_value = max(row[f"{cols[0]}_f1"], row[f"{cols[1]}_f1"])  # 较大值
#     y_l_value = min(row[f"{cols[0]}_f1"], row[f"{cols[1]}_f1"])  # 较小值
#     y_w_source = cols[0] if row[f"{cols[0]}_f1"] > row[f"{cols[1]}_f1"] else cols[1]
#     y_l_source = cols[1] if row[f"{cols[0]}_f1"] > row[f"{cols[1]}_f1"] else cols[0]
#     max_f1 = max(max(row['cmn_f1'], row['mrg_f1']), row['dpo_f1'])
#     return max_pair, max_diff, y_w_value, y_l_value, y_w_source, y_l_source, max_f1
#
# def calculate_differences_ratescore(row):
#     differences = {
#         'dpo_cmn': abs(row['dpo_ratescore'] - row['cmn_ratescore']),
#         'dpo_mrg': abs(row['dpo_ratescore'] - row['mrg_ratescore']),
#         'cmn_mrg': abs(row['cmn_ratescore'] - row['mrg_ratescore'])
#     }
#     # 找出差值最大的组合
#     max_pair = max(differences, key=differences.get)
#     max_diff = differences[max_pair]
#     cols = max_pair.split('_')
#     y_w_value = max(row[f"{cols[0]}_ratescore"], row[f"{cols[1]}_ratescore"])  # 较大值
#     y_l_value = min(row[f"{cols[0]}_ratescore"], row[f"{cols[1]}_ratescore"])  # 较小值
#     y_w_source = cols[0] if row[f"{cols[0]}_ratescore"] > row[f"{cols[1]}_ratescore"] else cols[1]
#     y_l_source = cols[1] if row[f"{cols[0]}_ratescore"] > row[f"{cols[1]}_ratescore"] else cols[0]
#     max_ratescore = max(max(row['cmn_ratescore'], row['mrg_ratescore']), row['dpo_ratescore'])
#     return max_pair, max_diff, y_w_value, y_l_value, y_w_source, y_l_source, max_ratescore
# # 应用函数到每一行
# data[['max_pair', 'max_diff', 'y_w_f1', 'y_l_f1', 'y_w_source', 'y_l_source', 'max_ratescore']] = data.apply(
#     lambda row: pd.Series(calculate_differences_ratescore(row)),
#     axis=1
# )
#
# # 根据 y_w_source 和 y_l_source 找到对应的报告列值
# report_mapping = {
#     'cmn': 'report_cmn',
#     'mrg': 'report_mrg',
#     'dpo': 'report_dpo'
# }
#
# data['y_w'] = data.apply(lambda row: row[report_mapping[row['y_w_source']]], axis=1)
# data['y_l'] = data.apply(lambda row: row[report_mapping[row['y_l_source']]], axis=1)
# #
# filtered_data = data[data['max_ratescore'] >= 0.5]
# # 按照差值从大到小排序，同时在 max_diff 相同的情况下，按照 y_w 的 cliq 值（即来源列的值）从小到大排序
# filtered_data = filtered_data.sort_values(
#     by=['max_ratescore'],  # 按 `max_diff` 降序，再按 `y_w` 的 cliq 值升序
#     ascending=[False]
# )
# print(f'len:{len(filtered_data)}')
# # # 保存结果
# output_path = 'sorted_filter_10000_ratescore_2.csv'  # 输出文件路径
#
#
# filtered_data = filtered_data.drop_duplicates(subset=['y_w', 'y_l'], keep='first')
# print(f'len:{len(filtered_data)}')
# filtered_data = filtered_data.head(10000)
# #
# filtered_data.to_csv(output_path, index=False, encoding='utf-8-sig')
# print(f"处理完成，结果已保存到 {output_path}")
#
# #
# print('save successful!!!')
# Include additional requested columns (cmn_f1, cmn_cliq, mrg_f1, mrg_cliq)
# columns_to_save = [
#     'id', 'cmn_b1', 'cmn_b4', 'cmn_rg', 'mrg_b1', 'mrg_b4', 'mrg_rg',
#     'cmn_f1', 'cmn_cliq', 'mrg_f1', 'mrg_cliq',
#     'report_cmn', 'report_mrg', 'report_gt',
#     'y_w', 'y_l', 'y_w_source', 'y_l_source'
# ]
# output_data = top_10000_data[columns_to_save]
#
# # Save the final result to an Excel file
# output_path = 'process_data/filtered_data_with_sources_50000_nlg.xlsx'
# output_data.to_excel(output_path, index=False)
#
# print('save successful!!!')

# EOF json文件保存
# Load the JSON file
import json
import pandas as pd
json_path = '/ssd/shilei/dataset/annotations/mimic_annotation_promptmrg.json'  # Placeholder for JSON file path
with open(json_path, 'r') as f:
    json_data = json.load(f)

# Load the Excel file with filtered data
excel_path = '/home/shilei/project/PromptMRG/data_green/GPT-4o/top_10000_green_rows.csv'  # Placeholder for Excel file path
excel_data = pd.read_csv(excel_path)
print(len(excel_data))
# Extract the IDs from the Excel file
excel_ids = excel_data['id'].tolist()
# print(excel_ids)
# Convert Excel data to a dictionary with id as the key
excel_dict = excel_data.set_index('id').to_dict(orient='index')
print(len(json_data['train']))
# Filter JSON data to only include entries with matching IDs
filtered_json = {
    "train": [
        {**entry, **excel_dict[entry['id']]}  # Merge JSON entry with Excel columns
        for entry in json_data['train']
        if entry["id"] in excel_ids
    ],
    "test": json_data.get("test", []),
    "val": json_data.get("val", [])
}

print(f'len:{len(filtered_json["train"])}')
# Save the filtered JSON to a new file
output_json_path = 'data/mimic_cxr/FINAL2/GPT-4o/mimic_prefer_green_gpt_final.json'
with open(output_json_path, 'w') as f:
    json.dump(filtered_json, f, indent=4)

