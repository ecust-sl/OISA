import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ----------------------
# 1. 数据预处理阶段
# ----------------------
def process_data(file_path):
    # 读取Excel文件
    df = pd.read_excel(file_path)

    # 对两列进行log转换
    for col in ['RadGraphF1', 'CliQ']:
        df[f'{col}_processed'] = df[col]

    # 提取p1值，假设p1列已经存在于数据中
    if 'p1' not in df.columns:
        raise ValueError("p1列不存在，请确认数据中是否包含p1列")

    # 分割四个类别
    groups = {
        'I0': df[df['type'] == 'I0'],
        'I1': df[df['type'] == 'I1'],
        'I2': df[df['type'] == 'I2'],
        'I3': df[df['type'] == 'I3']
    }

    # 验证I0只有一行
    if len(groups['I0']) != 1:
        raise ValueError("I0类别应该只包含一行数据")

    return groups


# ----------------------
# 2. 可视化模块
# ----------------------
# 修改后的可视化核心代码
def find_pareto_front(df):
    pareto_indices = []
    for idx_i, row_i in df.iterrows():
        is_dominated = False
        for idx_j, row_j in df.iterrows():
            if idx_i == idx_j:
                continue

            j_dominates_i = (
                (row_j['RadGraphF1'] >= row_i['RadGraphF1'] and
                 row_j['CliQ'] <= row_i['CliQ']) and
                (row_j['RadGraphF1'] > row_i['RadGraphF1'] or
                 row_j['CliQ'] < row_i['CliQ'])
            )

            if j_dominates_i:
                is_dominated = True
                break

        if not is_dominated:
            pareto_indices.append(idx_i)

    return set(pareto_indices), df.loc[pareto_indices]


def plot_data(groups, output_path="output.pdf"):
    plt.figure(figsize=(10, 8))
    i0 = groups['I0']
    if i0.empty:
        raise ValueError("I0数据不存在")

    all_processed = pd.concat([g[['RadGraphF1_processed', 'CliQ_processed']] for g in groups.values()])
    max_x = all_processed['RadGraphF1_processed'].max()
    max_y = all_processed['CliQ_processed'].min()

    # 更新颜色设置，调整I2的颜色
    colors = [
        (1, 0.647, 0, 0.5),  # Orange
        (0.6, 0.8, 1, 0.8),  # Blue with higher opacity
        (0.8, 0.8, 0.9,0.8)  # Light Blue (adjusted I2 color)
    ]

    # 绘制I1-I3
    for i, (type_name, df) in enumerate([('Iteration1', groups['I1']),
                                         ('Iteration2', groups['I2']),
                                         ('Iteration3', groups['I3'])]):
        sorted_df = df.sort_values('RadGraphF1_processed').reset_index(drop=True)
        pareto_indices, pareto_df = find_pareto_front(sorted_df)

        # 绘制连线（只连接帕累托前沿点）
        if len(pareto_indices) >= 2:
            plt.plot(pareto_df['RadGraphF1_processed'],
                     pareto_df['CliQ_processed'],
                     color=colors[i],
                     linestyle='-',
                     alpha=0.5)

        # 散点参数调整，并增加点的大小
        scatter = plt.scatter(sorted_df['RadGraphF1_processed'],
                              sorted_df['CliQ_processed'],
                              color=colors[i],
                              s=100,  # 增大圆点的大小
                              edgecolors='w',
                              label=type_name)

        # 将p1值放到圆圈内部，调整字体大小
        for j, row in sorted_df.iterrows():
            plt.text(row['RadGraphF1_processed'], row['CliQ_processed'],
                     str(row['p1']),
                     fontsize=5.5, ha='center', va='center', color='black')

    # SFT点
    plt.scatter(i0['RadGraphF1_processed'],
                i0['CliQ_processed'],
                c='red',
                s=200,  # 增大圆点的大小
                marker='s',
                edgecolors='black',
                label='SFT')

    # Pareto点
    # plt.scatter(max_x, max_y,
    #             c='green',
    #             s=300,  # 增大圆点的大小
    #             marker='o',
    #             edgecolors='black',
    #             label='Pareto')

    # 新图例系统
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='SFT',
               markersize=12, markerfacecolor='red'),
        Line2D([0], [0], marker='o', color='w', label='Iteration1',
               markersize=10, markerfacecolor=colors[0]),
        Line2D([0], [0], marker='o', color='w', label='Iteration2',
               markersize=10, markerfacecolor=colors[1]),
        Line2D([0], [0], marker='o', color='w', label='Iteration3',
               markersize=10, markerfacecolor=colors[2]),
        # Line2D([0], [0], marker='o', color='w', label='Pareto',
        #        markersize=12, markerfacecolor='green')
    ]

    plt.gca().invert_yaxis()  # 翻转y轴
    plt.legend(handles=legend_elements, fontsize=10)
    plt.xlabel("RadGraphF1", fontsize=14)
    plt.ylabel("RadCliQ", fontsize=14)
    plt.savefig("data_green/figure/no-pareto/F1-CliQ-new.png", bbox_inches = 'tight', dpi = 600)
    plt.show()


# ----------------------
# 使用示例
# ----------------------
if __name__ == "__main__":
    # 替换为你的文件路径
    data_groups = process_data("data_green/GPT-4o-Report/F1-CliQ.xlsx")
    plot_data(data_groups)
